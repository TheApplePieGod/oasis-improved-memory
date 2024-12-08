import numpy as np
import torch
from torchvision.io import read_video, write_video
from utils import load_models, load_prompt, load_actions, sigmoid_beta_schedule, get_dataloader, save_state_dict, parse_img_size
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from torch import optim
from distutils.util import strtobool
import argparse
from pprint import pprint
from config import default_device
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler
from memory_bank import MemoryBank
import lpips
import time
import os


def train_vae(args):
    _, model = load_models(None, args.vae_ckpt, parse_img_size(args.default_img_size))
    model = model.train()
    model.requires_grad_(True)

    # TODO: good LR / optimizer. the scheduler does not appear to be working rn
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.epochs
    )

    train_loader = get_dataloader(
        args.batch,
        args.num_workers,
        data_dir=args.data_dir,
        image_size=(model.input_width, model.input_height),
        max_seq_len=1,
        load_actions=False,
        preload_videos=args.preload_videos
        #max_datapoints=1
    )

    lpips_loss_fn = lpips.LPIPS(net="alex").to(default_device)

    writer = SummaryWriter(f"logs/{args.vae_exp_name}/summary")
    for epoch in range(args.epochs):
        avg_train_loss = 0
        avg_recon_loss = 0
        avg_kl_loss = 0
        avg_lpip_loss = 0
        train_steps = 0
        prev_time = time.time()
        for X, _ in tqdm(train_loader):
            next_time = time.time()
            #tqdm.write(f"One batch: {(next_time - prev_time)*1000:.2f} ms")

            X = X.squeeze(1).to(default_device)
            optimizer.zero_grad()

            print("original x shape ", X.shape)
            X, x_prime, dist, latent = model.autoencode(X)
            print(X.shape, x_prime.shape)

            l1 = torch.nn.functional.mse_loss(x_prime, X, reduction='sum') / X.shape[0]
            dkl = (0.5 * torch.sum(dist.mean ** 2 + dist.var - dist.logvar - 1)) / X.shape[0] * args.kl_scale
            lp = lpips_loss_fn(x_prime, X).sum() / X.shape[0] * args.lpips_scale
            loss = l1 + dkl + lp

            loss.backward()

            avg_train_loss += loss.item()
            avg_recon_loss += l1.item()
            avg_kl_loss += dkl.item()
            avg_lpip_loss += lp.item()
            train_steps += 1

            optimizer.step()

            prev_time = time.time()

        lr_scheduler.step()

        avg_train_loss = avg_train_loss / train_steps
        avg_recon_loss = avg_recon_loss / train_steps
        avg_kl_loss = avg_kl_loss / train_steps
        avg_lpip_loss = avg_lpip_loss / train_steps

        writer.add_scalar("Train Loss", avg_train_loss, epoch)
        writer.add_scalar("Recon Loss", avg_recon_loss, epoch)
        writer.add_scalar("KL Loss", avg_kl_loss, epoch)
        writer.add_scalar("LPIP Loss", avg_lpip_loss, epoch)
        writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], epoch)

        writer.flush()

        print(f'Epoch: {epoch} Train Loss: {avg_train_loss:.4f}')

        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_state_dict(
                {
                    "epoch": epoch,
                    "vae_state_dict": model.state_dict(),
                    "input_width": model.input_width,
                    "input_height": model.input_height,
                    "model": model.name
                },
                f"logs/{args.vae_exp_name}/ckpt",
                f"model_{epoch}.pt"
            )

    writer.close()


def train_dit(args):
    model, vae = load_models(
        args.dit_ckpt,
        args.vae_ckpt,
        default_img_size=parse_img_size(args.default_img_size),
        dit_use_mem=args.use_memory
    )
    model = model.train()
    model.requires_grad_(True)
    if vae:
        vae = vae.eval()
        img_size = (vae.input_width, vae.input_height)
    else:
        img_size = (model.input_w, model.input_h)

    # params
    max_timesteps = 1000
    n_prompt_frames = model.max_frames
    if args.use_memory:
        # Double the sequence length so we can prepopulate the mem bank
        n_prompt_frames *= 2

    # get alphas
    betas = sigmoid_beta_schedule(max_timesteps).float().to(default_device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    cum_snr_decay = 0.96
    noise_clip = 6.0
    snr_clip = 5.0
    snr_arr = alphas_cumprod / (1 - alphas_cumprod)
    clipped_snr_arr = snr_arr.clamp(max=snr_clip)

    def forward_sample(x_0, t, e):
        alphabar_t = alphas_cumprod[t]
        return torch.sqrt(alphabar_t) * x_0 + torch.sqrt(1 - alphabar_t) * e

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    #optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=2e-3, betas=[0.9, 0.99])

    train_loader = get_dataloader(
        args.batch,
        args.num_workers,
        data_dir=args.data_dir,
        image_size=img_size,
        max_seq_len=n_prompt_frames,
        preload_json=args.preload_json,
        preload_videos=args.preload_videos
        #max_datapoints=100
    )

    if args.use_memory:
        memory_embedder = lambda x: torch.flatten(x, start_dim=2)

    writer = SummaryWriter(f"logs/{args.dit_exp_name}/summary")
    for epoch in range(args.epochs):
        avg_train_loss = 0
        train_steps = 0
        prev_time = time.time()
        for X, A in tqdm(train_loader):
            next_time = time.time()
            #tqdm.write(f"One batch: {(next_time - prev_time)*1000:.2f} ms")

            optimizer.zero_grad()
            X = X.to(default_device)
            A = A.to(default_device)

            B, T = X.shape[:2]

            X_pre_vae = X
            X = rearrange(X, "b t c h w -> (b t) c h w")
            with torch.no_grad():
                with autocast(default_device, dtype=torch.half):
                    X = vae.encode(X).sample() * args.vae_scale
            X = rearrange(X, "(b t) (h w) c -> b t c h w", t=T, h=vae.seq_h, w=vae.seq_w)

            if args.use_memory:
                # Populate memory bank
                # TODO: reuse this? doesnt really work bc batch size can vary
                memory_bank = MemoryBank(model.memory_input_dim, batch_size=B)

                #mem_embeddings = memory_embedder(X_pre_vae)
                mem_embeddings = memory_embedder(X)
                m = []
                for s in range(T):
                    memory_bank.push(mem_embeddings[:, s])
                    if s >= T // 2:
                        m.append(memory_bank.get_snapshot())

                # Update T to only use the second half of the sequence, since we used
                # the first half just to populate the memory bank
                T = T // 2
                X = X[:, T:]
                A = A[:, T:]

            # Sample a batch of times for training
            t = torch.randint(0, max_timesteps, (B, T), dtype=torch.long, device=default_device)

            # Calculate the loss
            e = torch.randn_like(X)
            e = torch.clamp(e, -noise_clip, noise_clip)
            x_t = forward_sample(X, t, e)
            with autocast(default_device, dtype=torch.half, enabled=args.fp16):
                if args.use_memory:
                    e_pred = model(x_t, t, m, A)
                else:
                    e_pred = model(x_t, t, A)
            e_pred = torch.clamp(e_pred, -noise_clip, noise_clip)
            loss = torch.nn.functional.mse_loss(e, e_pred.float(), reduction="none")

            snr = snr_arr[t]
            clipped_snr = clipped_snr_arr[t]

            fused_snr = True
            if fused_snr:
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip
                cum_snr = torch.zeros_like(normalized_snr)
                for frame_idx in range(0, T):
                    if frame_idx == 0:
                        cum_snr[:, frame_idx] = normalized_clipped_snr[:, frame_idx]
                    else:
                        cum_snr[:, frame_idx] = cum_snr_decay * cum_snr[:, frame_idx - 1] + (1 - cum_snr_decay) * normalized_clipped_snr[:, frame_idx]

                cum_snr = torch.nn.functional.pad(cum_snr[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0)
                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_clipped_snr)
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                loss_weight = clipped_fused_snr / fused_snr
            else:
                loss_weight = clipped_snr / snr

            loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
            loss = (loss * loss_weight).mean()

            loss = loss.mean()

            loss.backward()

            avg_train_loss += loss.item()
            train_steps += 1

            optimizer.step()

            prev_time = time.time()

        avg_train_loss = avg_train_loss / train_steps

        writer.add_scalar("Train Loss", avg_train_loss, epoch)

        writer.flush()

        print(f'Epoch: {epoch} Train Loss: {avg_train_loss:.4f}')

        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_state_dict(
                {
                    "epoch": epoch,
                    "dit_state_dict": model.state_dict(),
                    "input_width": model.input_w,
                    "input_height": model.input_h,
                    "patch_size": model.patch_size,
                    "model": model.name
                },
                f"logs/{args.dit_exp_name}/ckpt",
                f"model_{epoch}.pt"
            )


def main(args):
    if args.train_vae:
        print("Training vae...")
        train_vae(args)
    if args.train_dit:
        print("Training dit...")
        train_dit(args)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--epochs",
        type=int,
        help="Number of train epochs",
        default=50,
    )
    parse.add_argument(
        "--batch",
        type=int,
        help="Train batch size",
        default=4,
    )
    parse.add_argument(
        "--ckpt-every",
        type=int,
        help="Number of epochs for every checkpoint save",
        default=5,
    )
    parse.add_argument(
        "--default-img-size",
        type=str,
        help="Default size of the input images, 'WxH'",
        default="320x160",
    )
    parse.add_argument(
        "--kl-scale",
        type=float,
        help="Scale for the KL-divergence term of the VAE",
        default=1e-6,
    )
    parse.add_argument(
        "--lpips-scale",
        type=float,
        help="Scale for the LPIPS term of the VAE",
        default=1e-1,
    )
    parse.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parse.add_argument(
        "--vae-scale",
        type=float,
        required=True,
        help="Scaling factor for transforming the VAE before DiT training",
    )
    parse.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers to assist with data loading",
    )
    parse.add_argument(
        "--preload-json",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether or not to preload the dataset JSON data to prevent parsing at runtime",
    )
    parse.add_argument(
        "--preload-videos",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether or not to preload the dataset video data to prevent loading at runtime",
    )
    parse.add_argument(
        "--fp16",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether or not to use half precision floats where possible",
    )
    parse.add_argument(
        "--use-memory",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Whether or not to use the memory bank for diffusion",
    )
    parse.add_argument(
        "--data-dir",
        type=str,
        help="Dataset directory",
        default="data",
    )
    parse.add_argument(
        "--train-vae",
        type=lambda x: bool(strtobool(x)),
        help="Should train the VAE",
        default=False,
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to VAE ckpt for DiT training",
        default=None
    )
    parse.add_argument(
        "--vae-exp-name",
        type=str,
        help="Name of the vae experiment",
        default="vae"
    )
    parse.add_argument(
        "--train-dit",
        type=lambda x: bool(strtobool(x)),
        help="Should train the diffusion transformer",
        default=False,
    )
    parse.add_argument(
        "--dit-ckpt",
        type=str,
        help="Path to DiT ckpt for resuming DiT training",
        default=None
    )
    parse.add_argument(
        "--dit-exp-name",
        type=str,
        help="Name of the dit experiment",
        default="dit"
    )

    args = parse.parse_args()
    print("train args:")
    pprint(vars(args))
    main(args)

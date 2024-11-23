import numpy as np
import torch
from torchvision.io import read_video, write_video
from utils import load_models, load_prompt, load_actions, sigmoid_beta_schedule, get_dataloader, save_state_dict, parse_img_size
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from torch import optim
import argparse
from pprint import pprint
from config import default_device
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler
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
        data_dir=args.data_dir,
        image_size=(model.input_width, model.input_height),
        max_seq_len=1,
        load_actions=False,
        #max_datapoints=1
    )

    lpips_loss_fn = lpips.LPIPS(net="alex").to(default_device)

    writer = SummaryWriter("logs/vae/summary")
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

            x_prime, dist, latent = model.autoencode(X)

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
                },
                "logs/vae/ckpt",
                f"model_{epoch}.pt"
            )

    writer.close()


def train_dit(args):
    model, vae = load_models(args.dit_ckpt, args.vae_ckpt, (0, 0))
    model = model.train()
    model.requires_grad_(True)

    # params
    max_timesteps = 1000
    n_prompt_frames = model.max_frames

    # get alphas
    betas = sigmoid_beta_schedule(max_timesteps).float().to(default_device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    cum_snr_decay = 0.98
    noise_clip = 20.0
    snr_clip = 5.0
    snr_arr = alphas_cumprod / (1 - alphas_cumprod)
    clipped_snr_arr = snr_arr.clamp(max=snr_clip)

    def forward_sample(x_0, t, e):
        alphabar_t = alphas_cumprod[t]
        return torch.sqrt(alphabar_t) * x_0 + torch.sqrt(1 - alphabar_t) * e

    # TODO: good LR / optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_loader = get_dataloader(
        args.batch,
        data_dir=args.data_dir,
        image_size=(vae.input_width, vae.input_height),
        max_seq_len=n_prompt_frames,
        #max_datapoints=10
    )

    writer = SummaryWriter("logs/dit/summary")
    for epoch in range(args.epochs):
        avg_train_loss = 0
        train_steps = 0
        for X, A in tqdm(train_loader):
            optimizer.zero_grad()
            X = X.to(default_device)
            A = A.to(default_device)

            B, T = X.shape[:2]
            scaling_factor = 0.96
            X = rearrange(X, "b t c h w -> (b t) c h w")
            with torch.no_grad():
                with autocast(default_device, dtype=torch.half):
                    X = vae.encode(X).sample() * scaling_factor
            X = rearrange(X, "(b t) (h w) c -> b t c h w", t=T, h=vae.seq_h, w=vae.seq_w)

            # Sample a batch of times for training
            #t_ctx = torch.full((B, T - 1), 0, dtype=torch.long, device=default_device)
            #t = torch.randint(0, max_timesteps, (B, 1), dtype=torch.long, device=default_device)
            #t = torch.cat([t_ctx, t], dim=1)
            t = torch.randint(0, max_timesteps, (B, T), dtype=torch.long, device=default_device)

            # Calculate the loss
            e = torch.randn_like(X)
            e = torch.clamp(e, -noise_clip, noise_clip)
            x_t = forward_sample(X, t, e)
            e_pred = model(x_t, t, A)
            e_pred = torch.clamp(e_pred, -noise_clip, noise_clip)
            loss = torch.nn.functional.mse_loss(e, e_pred, reduction="none")

            snr = snr_arr[t]
            clipped_snr = clipped_snr_arr[t]

            """
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
            """

            loss_weight = clipped_snr / snr
            loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))

            #loss = (loss * loss_weight).sum() / B
            loss = (loss * loss_weight).mean()

            loss = loss.mean()

            loss.backward()

            avg_train_loss += loss.item()
            train_steps += 1

            optimizer.step()

        avg_train_loss = avg_train_loss / train_steps

        writer.add_scalar("Train Loss", avg_train_loss, epoch)

        writer.flush()

        print(f'Epoch: {epoch} Train Loss: {avg_train_loss:.4f}')

        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_state_dict(
                {
                    "epoch": epoch,
                    "dit_state_dict": model.state_dict()
                },
                "logs/dit/ckpt",
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
        default="640x320",
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
        "--data-dir",
        type=str,
        help="Dataset directory",
        default="data",
    )
    parse.add_argument(
        "--train-vae",
        type=bool,
        help="Should train the VAE",
        default=False,
    )
    parse.add_argument(
        "--train-dit",
        type=bool,
        help="Should train the diffusion transformer",
        default=False,
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to VAE ckpt for DiT training",
        default="logs/vae/ckpt/model_4999.pt"
    )
    parse.add_argument(
        "--dit-ckpt",
        type=str,
        help="Path to DiT ckpt for resuming DiT training",
        default=None
    )

    args = parse.parse_args()
    print("train args:")
    pprint(vars(args))
    main(args)

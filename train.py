import numpy as np
import torch
from torchvision.io import read_video, write_video
from utils import load_models, load_prompt, load_actions, sigmoid_beta_schedule, get_dataloader, save_state_dict
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from torch import optim
import argparse
from pprint import pprint
from config import default_device
from torch.utils.tensorboard import SummaryWriter
import time
import os


def train_vae(args):
    _, model = load_models(None, None)
    model = model.train()

    # TODO: good LR / optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter("logs/vae/summary")
    train_loader = get_dataloader(args.data_dir, 1, args.batch)
    for epoch in range(args.epochs):
        train_loss = 0
        prev_time = time.time()
        for X in tqdm(train_loader):
            next_time = time.time()
            #tqdm.write(f"One batch: {(next_time - prev_time)*1000:.2f} ms")

            X = X.squeeze(1).to(default_device)
            optimizer.zero_grad()
            x_prime, dist, latent = model(X, None)

            l1 = torch.nn.functional.mse_loss(x_prime, X, reduction='sum')
            dkl = 0.5 * torch.sum(dist.mean ** 2 + dist.var - dist.logvar - 1)
            loss = l1 + dkl

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            prev_time = time.time()

        train_loss = train_loss / len(train_loader.dataset)
        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.flush()
        print(f'Epoch: {epoch} Train Loss: {train_loss:.4f}')

        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_state_dict(
                {
                    "epoch": epoch,
                    "vae_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                },
                "logs/vae/ckpt",
                f"model_{epoch}.pt"
            )

    writer.close()


def train_dit(args):
    model, vae = load_models(None, None)
    model = model.train()

    # params
    max_timesteps = 1000
    n_prompt_frames = 10

    # model.max_frames

    # get alphas
    betas = sigmoid_beta_schedule(max_timesteps).float().to(default_device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    def forward_sample(x_0, t, e):
        alphabar_t = alphas_cumprod[t.cpu()]
        print(alphabar_t.shape)
        return torch.sqrt(alphabar_t) * x_0 + torch.sqrt(1 - alphabar_t) * e

    # TODO: good LR / optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = get_dataloader(args.data_dir, n_prompt_frames, args.batch)
    for epoch in range(args.epochs):
        train_losses = []
        for X in tqdm(train_loader):
            optimizer.zero_grad()
            X = X.to(default_device)

            B = X.shape[0]
            H, W = X.shape[-2:]
            scaling_factor = 0.07843137255 # TODO: ?
            X = rearrange(X, "b t c h w -> (b t) c h w")
            with torch.no_grad():
                with autocast(default_device, dtype=torch.half):
                    X = vae.encode(X).sample() * scaling_factor
            X = rearrange(X, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)

            # Sample a batch of times for training
            t_ctx = torch.full((B, n_prompt_frames - 1), 0, dtype=torch.long, device=default_device)
            t = torch.randint(0, max_timesteps, (B, 1), dtype=torch.long, device=default_device)
            t = torch.cat([t_ctx, t], dim=1)

            # Calculate the loss
            e = torch.randn_like(X)
            x_t = forward_sample(X, t, e)
            e_pred = model(x_t, t)
            loss = torch.nn.functional.mse_loss(e, e_pred)

            # Gradient step
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print("Epoch: {} Loss: {}".format(epoch, np.mean(train_losses)))


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

    args = parse.parse_args()
    print("train args:")
    pprint(vars(args))
    main(args)

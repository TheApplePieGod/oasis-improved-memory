import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video, write_video
from utils import load_models, get_dataloader
from tqdm import tqdm
from einops import rearrange
from torch import autocast
import argparse
from pprint import pprint
from config import default_device
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_latents(model, i=0, j=1):
    image_width = model.input_width
    image_height = model.input_height
    latent_w = image_width // model.patch_size
    latent_h = image_height // model.patch_size
    grid_size = 10

    loader = get_dataloader("data", 1, 1)
    x_in = loader.dataset[0].to(default_device)
    with torch.no_grad():
        z = model.encode(x_in).mean
        #z = torch.rand((1, latent_h * latent_w, model.latent_dim)).to(default_device)
        x_out = model.decode(z)

    x_in = (rearrange(x_in, "1 c h w -> h w c") + 1) / 2
    x_out = (rearrange(x_out, "1 c h w -> h w c") + 1) / 2

    fig = plt.figure(figsize=(15, 15))
    fig.add_subplot(2, 1, 1)
    plt.imshow(x_in.cpu())
    fig.add_subplot(2, 1, 2)
    plt.imshow(x_out.cpu())
    plt.show()


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.mps.manual_seed(0)

    model, vae = load_models(None, None)
    vae_ckpt = torch.load(f"logs/vae/ckpt/{args.vae_ckpt}")
    vae.load_state_dict(vae_ckpt["vae_state_dict"])

    model = model.eval()
    vae = vae.eval()

    plot_latents(vae, 0, 0)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default=None
    )

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)

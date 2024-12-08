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


def plot_latents(model, args):
    image_width = model.input_width
    image_height = model.input_height
    latent_w = image_width // model.patch_size
    latent_h = image_height // model.patch_size

    loader = get_dataloader(
        1, 1,
        data_dir=args.data_dir,
        image_size=(image_width, image_height),
        max_seq_len=1,
        load_actions=False
    )

    num_images = 3
    fig = plt.figure()
    for i in range(num_images):
        x_in, _ = loader.dataset[i]
        x_in = x_in.to(default_device)
        with torch.no_grad():
            z = model.encode(x_in).mean
            #z = torch.rand((1, latent_h * latent_w, model.latent_dim)).to(default_device)
            x_out = model.decode(z)

        x_in = (rearrange(x_in, "1 c h w -> h w c") + 1) / 2
        x_out = (rearrange(x_out, "1 c h w -> h w c") + 1) / 2

        fig.add_subplot(num_images, 2, i * 2 + 1)
        plt.imshow(x_in.cpu())
        fig.add_subplot(num_images, 2, i * 2 + 2)
        plt.imshow(x_out.cpu())
    plt.show()


def compute_scaling_factor(model, args):
    all_latents = []
    loader = get_dataloader(
        4, 1,
        data_dir=args.data_dir,
        image_size=(model.input_width, model.input_height),
        max_seq_len=200,
        load_actions=False,
        #max_datapoints=100
    )
    with torch.no_grad():
        for X, _ in tqdm(loader):
            X = X.to(default_device)
            X = rearrange(X, "b t c h w -> (b t) c h w")
            latents = model.encode(X).sample()
            all_latents.append(latents.cpu())

    all_latents_tensor = torch.cat(all_latents)
    std = all_latents_tensor.std().item()
    normalizer = 1 / std
    print(f'{normalizer=}')


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.mps.manual_seed(0)

    # Test
    _, vae, _ = load_models(None, args.vae_ckpt, None, (0, 0))

    # Baseline
    #_, vae = load_models(None, f"./vit-l-20.safetensors", (0, 0))

    vae = vae.eval()

    plot_latents(vae, args)
    #compute_scaling_factor(vae, args)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default=None
    )
    parse.add_argument(
        "--data-dir",
        type=str,
        help="Dataset directory",
        default="data",
    )

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)

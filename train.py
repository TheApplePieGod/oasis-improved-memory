import numpy as np
import torch
from torchvision.io import read_video, write_video
from utils import load_models, load_prompt, load_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from torch import optim
import argparse
from pprint import pprint
from config import default_device
from dataset import OasisDataset
import torch.utils.data.dataloader as dataloader
import os


def train_vae(args):
    _, model = load_models(None, None)
    model = model.train()

    # TODO: good LR / optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = OasisDataset()
    train_loader = dataloader.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    for epoch in range(args.epochs):
        train_loss = 0
        for X in tqdm(train_loader):
            X = X.to(default_device)
            optimizer.zero_grad()
            x_prime, dist, latent = model(X, None)

            l1 = torch.nn.functional.mse_loss(x_prime, X, reduction='sum')
            dkl = 0.5 * torch.sum(dist.mean ** 2 + dist.var - dist.logvar - 1)
            loss = l1 + dkl

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {} Train Loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


def train_dit(args):
    model, vae = load_models(None, None)
    model = model.train()

    # get alphas
    max_timesteps = 1000
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

    # TODO
    """
    dataset = OasisDataset()
    train_loader = dataloader.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    for epoch in range(args.epochs):
        train_losses = []
        for X in tqdm(train_loader):
            optimizer.zero_grad()
            batch_size = X.shape[0]
            X = X.to(default_device)
            #X = vae.encode(X * 2 - 1)

            # TODO
            num_tokens = 1

            # Sample a batch of times for training
            t = torch.randint(0, max_timesteps, (batch_size, num_tokens), device=default_device).long()

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
    """


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

"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
"""

import math
import torch
from torch import nn
from torchvision.io import read_image, read_video
from torchvision.transforms.functional import resize
from einops import rearrange
from typing import Mapping, Sequence
from config import default_device
from safetensors.torch import load_model
from dit import DiT_models
from vae import VAE_models
from dataset import OasisDataset
from torch.utils.data import DataLoader
import os


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
VIDEO_EXTENSIONS = {"mp4"}


def load_prompt(path, video_offset=None, n_prompt_frames=1, size=(360, 640)):
    if path.lower().split(".")[-1] in IMAGE_EXTENSIONS:
        print("prompt is image; ignoring video_offset and n_prompt_frames")
        prompt = read_image(path)
        # add frame dimension
        prompt = rearrange(prompt, "c h w -> 1 c h w")
    elif path.lower().split(".")[-1] in VIDEO_EXTENSIONS:
        prompt = read_video(path, pts_unit="sec")[0]
        if video_offset is not None:
            prompt = prompt[video_offset:]
        prompt = prompt[:n_prompt_frames]
        prompt = rearrange(prompt, "t h w c -> t c h w")
    else:
        raise ValueError(f"unrecognized prompt file extension; expected one in {IMAGE_EXTENSIONS} or {VIDEO_EXTENSIONS}")
    assert prompt.shape[0] == n_prompt_frames, f"input prompt {path} had less than n_prompt_frames={n_prompt_frames} frames"
    prompt = resize(prompt, (size[1], size[0]))
    # add batch dimension
    prompt = rearrange(prompt, "t c h w -> 1 t c h w")
    prompt = prompt.float() / 255.0
    return prompt


def load_actions(path, action_offset=None):
    if path.endswith(".actions.pt"):
        actions = one_hot_actions(torch.load(path))
    elif path.endswith(".one_hot_actions.pt"):
        actions = torch.load(path, weights_only=True)
    else:
        raise ValueError("unrecognized action file extension; expected '*.actions.pt' or '*.one_hot_actions.pt'")
    if action_offset is not None:
        actions = actions[action_offset:]
    actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
    # add batch dimension
    actions = rearrange(actions, "t d -> 1 t d")
    return actions


def load_models(dit_ckpt, vae_ckpt, default_img_size):
    def load_vae(img_size):
        return VAE_models["vit-l-20-shallow-encoder"](
            input_width=img_size[0],
            input_height=img_size[1]
        )

    # load VAE checkpoint
    vae = None
    if vae_ckpt is not None:
        print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
        if vae_ckpt.endswith(".pt"):
            # Load size info from the checkpoint
            vae_ckpt = torch.load(vae_ckpt)
            vae = load_vae((vae_ckpt["input_width"], vae_ckpt["input_height"]))
            vae.load_state_dict(vae_ckpt["vae_state_dict"])
        elif vae_ckpt.endswith(".safetensors"):
            # Size for oasis pretrained weights
            vae = load_vae((640, 320))
            load_model(vae, vae_ckpt)

    if vae is None:
        # Populate size with defaults specified
        vae = load_vae(default_img_size)

    vae = vae.to(default_device)

    print(f"VAE has input dim {vae.input_width}x{vae.input_height}")

    # ------------

    model = DiT_models["DiT-S/2"](
        input_w=vae.seq_w,
        input_h=vae.seq_h
    )
    if dit_ckpt is not None:
        print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(dit_ckpt)}...")
        if dit_ckpt.endswith(".pt"):
            ckpt = torch.load(dit_ckpt)
            model.load_state_dict(ckpt["dit_state_dict"], strict=False)
        elif dit_ckpt.endswith(".safetensors"):
            load_model(model, dit_ckpt)

    model = model.to(default_device)

    return model, vae


def get_dataloader(batch, **kwargs):
    dataset = OasisDataset(**kwargs)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=True
    )


def save_state_dict(state_dict, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(state_dict, os.path.join(dir, filename))


def parse_img_size(size_str):
    split = size_str.strip().split("x")
    return int(split[0]), int(split[1])

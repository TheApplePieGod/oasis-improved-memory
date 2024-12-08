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
from dataset import OasisDataset, oasis_dataset_collate
from torch.utils.data import DataLoader
from tqdm import tqdm
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


def load_models(dit_ckpt, vae_ckpt, mem_encoder_ckpt, default_img_size, dit_use_mem=False):
    def load_vae(name, img_size):
        print(f"Loading VAE {name}")
        return VAE_models[name](input_width=img_size[0], input_height=img_size[1])

    # load VAE checkpoint
    vae = None
    if vae_ckpt is not None:
        print(f"Loading VAE from ckpt {os.path.abspath(vae_ckpt)}...")
        if vae_ckpt.endswith(".pt"):
            # Load size info from the checkpoint
            vae_ckpt = torch.load(vae_ckpt, map_location=default_device)
            if "model" in vae_ckpt:
                model_name = vae_ckpt["model"]
            else:
                model_name = "vit-l-20-shallow-encoder"
            vae = load_vae(model_name, (vae_ckpt["input_width"], vae_ckpt["input_height"]))
            vae.load_state_dict(vae_ckpt["vae_state_dict"])
        elif vae_ckpt.endswith(".safetensors"):
            # Size for oasis pretrained weights
            vae = load_vae("vit-l-20-shallow-encoder", (640, 320))
            load_model(vae, vae_ckpt)

    if vae is None:
        # Populate size with defaults specified
        #vae = load_vae("vit-l-small", default_img_size)
        vae = load_vae("vit-mim", (64, 64)) # for vit_mim this is downsampled image size

    vae = vae.to(default_device)
    print(f"VAE has input dim {vae.input_width}x{vae.input_height}")

    # ------------

    mem_encoder = None
    if mem_encoder_ckpt is not None:
        print(f"Loading mem embedder from ckpt {os.path.abspath(mem_encoder_ckpt)}...")
        # Load size info from the checkpoint
        mem_encoder_ckpt = torch.load(mem_encoder_ckpt, map_location=default_device)
        model_name = mem_encoder_ckpt["model"]
        mem_encoder = load_vae(model_name, (mem_encoder_ckpt["input_width"], mem_encoder_ckpt["input_height"]))
        mem_encoder.load_state_dict(mem_encoder_ckpt["vae_state_dict"])

        mem_encoder = mem_encoder.to(default_device)
        print(f"Mem encoder has input dim {mem_encoder.input_width}x{mem_encoder.input_height}")

    # ------------

    def load_dit(name, ckpt=None):
        print(f"Loading DiT {name}")
        if dit_use_mem:
            assert mem_encoder_ckpt is not None
            return DiT_models[name](
                input_w=vae.seq_w,
                input_h=vae.seq_h,
                in_channels=vae.latent_dim,
                patch_size=2,
                #memory_input_dim=vae.seq_w * vae.seq_h * vae.latent_dim
                memory_input_dim=mem_encoder.mem_dim
            )
        else:
            return DiT_models[name](
                input_w=vae.seq_w,
                input_h=vae.seq_h,
                in_channels=vae.latent_dim,
                patch_size=2
            )

    model = None
    if dit_ckpt is not None:
        print(f"Loading DiT from ckpt {os.path.abspath(dit_ckpt)}...")
        if dit_ckpt.endswith(".pt"):
            ckpt = torch.load(dit_ckpt, map_location=default_device)
            if "model" in ckpt:
                model_name = ckpt["model"]
            else:
                model_name = "DiT-S/2"
            model = load_dit(model_name, ckpt)
            model.load_state_dict(ckpt["dit_state_dict"], strict=False)
        elif dit_ckpt.endswith(".safetensors"):
            model = load_dit("DiT-S/2")
            load_model(model, dit_ckpt)

    if model is None:
        if dit_use_mem:
            model = load_dit("DiT-S/2-Small-MiT-MiM")
        else:
            model = load_dit("DiT-S/2-Small")

    model = model.to(default_device)

    print(f"DiT has input dim {model.input_w}x{model.input_h}, patch size {model.patch_size}")

    return model, vae, mem_encoder


def get_dataloader(batch, num_workers, **kwargs):
    dataset = OasisDataset(**kwargs)
    return DataLoader(
        dataset,
        collate_fn=oasis_dataset_collate,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
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

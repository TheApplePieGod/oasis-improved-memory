"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video, write_video
from utils import load_models, load_prompt, load_actions, sigmoid_beta_schedule, get_dataloader
from tqdm import tqdm
from einops import rearrange
from torch import autocast
import argparse
from pprint import pprint
from config import default_device
import os

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.mps.manual_seed(0)

    model, vae = load_models(args.oasis_ckpt, args.vae_ckpt, (0, 0))
    model = model.eval()
    vae = vae.eval()

    # sampling params
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_clip = 20
    stabilization_level = 15

    # get prompt image/video
    load_fixed_datapoint = False
    if load_fixed_datapoint:
        x = load_prompt(
            args.prompt_path,
            video_offset=args.video_offset,
            n_prompt_frames=n_prompt_frames,
            size=(vae.input_width, vae.input_height)
        )
        x = x * 2 - 1
        # get input action stream
        actions = load_actions(args.actions_path, action_offset=args.video_offset)[:, :total_frames]
    else:
        loader = get_dataloader(
            1,
            data_dir="data",
            image_size=(vae.input_width, vae.input_height),
            max_seq_len=total_frames,
            #max_datapoints=1
        )
        x, actions = loader.dataset[0]
        x = x.unsqueeze(0)
        actions = actions.unsqueeze(0)

    # sampling inputs
    x = x.to(default_device)
    actions = actions.to(default_device)

    # vae encoding
    B, T = x.shape[:2]
    H, W = x.shape[-2:]
    scaling_factor = 0.9791
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with autocast(default_device, dtype=torch.half):
            x = vae.encode(x).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=T, h=vae.seq_h, w=vae.seq_w)
    x = x[:, :n_prompt_frames]

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(default_device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # sampling loop
    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=default_device)
        chunk = torch.clamp(chunk, -noise_clip, +noise_clip)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            # set up noise values
            t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=default_device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=default_device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=default_device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # sliding window
            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            # get model predictions
            with torch.no_grad():
                with autocast(default_device, dtype=torch.half):
                    v = model(x_curr, t, actions[:, start_frame : i + 1])

            x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    # vae decoding
    x = rearrange(x, "b t c h w -> (b t) (h w) c").float()
    with torch.no_grad():
        x = (vae.decode(x / scaling_factor) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    # save video
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    write_video(args.output_path, x[0].cpu(), fps=args.fps)
    print(f"generation saved to {args.output_path}.")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="oasis500m.safetensors",
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default="vit-l-20.safetensors",
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=32,
    )
    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        default="sample_data/sample_image_0.png",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="sample_data/sample_actions_0.one_hot_actions.pt",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=None,
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        help="If the prompt is a video, how many frames to condition on.",
        default=1,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="video.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=10)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)

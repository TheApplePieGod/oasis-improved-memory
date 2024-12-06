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
from memory_embedder import memory_embedding_models
import time
import os


def test_embedder(name):
    model = memory_embedding_models[name](
        input_dim=64, # Memory dim
        input_seq_len=8, # Max memory vals
        output_dim=32, # Output condition dim
        frame_w=16, # Input width of latent frame
        frame_h=8, # Input height of latent frame
        frame_c=16 # Input dim of latent frame
    )

    m = [
        torch.randn((7, 64)),
        torch.randn((3, 64))
    ]
    f = torch.randn((2, 4, 16, 8, 16))
    out = model(m, f)
    print(f"{name}: {out.shape} (Frame included)")
    out = model(m, None)
    print(f"{name}: {out.shape} (Frame excluded)")


if __name__ == "__main__":
    test_embedder("linear")
    test_embedder("mit")

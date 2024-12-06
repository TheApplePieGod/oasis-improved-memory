import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange
from typing import Optional
from attention import SpatialAxialAttention, TemporalAxialAttention
from timm.models.vision_transformer import Mlp
from timm.layers.helpers import to_2tuple
from dit import PatchEmbed
import math


def pad_sequences(seqs: list[torch.tensor], min_seq_len=0):
    """
    Pad a list of tensors to be grouped in a batch.
    Returns the padded tensor and the mask tensor
    """
    # Dim is (T, *D)
    max_seq = max(max(x.shape[0] for x in seqs), min_seq_len)

    padded_sequences = []
    masks = []
    for s in seqs:
        padding_len = max_seq - s.shape[0]
        padded = torch.cat([torch.zeros((padding_len, *s.shape[1:])), s], dim=0)
        mask = torch.cat([torch.zeros(padding_len), torch.ones(s.shape[0])])
        padded_sequences.append(padded)
        masks.append(mask)

    return (torch.stack(padded_sequences), torch.stack(masks))


class LinearEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_seq_len: int,
        output_dim: int,
        # Unused
        frame_w=0,
        frame_h=0,
        frame_c=0,
    ):
        super().__init__()

        self.input_seq_len = input_seq_len

        self.linear = nn.Linear(input_dim * input_seq_len, output_dim)

    def forward(self, m, f):
        # [T D] * B
        m, _ = pad_sequences(m, min_seq_len=self.input_seq_len)
        m = rearrange(m, "B T D -> B (T D)")
        m = self.linear(m)
        m = torch.nn.functional.normalize(m, p=2, dim=1)
        return m


class FMT(nn.Module):
    def __init__(
        self,
        token_dim,
        frame_h,
        frame_w,
        frame_c,
        frame_patch_size,
        memory_dim
    ):
        super().__init__()

        memory_embedding = 2 * torch.rand((1, token_dim)) - 1
        frame_embedding = 2 * torch.rand((1, token_dim)) - 1
        self.register_buffer('memory_embedding', memory_embedding)
        self.register_buffer('frame_embedding', frame_embedding)

        self.patch_embed = PatchEmbed(
            frame_h,
            frame_w,
            frame_patch_size,
            frame_c,
            token_dim,
            flatten=False
        )

        # Ideal when true
        self.memory_linear = None
        if memory_dim != token_dim:
            self.memory_linear = nn.Linear(memory_dim, token_dim)

    def memory_embed(self, m):
        if self.memory_linear is not None:
            m = self.memory_linear(m)
        m += self.memory_embedding
        return m

    def frame_embed(self, f):
        T = f.shape[1]
        f = rearrange(f, "b t c h w -> (b t) c h w")
        f = self.patch_embed(f) # -> B*T H/P W/P  D
        # TODO: add some sort of positional embeddings?
        # Rotary embeddings were being weird
        f = rearrange(f, "(b t) h w d -> b (t h w) d", t=T)
        return f

    def forward(self, m: torch.tensor, f: Optional[torch.tensor]):
        # m: B T1 C
        # f: B T2 C H W
        m = self.memory_embed(m) # -> B T1 D
        if f is not None:
            f = self.frame_embed(f) # -> B T1 D
            return torch.cat([m, f], dim=1)
        return m


class MiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_seq_len: int,
        output_dim: int,
        frame_w=16,
        frame_h=8,
        frame_c=8,
        frame_patch_size=8,
        token_dim=64,
        enc_depth=4,
        enc_heads=8,
        enc_dim=256
    ):
        super().__init__()

        self.fmt = FMT(
            token_dim=token_dim,
            frame_h=frame_h,
            frame_w=frame_w,
            frame_c=frame_c,
            frame_patch_size=frame_patch_size,
            memory_dim=input_dim
        )

        self.encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=enc_heads,
                dim_feedforward=enc_dim,
                batch_first=True
            )
            for _ in range(enc_depth)
        ])

    def forward(self, m: torch.tensor, f: Optional[torch.tensor]):
        # m: [T1 C] * B
        # f: B T2 C H W

        B = len(m)
        if f is not None:
            assert B == len(m)

        # Pad memory input since the length may vary for each elem
        # in the batch. We do not do this for f because we assume
        # f will always have a maximal amount of elements
        m, mask = pad_sequences(m)

        out = self.fmt(m, f) # -> B T1+T2 D

        # Add 1s for the frame portion of the mask
        mask = torch.cat([mask, torch.ones(B, out.shape[1] - m.shape[1])], dim=1)

        for enc in self.encoders:
            out = enc(out, src_key_padding_mask=mask)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        out = out[:, -1]

        return out


def MiT_Small(**kwargs):
    return MiT(
        **kwargs
    )


memory_embedding_models = {
    "linear": LinearEmbedder,
    "mit": MiT_Small,
}

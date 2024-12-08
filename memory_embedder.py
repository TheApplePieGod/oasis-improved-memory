import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange
from typing import Optional
from attention import SpatialAxialAttention, TemporalAxialAttention
from timm.models.vision_transformer import Mlp
from timm.layers.helpers import to_2tuple
from dit import PatchEmbed
from memory_bank import MemorySnapshot, pad_sequences
import math


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
        m = torch.nn.functional.normalize(m, p=2, dim=-1)
        m += self.memory_embedding
        return m

    def frame_embed(self, f):
        T = f.shape[1]
        f = rearrange(f, "b t c h w -> (b t) c h w")
        f = self.patch_embed(f) # -> B*T H/P W/P  D
        # TODO: add some sort of positional embeddings?
        # Rotary embeddings were being weird
        f = rearrange(f, "(b t) h w d -> b (t h w) d", t=T)
        f = torch.nn.functional.normalize(f, p=2, dim=-1)
        f += self.frame_embedding
        return f

    def forward(self, m: torch.tensor, f: Optional[torch.tensor]):
        # m: B C D
        # f: B T2 C H W
        m = self.memory_embed(m) # -> B C D
        if f is not None:
            f = self.frame_embed(f) # -> B T2 D
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

        self.linear = None
        if token_dim != output_dim:
            self.linear = nn.Linear(token_dim, output_dim)

        self.encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=enc_heads,
                dim_feedforward=enc_dim,
                batch_first=True
            )
            for _ in range(enc_depth)
        ])

    def forward(self, m: list[MemorySnapshot], f: Optional[torch.tensor], frame_seq_len: Optional[int]):
        # m: [Snapshot] * T
        # f: B T2 C H W

        tokens = []
        masks = []
        # Iterate in reverse since the last memory snapshot should have the most frame context available and
        # the size of the snapshots and frame context will likely differ
        for i, s in enumerate(reversed(m)):
            frame_ctx_padded = None
            if f is not None and frame_seq_len > 0:
                frame_ctx = f[:, -frame_seq_len-i:max(f.shape[1]-i, 0)]

                # Add 0s for missing frame context
                padding_len = frame_seq_len - frame_ctx.shape[1]
                zeros = torch.zeros((frame_ctx.shape[0], padding_len, *frame_ctx.shape[2:]), device=frame_ctx.device)
                frame_ctx_padded = torch.cat([zeros, frame_ctx], dim=1)

            tok = self.fmt(s.memory, frame_ctx_padded) # -> B T1+T2 D
            tokens.append(tok)

            if frame_ctx_padded is not None:
                # Add 1s for the frame portion of the mask and zeros for the padded part of the frame ctx
                tok_per_frame = (tok.shape[1] - s.memory.shape[1]) // frame_seq_len
                zeros = torch.zeros((f.shape[0], padding_len * tok_per_frame), device=tok.device)
                ones = torch.ones((f.shape[0], frame_ctx.shape[1] * tok_per_frame), device=tok.device)
                mask = torch.cat([s.mask, zeros, ones], dim=1)
            else:
                mask = s.mask
            masks.append(mask)

        # Reverse again so the tokens/masks are back in the correct sequence
        tokens = torch.vstack(list(reversed(tokens)))
        masks = torch.vstack(list(reversed(masks)))

        for enc in self.encoders:
            tokens = enc(tokens, src_key_padding_mask=masks)

        tokens = rearrange(tokens, "(b t) c d -> b t c d", t=len(m))
        tokens = tokens[:, :, -1]
        tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)

        if self.linear is not None:
            tokens = self.linear(tokens)

        return tokens


def MiT_Small(**kwargs):
    return MiT(
        **kwargs
    )


memory_embedding_models = {
    "linear": LinearEmbedder,
    "mit": MiT_Small,
}

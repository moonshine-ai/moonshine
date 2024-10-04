from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
import math
import torch

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t):
        freqs = torch.einsum("i , j -> i j", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1)
        return rearrange(freqs, "... d r -> ... (d r)")


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(t, freqs):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[-seq_len:, :]

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = t * freqs.cos() + rotate_half(t) * freqs.sin()
    out = torch.cat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, inner_dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    # Scaled dot product attention
    def sdp_attention(self, q, k_t, v, mask=None):
        d_tensor = v.shape[3]

        op = (q @ k_t) / math.sqrt(d_tensor)
        if mask is not None:
            op = op.masked_fill(mask, -torch.finfo(op.dtype).max)
        score = self.softmax(op)
        out = score @ v

        # concat and pass to linear layer
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def forward(self, q, k, v, rot_pos_emb=None, mask=None):
        # dot product with weight matrices
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_head)

        # apply RoPE
        if rot_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rot_pos_emb)
            k = apply_rotary_pos_emb(k, rot_pos_emb)

        k_t = k.transpose(2, 3)

        return self.sdp_attention(q, k_t, v, mask), k_t, v


class MultiHeadCausalSelfAttentionWithKVCache(MultiHeadAttention):
    def __init__(self, dim, inner_dim, n_head):
        super().__init__(dim, inner_dim, n_head)

    def forward(self, q, k, v, k_cache, v_cache, rot_pos_emb, mask):
        # dot product with weight matrices
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_head)

        # apply RoPE
        q = apply_rotary_pos_emb(q, rot_pos_emb)
        k = apply_rotary_pos_emb(k, rot_pos_emb)

        k_t = k.transpose(2, 3)

        # Append new rows to K and V caches.
        k_t = torch.concat((k_cache, k_t), dim=3)
        v = torch.concat((v_cache, v), dim=2)

        return super().sdp_attention(q, k_t, v, mask=mask), k_t, v


class MultiHeadCrossAttentionWithKVCache(MultiHeadAttention):
    def __init__(self, dim, inner_dim, n_head):
        super().__init__(dim, inner_dim, n_head)

    def forward(self, q, k_cache, v_cache):
        q = self.to_q(q)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)

        return super().sdp_attention(q, k_cache, v_cache)


class FFLinearGelu(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult, bias=True),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim, bias=True),
        )

    def forward(self, x):
        return self.ff(x)


class FFSwiGLU(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()

        self.ff_proj = nn.Linear(dim, dim * ff_mult * 2, bias=True)
        self.ff_act = nn.SiLU()
        self.ff_out = nn.Linear(dim * ff_mult, dim, bias=True)

    def forward(self, x):
        x, gate = self.ff_proj(x).chunk(2, dim=-1)
        x = x * self.ff_act(gate)
        return self.ff_out(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim, inner_dim, n_head, ff_swiglu, ff_mult=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, bias=False)

        self.attention = MultiHeadAttention(dim, inner_dim=inner_dim, n_head=n_head)

        self.norm2 = nn.LayerNorm(dim, bias=False)

        self.ff = FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)

    def forward(self, x, rot_pos_emb):
        _x = x
        x = self.norm1(x)
        x, _, _ = self.attention(q=x, k=x, v=x, rot_pos_emb=rot_pos_emb)
        x = x + _x

        _x = x
        x = self.norm2(x)
        x = self.ff(x)

        x = x + _x
        return x


class Encoder(nn.Module):
    def __init__(self, dim, inner_dim, n_head, n_layers, ff_swiglu):
        super().__init__()
        rot_embed_dim = max(inner_dim / n_head / 2, 32)
        self.rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        self.layers = nn.ModuleList(
            [EncoderLayer(dim, inner_dim, n_head, ff_swiglu) for _ in range(n_layers)]
        )
        self.post_norm = nn.LayerNorm(dim, bias=False)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        rot_pos_emb = self.rot_pos_emb(pos)

        for layer in self.layers:
            x = layer(x, rot_pos_emb=rot_pos_emb)
        return self.post_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, dim, inner_dim, n_head, ff_swiglu, ff_mult=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, bias=False)

        self.self_attention = MultiHeadCausalSelfAttentionWithKVCache(
            dim, inner_dim=inner_dim, n_head=n_head
        )

        self.norm2 = nn.LayerNorm(dim, bias=False)
        self.cross_attention = MultiHeadCrossAttentionWithKVCache(
            dim, inner_dim=inner_dim, n_head=n_head
        )

        self.norm3 = nn.LayerNorm(dim, bias=False)
        self.ff = FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)

    def forward(self, x, k_cache, v_cache, x_attn_k_cache, x_attn_v_cache, rot_pos_emb):
        dim = x.size()[1]
        causal_mask = torch.ones((dim, dim), dtype=torch.bool).triu(1).to(x.device)
        _x = x
        x = self.norm1(x)
        x, new_k_cache, new_v_cache = self.self_attention(
            q=x,
            k=x,
            v=x,
            k_cache=k_cache,
            v_cache=v_cache,
            rot_pos_emb=rot_pos_emb,
            mask=causal_mask,
        )
        x = x + _x

        _x = x
        x = self.norm2(x)
        x = self.cross_attention(q=x, k_cache=x_attn_k_cache, v_cache=x_attn_v_cache)
        x = x + _x

        _x = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + _x

        return x, new_k_cache, new_v_cache


class Decoder(nn.Module):
    def __init__(self, dim, inner_dim, n_head, n_layers, dec_voc_size, ff_swiglu):
        super().__init__()

        self.n_head = n_head
        self.d_head = inner_dim // n_head

        rot_embed_dim = max(inner_dim / n_head / 2, 32)
        self.rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        self.layers = nn.ModuleList(
            [DecoderLayer(dim, inner_dim, n_head, ff_swiglu) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(dim, bias=False)
        self.token_embedding = nn.Embedding(dec_voc_size, dim)

    def forward(self, x, *args):
        pos = torch.arange(x.shape[1], device=x.device)
        rot_pos_emb = self.rot_pos_emb(pos)
        x = self.token_embedding(x)

        k_cache_new = []
        v_cache_new = []

        n_layer = len(self.layers)
        k_cache, v_cache, x_attn_k_cache, x_attn_v_cache = [
            args[i : i + n_layer] for i in range(0, 4 * n_layer, n_layer)
        ]
        for idx, layer in enumerate(self.layers):
            x, new_k_line, new_v_line = layer(
                x[:, -1:],
                k_cache=k_cache[idx],
                v_cache=v_cache[idx],
                x_attn_k_cache=x_attn_k_cache[idx],
                x_attn_v_cache=x_attn_v_cache[idx],
                rot_pos_emb=rot_pos_emb,
            )
            k_cache_new.append(new_k_line)
            v_cache_new.append(new_v_line)

        x = self.final_norm(x)

        return x @ self.token_embedding.weight.t(), *k_cache_new, *v_cache_new


class InitialDecoderLayer(nn.Module):
    def __init__(self, dim, inner_dim, n_head, ff_swiglu, ff_mult=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, bias=False)

        self.self_attention = MultiHeadAttention(
            dim, inner_dim=inner_dim, n_head=n_head
        )

        self.norm2 = nn.LayerNorm(dim, bias=False)
        self.cross_attention = MultiHeadAttention(
            dim, inner_dim=inner_dim, n_head=n_head
        )

        self.norm3 = nn.LayerNorm(dim, bias=False)
        self.ff = FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)

    def forward(self, x, context, rot_pos_emb):
        dim = x.size()[1]
        causal_mask = torch.ones((dim, dim), dtype=torch.bool).triu(1).to(x.device)
        _x = x
        x = self.norm1(x)
        x, new_k_cache, new_v_cache = self.self_attention(
            q=x,
            k=x,
            v=x,
            rot_pos_emb=rot_pos_emb,
            mask=causal_mask,
        )
        x = x + _x

        _x = x
        x = self.norm2(x)
        x, x_attn_k_cache, x_attn_v_cache = self.cross_attention(
            q=x, k=context, v=context
        )
        x = x + _x

        _x = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + _x

        return x, new_k_cache, new_v_cache, x_attn_k_cache, x_attn_v_cache


class DecoderInitial(Decoder):
    def __init__(self, dim, inner_dim, n_head, n_layers, dec_voc_size, ff_swiglu):
        super().__init__(dim, inner_dim, n_head, n_layers, dec_voc_size, ff_swiglu)
        self.layers = nn.ModuleList(
            [
                InitialDecoderLayer(dim, inner_dim, n_head, ff_swiglu)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, enc_src):
        pos = torch.arange(x.shape[1], device=x.device)
        rot_pos_emb = self.rot_pos_emb(pos)
        x = self.token_embedding(x)

        # Shape [n_layers, batch_size, n_head, seq_len, inner_dim]. Cache K transposed.
        n_layer = len(self.layers)
        k_cache = []
        v_cache = []
        x_attn_k_cache = []
        x_attn_v_cache = []

        for idx, layer in enumerate(self.layers):
            x, new_k_line, new_v_line, new_x_attn_k_line, new_x_attn_v_line = layer(
                x,
                enc_src,
                rot_pos_emb,
            )

            k_cache.append(new_k_line)
            v_cache.append(new_v_line)
            x_attn_k_cache.append(new_x_attn_k_line)
            x_attn_v_cache.append(new_x_attn_v_line)

        x = self.final_norm(x)

        return (
            x @ self.token_embedding.weight.t(),
            *k_cache,
            *v_cache,
            *x_attn_k_cache,
            *x_attn_v_cache,
        )


class AudioPreprocessor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.audio_preprocess = nn.Sequential(
            nn.Conv1d(1, dim, 127, 64, bias=False),
            nn.Tanh(),
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, 2 * dim, 7, 3),
            nn.GELU(),
            nn.Conv1d(2 * dim, dim, 3, 2),
            nn.GELU(),
            Rearrange("... c s -> ... s c"),
        )

    def forward(self, src):
        assert (
            src.shape[-1] >= 1023
        ), f"src shape[-1] {src.shape[-1]} should be at least 1023"
        src = src.unsqueeze(-2)
        return self.audio_preprocess(src)


import time


class MoonshineModel(nn.Module):
    def __init__(
        self,
        dim,
        inner_dim,
        enc_depth,
        dec_depth,
        n_head=8,
        dec_voc_size=32768,
        enc_ff_swiglu=False,
        dec_ff_swiglu=False,
    ):
        super().__init__()
        self.preprocessor = AudioPreprocessor(dim)
        self.encoder = Encoder(
            dim, inner_dim, n_head, enc_depth, ff_swiglu=enc_ff_swiglu
        )
        self.decoder_initial = DecoderInitial(
            dim, inner_dim, n_head, dec_depth, dec_voc_size, ff_swiglu=dec_ff_swiglu
        )
        self.decoder = Decoder(
            dim, inner_dim, n_head, dec_depth, dec_voc_size, ff_swiglu=dec_ff_swiglu
        )
        self.dec_depth = dec_depth
        self.n_head = n_head
        self.d_head = inner_dim // n_head

    def generate(self, src, seq_len):
        start = time.time()
        preprocessed = self.preprocessor(src)
        start = time.time()
        enc = self.encoder(preprocessed)
        start = time.time()
        sot_token = 1
        eot_token = 2

        seq = torch.as_tensor([[sot_token]]).to(src.device)

        vals = self.decoder_initial(x=seq, enc_src=enc)
        logits = vals[0]
        k_cache, v_cache, x_attn_k_cache, x_attn_v_cache = [
            vals[i : i + self.dec_depth]
            for i in range(1, 1 + self.dec_depth * 4, self.dec_depth)
        ]

        start = time.time()

        sample = logits[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat((seq, sample), dim=-1)

        while sample != eot_token and len(seq.flatten()) <= seq_len:
            vals = self.decoder(
                seq,
                *k_cache,
                *v_cache,
                *x_attn_k_cache,
                *x_attn_v_cache,
            )
            logits = vals[0]
            k_cache = vals[1 : self.dec_depth + 1]
            v_cache = vals[self.dec_depth + 1 :]
            logits = logits[:, -1]  # get last token
            sample = logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat((seq, sample), dim=-1)

        return seq

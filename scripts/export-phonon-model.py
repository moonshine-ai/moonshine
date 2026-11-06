#!/usr/bin/env python3
# Copyright 2026 Useful Sensors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export Gradium's Phonon TTS to the ORT graphs the C++ ``PhononTTS`` engine runs.

Phonon (https://huggingface.co/gradium/phonon-31m-en) is a 31M-parameter
autoregressive text-to-speech model: a causal transformer backbone emits one
hidden state per 80 ms audio frame, a small flow-matching head turns that state
plus Gaussian noise into a 32-dim latent, and a Mimi-architecture decoder
(https://huggingface.co/gradium/phonon-mimi) turns latents into 24 kHz audio.
The voice is a prompt of precomputed speaker latents, so a voice is a 256 KB
file rather than a model.

Sources and licenses: the model and tokenizer are Gradium's, Apache 2.0; the
Mimi decoder is Gradium's build of Kyutai's Mimi codec, Apache 2.0; the voices
come from https://huggingface.co/gradium/phonon-voices, Apache 2.0. Only the
31M English checkpoint is exported: the 40M models are CC BY-NC 4.0. The
generation loop and text preparation the C++ follows are from Gradium's
``xn-ptts`` runtime (https://github.com/gradium-ai/xn-ptts, MIT or Apache 2.0).
The encoder that makes a voice file from a recording is not published;
Gradium's hosted Studio (https://studio.gradium.ai) provides cloning and voice
design, and its downloads load in the C++ engine unchanged.

The HF ``transformers`` implementation (>= 5.21) is the reference. The graphs
exported here are purely functional pieces of it, with the generation loop,
the KV cache, the noise and the end-of-speech countdown kept in C++
(``core/moonshine-tts/src/phonon-tts.cpp``):

  embed_prompt.onnx      (text_ids, voice_latents, duration_delta) -> prompt_embeds
  embed_latent.onnx      (latent, duration_delta) -> latent_embeds
  backbone.onnx          (inputs_embeds, past_k_*, past_v_*) -> condition, eos_logit, present_k_*, present_v_*
  flow_head.onnx         (latent, condition, t_start, t_end) -> velocity
  mimi_transformer.onnx  (latents, prev_hidden, start_position, past_k_*, past_v_*)
                             -> hidden, new_prev_hidden, present_k_*, present_v_*
  mimi_seanet.onnx       (hidden) -> audio            (24 kHz samples, 120 per hidden step)

The Mimi decoder is split where its state lives: the transformer half keeps a
sliding-window KV cache and the one-frame overlap of the upsampling
convolution, while the convolutional half has a receptive field under a frame
and is simply re-run over a short window.

Every graph is checked against the PyTorch modules it was cut from, then the
whole pipeline is run once end to end against ``model.generate`` with
``temperature=0`` (deterministic) and the audio compared. The ``.onnx`` files
are then converted to ``.ort`` with ``scripts/convert-models-to-ort.py`` and
written to ``--out-dir`` with the tokenizer and the English voices of
``gradium/phonon-voices``.

Usage (in a venv with torch, transformers>=5.21, onnx, onnxruntime, safetensors):

  python3 scripts/export-phonon-model.py --out-dir core/moonshine-tts/data/phonon
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
OPSET = 17

LM_REPO = "gradium/phonon-31m-en"
MIMI_REPO = "gradium/phonon-mimi"
VOICES_REPO = "gradium/phonon-voices"

GRAPHS = ("embed_prompt", "embed_latent", "backbone", "flow_head", "mimi_transformer", "mimi_seanet")


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """``positions`` [S] int64 -> cos, sin [1, S, head_dim] (HF "default" rope)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None], emb.sin()[None]


def attention(q, k, v, mask, scaling):
    """q [1,H,S,D], k/v [1,H,T,D], mask [S,T] bool (True = may attend)."""
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    neg = torch.full_like(scores, torch.finfo(scores.dtype).min)
    scores = torch.where(mask[None, None], scores, neg)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.transpose(1, 2).reshape(1, q.shape[2], -1)


# ---------------------------------------------------------------------------
# Backbone (flow LM) wrappers
# ---------------------------------------------------------------------------


class EmbedPrompt(nn.Module):
    """``[voice frames] [text tokens] [BOS]`` -> backbone input embeddings."""

    def __init__(self, lm):
        super().__init__()
        self.m = lm.model
        self.cfg = lm.config

    def condition_embeds(self, duration_delta):
        m, cfg = self.m, self.cfg
        num_speakers = torch.full((), cfg.num_speakers, dtype=torch.long)
        cond = m.num_speakers_proj(m.num_speakers_embed(num_speakers))
        half = cfg.duration_delta_embedding_dim // 2
        exponents = torch.arange(half, dtype=torch.float32) / (half - 1)
        phases = (cfg.duration_delta_scale * duration_delta) / (cfg.duration_delta_max_period**exponents)
        dur = torch.cat([phases.cos(), phases.sin()], dim=-1)
        return cond + m.duration_delta_proj(dur)

    def forward(self, text_ids, voice_latents, duration_delta):
        m = self.m
        voice = m.audio_prompt_proj(voice_latents)  # [1, Np, H]
        text = m.embed_tokens(text_ids)  # [1, Nt, H]
        bos = m.audio_latent_proj(m.bos_latent) + self.condition_embeds(duration_delta)
        return torch.cat([voice, text, bos[None, None, :]], dim=1)


class EmbedLatent(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.prompt = EmbedPrompt(lm)

    def forward(self, latent, duration_delta):
        m = self.prompt.m
        return (m.audio_latent_proj(latent) + self.prompt.condition_embeds(duration_delta))[:, None, :]


class Backbone(nn.Module):
    """One backbone call over ``S`` new positions with an explicit KV cache."""

    def __init__(self, lm):
        super().__init__()
        self.m = lm.model
        self.eos_head = lm.eos_head
        self.cfg = lm.config

    def forward(self, inputs_embeds, *past):
        cfg = self.cfg
        n_layers = cfg.num_hidden_layers
        heads, head_dim = cfg.num_attention_heads, cfg.head_dim
        S = inputs_embeds.shape[1]
        P = past[0].shape[2]
        positions = torch.arange(S, dtype=torch.long) + P
        cos, sin = rope_cos_sin(positions, head_dim, cfg.rope_parameters["rope_theta"])
        cos, sin = cos[:, None], sin[:, None]  # [1,1,S,D]
        # Causal over the concatenated [past, new] keys.
        q_idx = positions[:, None]
        k_idx = torch.arange(S, dtype=torch.long)[None, :] + P
        k_idx = torch.cat([torch.arange(P, dtype=torch.long)[None, :], k_idx], dim=1)
        mask = k_idx <= q_idx  # [S, P+S]

        h = inputs_embeds
        presents = []
        for i, layer in enumerate(self.m.layers[:n_layers]):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(1, S, heads, head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(1, S, heads, head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(1, S, heads, head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = torch.cat([past[2 * i], k], dim=2)
            v = torch.cat([past[2 * i + 1], v], dim=2)
            presents += [k, v]
            a = attention(q, k, v, mask, attn.scaling)
            h = h + attn.o_proj(a)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.m.norm(h)
        last = h[:, -1, :]
        eos = self.eos_head(last)[:, 0]
        return (last, eos, *presents)


class FlowHead(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.head = lm.flow_head

    def forward(self, latent, condition, t_start, t_end):
        times = [t_start.reshape(1, 1), t_end.reshape(1, 1)]
        return self.head(latent, condition, times=times)


# ---------------------------------------------------------------------------
# Mimi decoder wrapper
# ---------------------------------------------------------------------------


class MimiTransformer(nn.Module):
    """The streaming half of the Mimi decoder: normalized latents -> transformer hidden states.

    ``latents`` [1, T, 32] are denormalized, projected and upsampled 16x (a causal
    transposed convolution whose kernel spans the previous frame, carried in
    ``prev_hidden`` [1, 512, 1]), then run through the sliding-window causal
    transformer with an explicit KV cache. ``start_position`` is the absolute
    index of the first new step, so the caller may drop cache entries older than
    the window. Returns hidden [1, 512, 16T] for the SEANet half, the new
    ``prev_hidden`` and the present KV cache.
    """

    def __init__(self, lm, tokenizer):
        super().__init__()
        self.lm = lm
        self.tok = tokenizer
        self.dec = tokenizer.decoder
        self.cfg = tokenizer.decoder.config

    def forward(self, latents, prev_hidden, start_position, *past):
        cfg = self.cfg
        heads, head_dim = cfg.num_attention_heads, cfg.head_dim
        denorm = latents * self.lm.latent_std + self.lm.latent_mean
        x = self.tok.latent_proj(denorm.transpose(1, 2))  # [1, 512, T]
        new_prev = x[:, :, -1:]
        up = self.dec.upsample.conv(torch.cat([prev_hidden, x], dim=2))  # [1, 512, 16(T+1) + 16]
        stride = self.dec.upsample.conv.stride[0]
        # Drop the context frame's steps on the left and the causal trim on the right.
        up = up[:, :, stride : up.shape[2] - self.dec.upsample.padding_right]
        h = up.transpose(1, 2)  # [1, L, 512]
        L = h.shape[1]
        P = past[0].shape[2]
        positions = torch.arange(L, dtype=torch.long) + start_position
        cos, sin = rope_cos_sin(positions, head_dim, cfg.rope_parameters["rope_theta"])
        cos, sin = cos[:, None], sin[:, None]
        q_idx = positions[:, None]
        k_idx = torch.cat([torch.arange(P, dtype=torch.long) + (start_position - P), positions])[None, :]
        mask = (k_idx <= q_idx) & ((q_idx - k_idx) < cfg.sliding_window)
        presents = []
        for i, layer in enumerate(self.dec.decoder_transformer.layers):
            attn = layer.self_attn
            y = layer.input_layernorm(h)
            q = attn.q_proj(y).view(1, L, heads, head_dim).transpose(1, 2)
            k = attn.k_proj(y).view(1, L, heads, head_dim).transpose(1, 2)
            v = attn.v_proj(y).view(1, L, heads, head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = torch.cat([past[2 * i], k], dim=2)
            v = torch.cat([past[2 * i + 1], v], dim=2)
            presents += [k, v]
            a = attention(q, k, v, mask, attn.scaling)
            h = h + layer.self_attn_layer_scale(attn.o_proj(a))
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        return (h.transpose(1, 2), new_prev, *presents)


class FoldedConvTranspose(nn.Module):
    """A causal ``ConvTranspose1d`` with kernel ``2 * stride`` as a kernel-2 ``Conv1d`` plus a
    channel-to-time interleave.

    Output sample ``t = stride * n + p`` of the transposed convolution sums two taps,
    ``W[:, :, p]`` on input ``n`` and ``W[:, :, p + stride]`` on input ``n - 1``, so a kernel-2
    convolution over the input padded with one zero on the left, with ``stride * out_channels``
    output channels ordered ``(p, channel)``, computes the same numbers. ONNX Runtime runs Conv as
    a GEMM but ConvTranspose through a col2im path several times slower; these three layers were
    over half of the decoder's time.
    """

    def __init__(self, wrapped):
        super().__init__()
        ct = wrapped.conv
        cin, cout, k = ct.weight.shape
        stride = ct.stride[0]
        if (k != 2 * stride or ct.groups != 1 or ct.padding[0] != 0 or ct.dilation[0] != 1
                or wrapped.padding_left != 0 or wrapped.padding_right != stride):
            raise ValueError(f"cannot fold {wrapped}: expected causal, kernel 2 * stride, trimmed on the right")
        w = ct.weight.detach()  # [cin, cout, 2 * stride]
        taps = torch.stack([w[:, :, stride:], w[:, :, :stride]], dim=-1)  # [cin, cout, stride, 2]
        self.conv = nn.Conv1d(cin, stride * cout, 2, bias=ct.bias is not None)
        with torch.no_grad():
            self.conv.weight.copy_(taps.permute(2, 1, 0, 3).reshape(stride * cout, cin, 2))
            if ct.bias is not None:
                self.conv.bias.copy_(ct.bias.detach().repeat(stride))
        self.stride = stride
        self.out_channels = cout

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (1, 0)))  # [B, stride * cout, L]
        b, _, length = y.shape
        y = y.view(b, self.stride, self.out_channels, length).permute(0, 2, 3, 1)
        return y.reshape(b, self.out_channels, length * self.stride)


def fold_conv_transposes(seanet):
    """A copy of the SEANet decoder with every transposed convolution folded."""
    import copy

    from transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer import (
        PocketTTSAudioTokenizerConvTranspose1d,
    )

    folded = copy.deepcopy(seanet)
    for i, layer in enumerate(folded.layers):
        if isinstance(layer, PocketTTSAudioTokenizerConvTranspose1d):
            folded.layers[i] = FoldedConvTranspose(layer)
    return folded


class MimiSeanet(nn.Module):
    """The convolutional half: transformer hidden [1, 512, L] -> audio [1, L * 120].

    Every convolution is causal with a receptive field well under one frame
    (16 steps), so a caller streams it by re-running a window that starts
    ``kMimiSeanetContextFrames`` frames early and dropping those frames' samples.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.seanet = fold_conv_transposes(tokenizer.decoder.decoder_seanet)
        self.hop = 1
        for r in tokenizer.decoder.config.upsampling_ratios:
            self.hop *= r

    def forward(self, hidden):
        audio = self.seanet(hidden)
        return audio[:, 0, : hidden.shape[2] * self.hop]


class MimiDecoder(nn.Module):
    """Both halves in one call, for the checks: normalized latents -> audio."""

    def __init__(self, lm, tokenizer):
        super().__init__()
        self.transformer = MimiTransformer(lm, tokenizer)
        self.seanet = MimiSeanet(tokenizer)
        cfg = tokenizer.decoder.config
        self.frame_size = int(round(cfg.sampling_rate / lm.config.frame_rate))

    def forward(self, latents):
        cfg = self.transformer.cfg
        prev = torch.zeros(1, cfg.hidden_size, 1)
        past = [torch.zeros(1, cfg.num_attention_heads, 0, cfg.head_dim) for _ in range(2 * cfg.num_hidden_layers)]
        h = self.transformer(latents, prev, torch.zeros((), dtype=torch.long), *past)[0]
        return self.seanet(h)


# ---------------------------------------------------------------------------
# Export + checks
# ---------------------------------------------------------------------------


def export(module, args, path: Path, input_names, output_names, dynamic_axes):
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    import onnx

    onnx.checker.check_model(str(path))
    print(f"  exported {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def ort_session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run(sess, **feeds):
    feeds = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in feeds.items()}
    return sess.run(None, feeds)


def check(name, got, want, tol):
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    if got.shape != want.shape:
        raise SystemExit(f"{name}: shape {got.shape} != {want.shape}")
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    scale = float(np.max(np.abs(want))) if want.size else 1.0
    status = "ok" if err <= tol * max(scale, 1e-3) else "MISMATCH"
    print(f"  {name}: max abs err {err:.3e} (scale {scale:.3e}) {status}")
    if status != "ok":
        raise SystemExit(f"{name} parity check failed")


def load_models(dtype=torch.float32):
    from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

    proc = AutoProcessor.from_pretrained(LM_REPO)
    lm = GradiumPhononTTSForConditionalGeneration.from_pretrained(LM_REPO, dtype=dtype).eval()
    tok = proc.audio_tokenizer.eval()
    return proc, lm, tok


def load_voice(path: Path) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(str(path))["speaker_wavs"].float().transpose(1, 2).contiguous()  # [1, T, 512]


#: Frames of already-decoded hidden states re-run ahead of each SEANet window.
#: The convolutions' combined receptive field is under 16 steps (one frame), so
#: one frame reproduces a whole decode to 1e-4 relative L1 and none does not.
#: Each extra frame costs about 2 ms per generated frame on an M4, so no margin.
SEANET_CONTEXT_FRAMES = 1


class MimiStreamer:
    """The C++ streaming decode, in numpy: transformer with a KV cache, SEANet windowed."""

    def __init__(self, transformer_sess, seanet_sess, meta):
        self.tr = transformer_sess
        self.se = seanet_sess
        self.heads = meta["mimi_heads"]
        self.head_dim = meta["mimi_head_dim"]
        self.layers = meta["mimi_transformer_layers"]
        self.stride = meta["mimi_latent_stride"]
        self.window = meta["mimi_sliding_window"]
        self.hidden = meta["mimi_hidden_size"]
        self.hop = meta["frame_size"] // self.stride
        self.reset()

    def reset(self):
        self.prev = np.zeros((1, self.hidden, 1), dtype=np.float32)
        self.past = [np.zeros((1, self.heads, 0, self.head_dim), dtype=np.float32) for _ in range(2 * self.layers)]
        self.position = 0
        self.history = np.zeros((1, self.hidden, 0), dtype=np.float32)  # transformer output, last few frames

    def decode(self, latents):
        """latents [1, T, 32] normalized -> audio [T * frame_size]."""
        T = latents.shape[1]
        names = [f"past_{kv}_{i}" for i in range(self.layers) for kv in ("k", "v")]
        outs = run(self.tr, latents=latents, prev_hidden=self.prev,
                   start_position=np.array(self.position, dtype=np.int64), **dict(zip(names, self.past)))
        hidden, self.prev = outs[0], outs[1]
        keep = self.window - 1
        self.past = [p[:, :, -keep:] if p.shape[2] > keep else p for p in outs[2:]]
        self.position += T * self.stride
        ctx_frames = min(SEANET_CONTEXT_FRAMES, self.history.shape[2] // self.stride)
        window = np.concatenate([self.history[:, :, self.history.shape[2] - ctx_frames * self.stride:], hidden], axis=2)
        audio = run(self.se, hidden=window)[0][0]
        self.history = window[:, :, -SEANET_CONTEXT_FRAMES * self.stride:]
        return audio[ctx_frames * self.stride * self.hop:]


def pipeline_onnx(sessions, text_ids, voice, duration_delta, n_layers, heads, head_dim, latent_dim, max_frames,
                  eos_threshold, frames_after_eos, meta):
    """The C++ generation loop, in numpy, for the end-to-end check (temperature 0)."""
    emb_p, emb_l, backbone, flow, mimi_tr, mimi_se = sessions
    dd = np.array([duration_delta], dtype=np.float32)
    prompt = run(emb_p, text_ids=text_ids, voice_latents=voice, duration_delta=dd)[0]
    past = [np.zeros((1, heads, 0, head_dim), dtype=np.float32) for _ in range(2 * n_layers)]
    past_names = [f"past_{kv}_{i}" for i in range(n_layers) for kv in ("k", "v")]
    x = prompt
    latents = []
    remaining = -1
    for frame in range(max_frames):
        outs = run(backbone, inputs_embeds=x, **dict(zip(past_names, past)))
        cond, eos = outs[0], float(outs[1][0])
        past = outs[2:]
        noise = np.zeros((1, latent_dim), dtype=np.float32)
        t0 = np.array([0.0], dtype=np.float32)
        t1 = np.array([1.0], dtype=np.float32)
        vel = run(flow, latent=noise, condition=cond, t_start=t0, t_end=t1)[0]
        latent = noise + vel
        latents.append(latent)
        if remaining < 0 and eos > eos_threshold:
            remaining = frames_after_eos
        if remaining == 0:
            break
        if remaining > 0:
            remaining -= 1
        x = run(emb_l, latent=latent, duration_delta=dd)[0]
    lat = np.concatenate(latents, axis=0)[None]  # [1, F, 32]
    # Decode in uneven chunks, as the streaming engine would.
    streamer = MimiStreamer(mimi_tr, mimi_se, meta)
    audio, i, step = [], 0, 3
    while i < lat.shape[1]:
        audio.append(streamer.decode(lat[:, i : i + step]))
        i += step
        step = step * 2 if step < 20 else 7
    return lat, np.concatenate(audio)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "core/moonshine-tts/data/phonon"))
    parser.add_argument("--keep-onnx", action="store_true", help="Also keep the .onnx graphs in --out-dir")
    parser.add_argument("--skip-voices", action="store_true", help="Do not download the voice catalog")
    parser.add_argument("--voice", default=None, help="Voice .safetensors for the end-to-end check "
                        "(default: the catalog's Zoey)")
    parser.add_argument("--text", default="The birch canoe slid on the smooth planks.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print("Loading models...")
    proc, lm, tok = load_models()
    cfg = lm.config
    n_layers, heads, head_dim, hidden = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size
    latent_dim = cfg.latent_dim

    from huggingface_hub import hf_hub_download

    # Voices: the English catalog entries, by name, plus voices.json describing them.
    voices_dir = out_dir / "voices"
    voice_rows = []
    if not args.skip_voices:
        voices_dir.mkdir(exist_ok=True)
        rows = [json.loads(l) for l in open(hf_hub_download(VOICES_REPO, "voices.jsonl")).read().splitlines()]
        for row in rows:
            if row.get("language") != "en":
                continue
            src = hf_hub_download(VOICES_REPO, row["filename"])
            name = row["name"].strip().lower().replace(" ", "_")
            dst = voices_dir / f"{name}.safetensors"
            # copyfile, not copy: the cache's files are read-only and a copied
            # mode bit would block the next run from overwriting.
            dst.unlink(missing_ok=True)
            shutil.copyfile(src, dst)
            voice_rows.append({
                "id": name,
                "name": row["name"],
                "voice_id": row["voice_id"],
                "locale": row.get("locale", ""),
                "gender": row.get("gender", ""),
                "age": row.get("age", ""),
                "description": row.get("description", ""),
                "file": f"voices/{name}.safetensors",
            })
        voice_rows.sort(key=lambda r: r["id"])
        (out_dir / "voices.json").write_text(json.dumps({"voices": voice_rows}, indent=2) + "\n")
        print(f"  wrote {len(voice_rows)} English voices to {voices_dir}")
    voice_path = Path(args.voice) if args.voice else (
        voices_dir / "zoey.safetensors" if (voices_dir / "zoey.safetensors").exists()
        else Path(hf_hub_download(LM_REPO, "default-voice.safetensors")))
    voice = load_voice(voice_path)

    # The SentencePiece Unigram vocabulary goes into the headed BinTokenizer
    # form (spellings plus scores), which the C++ BinTokenizer encodes with the
    # same Viterbi search as the HF tokenizer.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from convert_tokenizer import convert_huggingface_json  # noqa: E402

    convert_huggingface_json(hf_hub_download(LM_REPO, "tokenizer.json"), str(out_dir / "tokenizer.bin"))

    # Model metadata the C++ engine reads instead of hard-coding.
    meta = {
        "source": {"lm": LM_REPO, "mimi": MIMI_REPO, "voices": VOICES_REPO},
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "head_dim": head_dim,
        "latent_dim": latent_dim,
        "audio_prompt_dim": cfg.audio_prompt_dim,
        "frame_rate": cfg.frame_rate,
        "sample_rate": tok.decoder.config.sampling_rate,
        "frame_size": int(round(tok.decoder.config.sampling_rate / cfg.frame_rate)),
        "mimi_hidden_size": tok.decoder.config.hidden_size,
        "mimi_heads": tok.decoder.config.num_attention_heads,
        "mimi_head_dim": tok.decoder.config.head_dim,
        "mimi_sliding_window": tok.decoder.config.sliding_window,
        "mimi_transformer_layers": tok.decoder.config.num_hidden_layers,
        "mimi_latent_stride": tok.decoder.config.latent_stride,
        "mimi_seanet_context_frames": SEANET_CONTEXT_FRAMES,
        "temperature": float(lm.generation_config.temperature),
        "eos_threshold": float(lm.generation_config.eos_threshold),
        "min_frames_before_eos": int(lm.generation_config.min_frames_before_eos),
        "num_flow_steps": int(lm.generation_config.num_flow_steps),
        "frames_after_eos": 1,
        "frames_after_eos_short_text": 3,
        "short_text_max_words": 4,
        "token_ids": {
            "pad": cfg.pad_token_id,
            "audio_bos": cfg.audio_bos_token_id,
            "audio_prompt": cfg.audio_prompt_token_id,
            "audio_latent": cfg.audio_latent_token_id,
        },
    }
    pc = getattr(proc, "frames_after_eos", None)
    if pc is not None:
        meta["frames_after_eos"] = int(pc)
        meta["frames_after_eos_short_text"] = int(getattr(proc, "frames_after_eos_short_text", 3))
        meta["short_text_max_words"] = int(getattr(proc, "short_text_max_words", 4))
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        text = args.text
        text_ids = torch.tensor([proc.tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long)
        dd = torch.zeros(1)

        print("Exporting...")
        embed_prompt = EmbedPrompt(lm)
        export(embed_prompt, (text_ids, voice, dd), tmp / "embed_prompt.onnx",
               ["text_ids", "voice_latents", "duration_delta"], ["prompt_embeds"],
               {"text_ids": {1: "text"}, "voice_latents": {1: "frames"}, "prompt_embeds": {1: "seq"}})

        embed_latent = EmbedLatent(lm)
        lat0 = torch.randn(1, latent_dim)
        export(embed_latent, (lat0, dd), tmp / "embed_latent.onnx",
               ["latent", "duration_delta"], ["latent_embeds"], {})

        backbone = Backbone(lm)
        past_names = [f"past_{kv}_{i}" for i in range(n_layers) for kv in ("k", "v")]
        present_names = [f"present_{kv}_{i}" for i in range(n_layers) for kv in ("k", "v")]
        P0 = 3
        past0 = tuple(torch.randn(1, heads, P0, head_dim) for _ in range(2 * n_layers))
        x0 = torch.randn(1, 5, hidden)
        dyn = {"inputs_embeds": {1: "seq"}}
        dyn.update({n: {2: "past"} for n in past_names})
        dyn.update({n: {2: "total"} for n in present_names})
        export(backbone, (x0, *past0), tmp / "backbone.onnx", ["inputs_embeds", *past_names],
               ["condition", "eos_logit", *present_names], dyn)

        flow = FlowHead(lm)
        export(flow, (lat0, torch.randn(1, hidden), torch.zeros(1), torch.ones(1)), tmp / "flow_head.onnx",
               ["latent", "condition", "t_start", "t_end"], ["velocity"], {})

        mimi = MimiDecoder(lm, tok)
        mcfg = tok.decoder.config
        m_past_names = [f"past_{kv}_{i}" for i in range(mcfg.num_hidden_layers) for kv in ("k", "v")]
        m_present_names = [f"present_{kv}_{i}" for i in range(mcfg.num_hidden_layers) for kv in ("k", "v")]
        latsT = torch.randn(1, 7, latent_dim)
        m_past0 = tuple(torch.randn(1, mcfg.num_attention_heads, 5, mcfg.head_dim) for _ in m_past_names)
        m_dyn = {"latents": {1: "frames"}, "hidden": {2: "steps"}}
        m_dyn.update({n: {2: "past"} for n in m_past_names})
        m_dyn.update({n: {2: "total"} for n in m_present_names})
        export(mimi.transformer, (latsT, torch.randn(1, mcfg.hidden_size, 1), torch.tensor(5, dtype=torch.long), *m_past0),
               tmp / "mimi_transformer.onnx", ["latents", "prev_hidden", "start_position", *m_past_names],
               ["hidden", "new_prev_hidden", *m_present_names], m_dyn)
        export(mimi.seanet, (torch.randn(1, mcfg.hidden_size, 7 * mcfg.latent_stride),), tmp / "mimi_seanet.onnx",
               ["hidden"], ["audio"], {"hidden": {2: "steps"}, "audio": {1: "samples"}})

        print("Checking graphs against PyTorch...")
        sess = [ort_session(tmp / n) for n in
                ("embed_prompt.onnx", "embed_latent.onnx", "backbone.onnx", "flow_head.onnx",
                 "mimi_transformer.onnx", "mimi_seanet.onnx")]
        with torch.no_grad():
            # Embeddings against the HF scatter-based forward.
            Np = voice.shape[1]
            ids = torch.cat([torch.full((1, Np), cfg.audio_prompt_token_id), text_ids,
                             torch.tensor([[cfg.audio_bos_token_id]])], dim=1)
            want_prompt = embed_prompt(text_ids, voice, dd)
            got_prompt = run(sess[0], text_ids=text_ids, voice_latents=voice, duration_delta=dd)[0]
            check("embed_prompt", got_prompt, want_prompt, 1e-4)

            # Backbone: the HF forward on the full prompt, no cache, versus prefill.
            hf = lm(input_ids=ids, audio_prompt_latents=voice, use_cache=False)
            empty = [np.zeros((1, heads, 0, head_dim), dtype=np.float32)] * (2 * n_layers)
            outs = run(sess[2], inputs_embeds=got_prompt, **dict(zip(past_names, empty)))
            check("backbone.condition (prefill)", outs[0], hf.last_hidden_state[:, -1], 1e-3)
            check("backbone.eos_logit (prefill)", outs[1], hf.logits[:, -1, 0], 1e-3)

            # One cached decode step versus the HF forward over prompt + 2 latents.
            lat_a = torch.randn(1, 1, latent_dim)
            ids2 = torch.cat([ids, torch.full((1, 1), cfg.audio_latent_token_id)], dim=1)
            hf2 = lm(input_ids=ids2, audio_prompt_latents=voice, audio_latents=lat_a, use_cache=False)
            x1 = run(sess[1], latent=lat_a[0], duration_delta=dd)[0]
            outs2 = run(sess[2], inputs_embeds=x1, **dict(zip(past_names, outs[2:])))
            check("backbone.condition (step)", outs2[0], hf2.last_hidden_state[:, -1], 1e-3)
            check("backbone.eos_logit (step)", outs2[1], hf2.logits[:, -1, 0], 1e-3)

            # Flow head.
            cond = torch.from_numpy(outs2[0])
            noise = torch.randn(1, latent_dim) * math.sqrt(0.3)
            want_lat = lm.sample_latent(cond, noise, 1)
            vel = run(sess[3], latent=noise, condition=cond, t_start=torch.zeros(1), t_end=torch.ones(1))[0]
            check("flow_head (1 step)", noise.numpy() + vel, want_lat, 1e-4)

            # Mimi decoder (torch wrapper, then the two ORT graphs in one call) versus HF decode.
            lat_n = torch.randn(1, 11, latent_dim)
            want_audio = tok.decode(lm.denormalize_latents(lat_n)).audio_values[0, 0, : 11 * mimi.frame_size]
            h_n = torch.randn(1, mcfg.hidden_size, 37)
            check("seanet with folded ConvTranspose", mimi.seanet.seanet(h_n), tok.decoder.decoder_seanet(h_n), 1e-5)
            check("mimi wrapper", mimi(lat_n)[0], want_audio, 1e-4)
            streamer = MimiStreamer(sess[4], sess[5], meta)
            check("mimi_transformer + mimi_seanet", streamer.decode(lat_n.numpy()), want_audio, 1e-3)

            # Streaming in chunks (KV cache trimmed to the window, SEANet re-windowed) must match one call.
            lat_long = torch.randn(1, 70, latent_dim)
            streamer.reset()
            full = streamer.decode(lat_long.numpy())
            streamer.reset()
            parts, i = [], 0
            for n in (1, 2, 5, 9, 16, 20, 17):
                parts.append(streamer.decode(lat_long[:, i : i + n].numpy()))
                i += n
            check("mimi streaming in chunks", np.concatenate(parts), full, 1e-3)

            # End to end at temperature 0 against model.generate.
            words = len(text.split())
            fae = meta["frames_after_eos_short_text"] if words <= meta["short_text_max_words"] else meta["frames_after_eos"]
            max_frames = math.ceil((text_ids.shape[1] / 3 + 2) * cfg.frame_rate)
            gen = lm.generate(input_ids=ids, audio_prompt_latents=voice, temperature=0.0, do_sample=True,
                              return_dict_in_generate=True, frames_after_eos=fae)
            want_lat = gen.audio_latents
            want_audio = tok.decode(want_lat).audio_values[0, 0]
            got_lat, got_audio = pipeline_onnx(sess, text_ids.numpy(), voice.numpy(), 0.0, n_layers, heads, head_dim,
                                               latent_dim, max_frames, meta["eos_threshold"], fae, meta)
            print(f"  end-to-end: {got_lat.shape[1]} frames (reference {want_lat.shape[1]})")
            if got_lat.shape[1] != want_lat.shape[1]:
                raise SystemExit("end-to-end frame count differs from the reference")
            check("end-to-end latents", got_lat * lm.latent_std.numpy() + lm.latent_mean.numpy(), want_lat, 2e-3)
            check("end-to-end audio", got_audio[0], want_audio[: got_audio.shape[1]], 5e-3)
            # The PyTorch render, for the C++ engine's parity test
            # (core/moonshine-tts/tests/phonon-tts-test.cpp); only meaningful for the default
            # --text and --voice, which are what the test synthesizes.
            if args.voice is None and args.text == parser.get_default("text"):
                want_audio.numpy().astype(np.float32).tofile(out_dir / "reference.f32")

        print("Converting to ORT format...")
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts/convert-models-to-ort.py"), "--no-split",
                        "--max-growth", "25", str(tmp)], check=True)
        for stem in GRAPHS:
            ort_path = tmp / f"{stem}.ort"
            if not ort_path.exists():
                raise SystemExit(f"conversion did not produce {ort_path.name}")
            shutil.copy(ort_path, out_dir / f"{stem}.ort")
            if args.keep_onnx:
                shutil.copy(tmp / f"{stem}.onnx", out_dir / f"{stem}.onnx")

    print(f"Wrote Phonon bundle to {out_dir}")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name:<24} {p.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()

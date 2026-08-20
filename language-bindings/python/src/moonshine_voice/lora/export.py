"""Export a Hugging Face Moonshine Streaming checkpoint to the runtime's ONNX graphs.

The deployment runtime does not load a single model. It loads five graphs that split
the work the way streaming needs it — `frontend`, `encoder`, `adapter`, `cross_kv` and
`decoder_kv` — plus `streaming_config.json` and `tokenizer.bin`. This script produces
them from a `MoonshineStreamingForConditionalGeneration` checkpoint, which is what you
have after fine-tuning with Transformers, using nothing but public code.

A LoRA adapter on the decoder's self-attention changes only `decoder_kv`, so
`--graphs decoder_kv,cross_kv` is enough to re-export a decoder-only adapter: the
other three graphs are bit-identical to the published ones and can be copied.
`--graphs all` exports everything, which is required for `--sites encoder|both`
or `--adapt full`.

    python -m moonshine_voice.lora --export \
        --model moonshine-ai/moonshine-streaming-medium --output-dir float/

Quantize the result with the public `quantize-streaming-model.sh` from
github.com/moonshine-ai/moonshine, which turns each `.onnx` into the `.ort` the
runtime loads.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 18 rather than 17 because the dynamo exporter emits opset-18 spellings of Split
# whatever it is asked for, and a graph that declares 17 while using them fails the
# ONNX checker.
OPSET = 18

# ONNX has had Asinh since opset 9, but the tracer never learned to emit it, and the
# frontend's amplitude compression is built on it.
torch.onnx.register_custom_op_symbolic(
    "aten::asinh", lambda g, x: g.op("Asinh", x), 9)


def frames_state_shapes(embedder, encoder_hidden):
    """Shapes of the frontend's carry-over state, which the runtime allocates."""
    return {
        "sample_buffer": (1, embedder.frame_len - 1),
        "sample_len": (1,),
        "conv1_buffer": (1, encoder_hidden, 4),
        "conv2_buffer": (1, encoder_hidden * 2, 4),
        "frame_count": (1,),
    }


class Frontend(nn.Module):
    """Audio to encoder features, one chunk at a time.

    The two strided convolutions are causal, so instead of re-padding every chunk the
    runtime carries the last four pre-convolution frames across calls: that is what
    makes the streaming output identical to running the whole utterance at once.
    """

    def __init__(self, embedder):
        super().__init__()
        self.frame_len = embedder.frame_len
        self.cmvn, self.comp, self.linear = embedder.cmvn, embedder.comp, embedder.linear
        self.conv1, self.conv2 = embedder.conv1, embedder.conv2

    def forward(self, audio_chunk: Tensor, sample_buffer: Tensor, sample_len: Tensor,
                conv1_buffer: Tensor, conv2_buffer: Tensor, frame_count: Tensor):
        buffered = sample_buffer.shape[1]
        keep = sample_len[0].long().clamp(max=buffered)
        combined = torch.cat([sample_buffer[:, :keep], audio_chunk], dim=1)
        num_frames = combined.shape[1] // self.frame_len
        used = num_frames * self.frame_len

        frames = combined[:, :used].reshape(1, num_frames, self.frame_len)
        hidden = F.silu(self.linear(self.comp(self.cmvn(frames)))).transpose(1, 2)

        conv1_in = torch.cat([conv1_buffer, hidden], dim=2)
        conv1_out = F.silu(F.conv1d(conv1_in, self.conv1.weight, self.conv1.bias,
                                    stride=self.conv1.stride[0]))
        conv2_in = torch.cat([conv2_buffer, conv1_out], dim=2)
        features = F.conv1d(conv2_in, self.conv2.weight, self.conv2.bias,
                            stride=self.conv2.stride[0]).transpose(1, 2)

        # Samples past the last whole frame wait here for the next chunk. Both the
        # padding and the reported length are built from tensor operations so they stay
        # dynamic in the graph: computing them from Python ints freezes the remainder at
        # whatever the dummy chunk happened to leave over, which is nothing.
        remainder = combined[:, used:]
        padded = torch.cat([remainder, torch.zeros_like(sample_buffer)], dim=1)
        remainder_len = torch.ones_like(remainder[0], dtype=torch.long).sum().reshape(1)
        return (features,
                padded[:, :buffered],
                remainder_len,
                hidden[:, :, -4:],
                conv1_out[:, :, -4:],
                frame_count + num_frames)


class Encoder(nn.Module):
    """Features to encoded states, with each layer's sliding window baked in.

    The public model builds these masks only when it is handed an attention mask, and
    the runtime never has one — so the windows have to be materialized here or the
    export silently becomes a full-attention model.
    """

    def __init__(self, encoder, windows):
        super().__init__()
        self.layers, self.final_norm = encoder.layers, encoder.final_norm
        self.windows = windows

    def forward(self, features: Tensor) -> Tensor:
        hidden = features
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        for layer, (past, future) in zip(self.layers, self.windows):
            # `past` and `future` are both inclusive: a (16, 4) layer sees itself, the
            # 16 frames behind it and the 4 ahead. Reading them as exclusive bounds
            # costs a frame on each side and measurably changes the output.
            allowed = (distance >= -future) & (distance <= past)
            mask = torch.zeros(1, 1, length, length, dtype=hidden.dtype)
            mask = mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0),
                                    torch.finfo(hidden.dtype).min)
            hidden = layer(hidden, attention_mask=mask)
        return self.final_norm(hidden)


class Adapter(nn.Module):
    """Encoded states to decoder memory, positioned from a running offset.

    `pos_offset` is what lets the runtime adapt one chunk at a time and get the same
    positions it would have got from the whole utterance.
    """

    def __init__(self, decoder):
        super().__init__()
        self.pos_emb, self.proj = decoder.pos_emb, decoder.proj

    def forward(self, encoded: Tensor, pos_offset: Tensor) -> Tensor:
        start = pos_offset[0].long()
        positions = torch.arange(encoded.shape[1], device=encoded.device) + start
        return self.proj(encoded + self.pos_emb(positions).unsqueeze(0))


class CrossKV(nn.Module):
    """Cross-attention keys and values for the whole memory span, once per chunk.

    Splitting these out of the decoder is the entire reason the runtime can decode a
    token without touching the audio again. Only the two projections per layer are
    held here, so the graph carries only the weights it uses.
    """

    def __init__(self, layers, heads, head_dim):
        super().__init__()
        self.k_projs = nn.ModuleList([layer.encoder_attn.k_proj for layer in layers])
        self.v_projs = nn.ModuleList([layer.encoder_attn.v_proj for layer in layers])
        self.heads, self.head_dim = heads, head_dim

    def forward(self, memory: Tensor):
        keys, values = [], []
        shape = (1, -1, self.heads, self.head_dim)
        for k_proj, v_proj in zip(self.k_projs, self.v_projs):
            keys.append(k_proj(memory).view(shape).transpose(1, 2))
            values.append(v_proj(memory).view(shape).transpose(1, 2))
        return torch.stack(keys), torch.stack(values)


class DecoderKV(nn.Module):
    """One decoding step: token plus caches to logits plus caches.

    The cross-attention keys and values arrive as inputs and leave untouched as
    outputs, so the runtime can thread the same tensors through every step of a line
    without copying them.
    """

    def __init__(self, decoder, head, heads, head_dim, rotary):
        super().__init__()
        self.embed_tokens, self.norm = decoder.embed_tokens, decoder.norm
        self.rotary = rotary
        # A decode step never projects the memory, so the cross-attention key and value
        # weights are deliberately left out: they live in cross_kv, and holding them
        # here would add their megabytes to every graph.
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "input_layernorm": layer.input_layernorm,
                "q_proj": layer.self_attn.q_proj,
                "k_proj": layer.self_attn.k_proj,
                "v_proj": layer.self_attn.v_proj,
                "o_proj": layer.self_attn.o_proj,
                "post_attention_layernorm": layer.post_attention_layernorm,
                "cross_q_proj": layer.encoder_attn.q_proj,
                "cross_o_proj": layer.encoder_attn.o_proj,
                "final_layernorm": layer.final_layernorm,
                "mlp": layer.mlp,
            }) for layer in decoder.layers])
        # When the output projection matches the embedding, reading the logits off the
        # embedding keeps one copy of a 32768 x 640 matrix instead of two. Checkpoints
        # that were saved untied still tend to hold identical values, so compare those
        # rather than identity.
        self.tied = torch.equal(head.weight, decoder.embed_tokens.weight)
        self.head = None if self.tied else head
        self.heads, self.head_dim = heads, head_dim
        self.scale = head_dim ** -0.5

    def forward(self, token: Tensor, k_self: Tensor, v_self: Tensor,
                out_k_cross: Tensor, out_v_cross: Tensor):
        from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
            apply_rotary_pos_emb,
        )

        length = token.shape[1]
        cached = k_self.shape[3]
        hidden = self.embed_tokens(token.long())

        positions = (torch.arange(length, device=hidden.device) + cached).unsqueeze(0)
        cos, sin = self.rotary(hidden, positions)

        shape = (1, length, self.heads, self.head_dim)
        query_pos = torch.arange(cached, cached + length, device=hidden.device)
        key_pos = torch.arange(cached + length, device=hidden.device)
        causal = (query_pos.unsqueeze(1) >= key_pos.unsqueeze(0))[None, None]

        keys, values = [], []
        for index, block in enumerate(self.blocks):
            normed = block["input_layernorm"](hidden)
            query = block["q_proj"](normed).view(shape).transpose(1, 2)
            key = block["k_proj"](normed).view(shape).transpose(1, 2)
            value = block["v_proj"](normed).view(shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            key = torch.cat([k_self[index], key], dim=2)
            value = torch.cat([v_self[index], value], dim=2)
            keys.append(key)
            values.append(value)

            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
            attended = torch.matmul(F.softmax(scores, dim=-1), value)
            attended = attended.transpose(1, 2).reshape(1, length, self.heads * self.head_dim)
            hidden = hidden + block["o_proj"](attended)

            normed = block["post_attention_layernorm"](hidden)
            query = block["cross_q_proj"](normed).view(shape).transpose(1, 2)
            scores = torch.matmul(query, out_k_cross[index].transpose(-2, -1)) * self.scale
            attended = torch.matmul(F.softmax(scores, dim=-1), out_v_cross[index])
            attended = attended.transpose(1, 2).reshape(1, length, self.heads * self.head_dim)
            hidden = hidden + block["cross_o_proj"](attended)

            hidden = hidden + block["mlp"](block["final_layernorm"](hidden))

        hidden = self.norm(hidden)
        logits = (hidden @ self.embed_tokens.weight.t() if self.tied
                  else self.head(hidden))
        return logits, torch.stack(keys), torch.stack(values), out_k_cross, out_v_cross


def drop_passthrough_renames(model):
    """Give a tensor that enters and leaves a graph unchanged the same name on both sides.

    The runtime threads the cross-attention cache straight through a decode step and
    binds it by name, but the exporter refuses to reuse a name and appends `_orig` to
    the input. Renaming it back — and dropping the Identity copies that stood between
    the two names — restores the contract the runtime expects.

    Which shape that takes depends on the torch version. Older exporters rename the
    input and copy it with one Identity; torch 2.13 keeps the name and emits an
    Identity from the tensor to itself. Either way, anything left producing a name the
    graph also imports is a duplicate definition that ONNX Runtime refuses to load, so
    resolve the aliases and collapse the whole chain.
    """
    graph = model.graph
    outputs = {out.name for out in graph.output}
    renames = {entry.name: entry.name[: -len("_orig")]
               for entry in graph.input
               if entry.name.endswith("_orig") and entry.name[: -len("_orig")] in outputs}
    for entry in graph.input:
        if entry.name in renames:
            entry.name = renames[entry.name]

    aliases = dict(renames)
    passthrough = {entry.name for entry in graph.input} & outputs
    if not passthrough:
        return
    collapsing = True
    while collapsing:
        collapsing = False
        for node in list(graph.node):
            if node.op_type != "Identity":
                continue
            if aliases.get(node.input[0], node.input[0]) not in passthrough:
                continue
            aliases[node.output[0]] = aliases.get(node.input[0], node.input[0])
            graph.node.remove(node)
            collapsing = True

    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in aliases:
                node.input[index] = aliases[name]
    produced = {name for node in graph.node for name in node.output}
    clashes = sorted(produced & {entry.name for entry in graph.input})
    assert not clashes, f"graph both imports and produces {clashes}"


def export(module, args, path, input_names, output_names, dynamic, metadata,
           scripted=False):
    """Trace a wrapper to ONNX under the input and output names the runtime binds by.

    The dynamo exporter is not optional here: the TorchScript one cannot lower this
    decoder's rotary embedding. Its optimize pass is not optional either, because
    without it the weights stay behind Constant nodes and the quantizer finds nothing
    to quantize. Shapes have to be declared through `dynamic_shapes` rather than
    `dynamic_axes`, which this exporter quietly ignores for some inputs, leaving an
    axis frozen at whatever length the dummy input had.

    The frontend is the exception, and takes `scripted=True`: it slices its sample
    buffer by a value read out of a tensor, which `torch.export` rejects as data
    dependent but the older tracer handles.
    """
    module.eval()
    with torch.no_grad():
        if scripted:
            torch.onnx.export(module, args, str(path), input_names=input_names,
                              output_names=output_names, dynamic_axes=dynamic,
                              opset_version=OPSET, do_constant_folding=True,
                              dynamo=False)
        else:
            torch.onnx.export(module, args, str(path), input_names=input_names,
                              output_names=output_names, dynamic_shapes=dynamic,
                              opset_version=OPSET, dynamo=True, optimize=True)
    import onnx

    model = onnx.load(path)
    drop_passthrough_renames(model)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, json.dumps(value)
    # The exporter writes weights to a sidecar .data file; the quantizer and the runtime
    # both expect one self-contained graph, so fold them back in.
    onnx.save(model, path, save_as_external_data=False)
    sidecar = path.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a Moonshine Streaming checkpoint to the runtime ONNX graphs."
    )
    parser.add_argument("--model", required=True, help="HF hub id or save_pretrained dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graphs", default="all",
                        help="'all' or a comma-separated subset of "
                             "frontend,encoder,adapter,cross_kv,decoder_kv")
    parser.add_argument("--tokenizer-bin", default=None,
                        help="tokenizer.bin to copy in; unchanged by fine-tuning, so "
                             "the published one is usually what you want")
    args = parser.parse_args(argv)

    from torch.export import Dim
    from transformers import MoonshineStreamingForConditionalGeneration

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = ({"frontend", "encoder", "adapter", "cross_kv", "decoder_kv"}
              if args.graphs == "all" else set(args.graphs.split(",")))

    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32).eval()
    config, encoder_config = model.config, model.config.encoder_config
    encoder, decoder = model.model.encoder, model.model.decoder
    heads, head_dim = config.num_attention_heads, config.head_dim
    depth = config.num_hidden_layers
    enc_hidden, dec_hidden = encoder_config.hidden_size, config.hidden_size
    windows = [tuple(w) for w in encoder_config.sliding_windows]
    frame_len = encoder.embedder.frame_len
    state_shapes = frames_state_shapes(encoder.embedder, enc_hidden)

    print(f"exporting {args.model}: {sum(p.numel() for p in model.parameters()):,} "
          f"parameters, {depth} decoder layers, vocab {config.vocab_size}")

    if "frontend" in wanted:
        export(Frontend(encoder.embedder),
               (torch.randn(1, 3200), torch.zeros(1, frame_len - 1),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(*state_shapes["conv1_buffer"]),
                torch.zeros(*state_shapes["conv2_buffer"]),
                torch.zeros(1, dtype=torch.long)),
               out / "frontend.onnx",
               ["audio_chunk", "sample_buffer", "sample_len", "conv1_buffer",
                "conv2_buffer", "frame_count"],
               ["features", "sample_buffer_out", "sample_len_out", "conv1_buffer_out",
                "conv2_buffer_out", "frame_count_out"],
               {"audio_chunk": {1: "chunk_len"}, "features": {1: "feat_len"}},
               {"frontend_config": {
                   "mode": "incremental", "frame_len": frame_len,
                   "d_model": enc_hidden, "c1": enc_hidden * 2, "c2": enc_hidden,
                   "state_shapes": {k: list(v) for k, v in state_shapes.items()}}},
               scripted=True)

    if "encoder" in wanted:
        export(Encoder(encoder, windows), (torch.randn(1, 37, enc_hidden),),
               out / "encoder.onnx", ["features"], ["encoded"],
               {"features": {1: Dim("feat_len", min=2, max=1 << 16)}},
               {"encoder_config": {
                   "encoder_dim": enc_hidden,
                   "total_lookahead": sum(w[1] for w in windows),
                   "windows": [list(w) for w in windows]}})

    if "adapter" in wanted:
        export(Adapter(decoder),
               (torch.randn(1, 37, enc_hidden), torch.zeros(1, dtype=torch.long)),
               out / "adapter.onnx", ["encoded", "pos_offset"], ["memory"],
               {"encoded": {1: Dim("enc_len", min=2, max=1 << 16)},
                "pos_offset": None},
               {"adapter_config": {"encoder_dim": enc_hidden, "decoder_dim": dec_hidden}})

    if "cross_kv" in wanted:
        export(CrossKV(decoder.layers, heads, head_dim),
               (torch.randn(1, 37, dec_hidden),),
               out / "cross_kv.onnx", ["memory"], ["k_cross", "v_cross"],
               {"memory": {1: Dim("mem_len", min=2, max=1 << 16)}},
               {"cross_kv_config": {"depth": depth, "nheads": heads,
                                    "head_dim": head_dim, "decoder_dim": dec_hidden}})

    if "decoder_kv" in wanted:
        cache_len = Dim("cache_len", min=1, max=1 << 14)
        memory_len = Dim("memory_len", min=2, max=1 << 16)
        # Every dummy length here is distinct and unlike head_dim: give two axes the
        # same size and the exporter fuses them into one symbol, which freezes the
        # memory length at whatever the dummy happened to be.
        export(DecoderKV(decoder, model.proj_out, heads, head_dim, decoder.rotary_emb),
               (torch.tensor([[1, 100, 200]]),
                torch.zeros(depth, 1, heads, 5, head_dim),
                torch.zeros(depth, 1, heads, 5, head_dim),
                torch.randn(depth, 1, heads, 37, head_dim),
                torch.randn(depth, 1, heads, 37, head_dim)),
               out / "decoder_kv.onnx",
               ["token", "k_self", "v_self", "out_k_cross", "out_v_cross"],
               ["logits", "out_k_self", "out_v_self", "out_k_cross", "out_v_cross"],
               {"token": {1: Dim("token_len", min=1, max=1 << 14)},
                "k_self": {3: cache_len}, "v_self": {3: cache_len},
                "out_k_cross": {3: memory_len}, "out_v_cross": {3: memory_len}},
               {"decoder_config": {"depth": depth, "nheads": heads, "head_dim": head_dim,
                                   "vocab_size": config.vocab_size,
                                   "decoder_dim": dec_hidden,
                                   "bos_id": config.bos_token_id,
                                   "eos_id": config.eos_token_id},
                "cache_shapes": {"k_self": [depth, 1, heads, 0, head_dim],
                                 "v_self": [depth, 1, heads, 0, head_dim]}})

    streaming_config = {
        "encoder_dim": enc_hidden, "decoder_dim": dec_hidden, "depth": depth,
        "nheads": heads, "head_dim": head_dim, "vocab_size": config.vocab_size,
        "bos_id": config.bos_token_id, "eos_id": config.eos_token_id,
        "frame_len": frame_len, "total_lookahead": sum(w[1] for w in windows),
        "d_model_frontend": enc_hidden, "c1": enc_hidden * 2, "c2": enc_hidden,
        "frontend_state_shapes": {k: list(v) for k, v in state_shapes.items()},
    }
    (out / "streaming_config.json").write_text(json.dumps(streaming_config, indent=2))
    print(f"  wrote streaming_config.json")

    if args.tokenizer_bin:
        (out / "tokenizer.bin").write_bytes(Path(args.tokenizer_bin).read_bytes())
        print(f"  copied tokenizer.bin")


def export_checkpoint(model, output_dir, graphs="all", tokenizer_bin=None):
    """Export a Transformers checkpoint (hub id or ``save_pretrained`` dir)."""
    argv = [
        "--model",
        str(model),
        "--output-dir",
        str(output_dir),
        "--graphs",
        graphs,
    ]
    if tokenizer_bin is not None:
        argv.extend(["--tokenizer-bin", str(tokenizer_bin)])
    main(argv)
    return Path(output_dir)


if __name__ == "__main__":
    main()

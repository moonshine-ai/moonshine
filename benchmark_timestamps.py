#!/usr/bin/env python3
"""
Benchmark: compare transcription-only vs two-pass timestamps vs single-pass timestamps.

Measures wall-clock time for each approach on the same audio input.
"""

import sys
import time
import wave

import numpy as np


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        from scipy.signal import resample
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples).astype(np.float32)

    return audio


def transcribe_only(model, input_values):
    """Basic transcription: encoder + decoder. No attention extraction."""
    import torch

    with torch.no_grad():
        generated = model.generate(input_values=input_values, max_new_tokens=448)
    return generated[0].tolist()


def transcribe_two_pass(model, input_values):
    """Two-pass: generate tokens, then teacher-forced forward pass for attention."""
    import torch

    # Pass 1: generate tokens
    with torch.no_grad():
        generated = model.generate(input_values=input_values, max_new_tokens=448)
        tokens = generated[0].tolist()

    # Pass 2: teacher-forced forward pass to get cross-attention weights
    decoder_input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )

    # Materialize: stack cross-attention tensors
    cross_attentions = torch.stack(outputs.cross_attentions)[:, 0]

    return tokens, cross_attentions


def transcribe_single_pass(model, input_values):
    """Single-pass: manual autoregressive decode collecting attention at each step."""
    import torch

    device = next(model.parameters()).device
    input_values = input_values.to(device)

    # Encoder (once)
    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_values)

    bos_id = model.config.decoder_start_token_id
    eos_id = model.config.eos_token_id

    generated_tokens = [bos_id]
    cross_attentions_per_step = []
    past_key_values = None

    for step in range(448):
        if past_key_values is None:
            decoder_input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        else:
            decoder_input_ids = torch.tensor(
                [[generated_tokens[-1]]], dtype=torch.long, device=device
            )

        with torch.no_grad():
            outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1]
        next_token = int(torch.argmax(logits))

        # Collect cross-attention
        step_cross_attn = torch.stack(
            [layer_attn[0] for layer_attn in outputs.cross_attentions]
        )
        cross_attentions_per_step.append(step_cross_attn.cpu())

        generated_tokens.append(next_token)
        if next_token == eos_id:
            break

    # Stack all steps
    all_cross_attn = torch.cat(cross_attentions_per_step, dim=2)

    return generated_tokens, all_cross_attn


def main():
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    import torch
    from transformers import AutoTokenizer, MoonshineForConditionalGeneration

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio.wav> [--warmup N] [--runs N]")
        sys.exit(1)

    audio_path = sys.argv[1]
    warmup = 1
    runs = 3
    for i, arg in enumerate(sys.argv):
        if arg == "--warmup" and i + 1 < len(sys.argv):
            warmup = int(sys.argv[i + 1])
        if arg == "--runs" and i + 1 < len(sys.argv):
            runs = int(sys.argv[i + 1])

    model_name = "UsefulSensors/moonshine-tiny"
    print(f"Loading model: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    audio = load_audio(audio_path)
    audio_duration = len(audio) / 16000.0
    input_values = torch.tensor(audio).unsqueeze(0)
    print(f"Audio: {audio_path}")
    print(f"Duration: {audio_duration:.2f}s")
    print(f"Warmup runs: {warmup}, Timed runs: {runs}")
    print()

    # --- 1. Transcription only ---
    print("=== 1. Transcription only (model.generate) ===")
    for _ in range(warmup):
        transcribe_only(model, input_values)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tokens = transcribe_only(model, input_values)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    avg1 = sum(times) / len(times)
    print(f"  Text: {text}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Times: {['%.3fs' % t for t in times]}")
    print(f"  Average: {avg1:.3f}s")
    print(f"  RTF: {avg1 / audio_duration:.4f}x")
    print()

    # --- 2. Two-pass (generate + teacher-forced attention) ---
    print("=== 2. Two-pass (generate + teacher-forced forward) ===")
    for _ in range(warmup):
        transcribe_two_pass(model, input_values)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tokens2, cross_attn2 = transcribe_two_pass(model, input_values)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    text2 = tokenizer.decode(tokens2, skip_special_tokens=True)
    avg2 = sum(times) / len(times)
    print(f"  Text: {text2}")
    print(f"  Tokens: {len(tokens2)}")
    print(f"  Cross-attn shape: {list(cross_attn2.shape)}")
    print(f"  Times: {['%.3fs' % t for t in times]}")
    print(f"  Average: {avg2:.3f}s")
    print(f"  RTF: {avg2 / audio_duration:.4f}x")
    print(f"  Overhead vs transcription-only: {avg2 / avg1:.2f}x ({(avg2 - avg1) * 1000:.1f}ms extra)")
    print()

    # --- 3. Single-pass (manual decode with attention) ---
    print("=== 3. Single-pass (manual autoregressive + attention) ===")
    for _ in range(warmup):
        transcribe_single_pass(model, input_values)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tokens3, cross_attn3 = transcribe_single_pass(model, input_values)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    text3 = tokenizer.decode(tokens3, skip_special_tokens=True)
    avg3 = sum(times) / len(times)
    print(f"  Text: {text3}")
    print(f"  Tokens: {len(tokens3)}")
    print(f"  Cross-attn shape: {list(cross_attn3.shape)}")
    print(f"  Times: {['%.3fs' % t for t in times]}")
    print(f"  Average: {avg3:.3f}s")
    print(f"  RTF: {avg3 / audio_duration:.4f}x")
    print(f"  Overhead vs transcription-only: {avg3 / avg1:.2f}x ({(avg3 - avg1) * 1000:.1f}ms extra)")
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Audio duration:       {audio_duration:.2f}s")
    print(f"  Transcription only:   {avg1 * 1000:7.1f}ms  (1.00x baseline)")
    print(f"  Two-pass (v1):        {avg2 * 1000:7.1f}ms  ({avg2/avg1:.2f}x baseline)")
    print(f"  Single-pass (v2):     {avg3 * 1000:7.1f}ms  ({avg3/avg1:.2f}x baseline)")
    print(f"  v2 vs v1 speedup:     {avg2/avg3:.2f}x")
    print()
    print(f"  Two-pass overhead:    +{(avg2 - avg1) * 1000:.1f}ms")
    print(f"  Single-pass overhead: +{(avg3 - avg1) * 1000:.1f}ms")


if __name__ == "__main__":
    main()

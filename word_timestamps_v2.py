#!/usr/bin/env python3
"""
Word-level timestamp extraction for Moonshine ASR using cross-attention DTW.
Single-pass version: generates tokens and collects cross-attention weights
in one autoregressive decode loop (no second forward pass).

Based on the technique used by OpenAI Whisper (timing.py).
"""

import argparse
import json
import os
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WordTiming:
    word: str
    start: float  # seconds
    end: float    # seconds
    probability: float = 1.0
    tokens: list[int] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordTiming]
    audio_duration: float
    encoder_frames: int
    tokens: list[int]


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio from a WAV file and return as float32 array."""
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
        sr = target_sr

    return audio, sr


def dtw(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamic Time Warping on a cost matrix.
    Returns (text_indices, time_indices) of the optimal alignment path.
    """
    N, M = cost_matrix.shape
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    trace = np.zeros((N, M), dtype=np.int32)

    for i in range(N):
        for j in range(M):
            candidates = [D[i, j], D[i, j + 1], D[i + 1, j]]
            argmin = int(np.argmin(candidates))
            trace[i, j] = argmin
            D[i + 1, j + 1] = cost_matrix[i, j] + candidates[argmin]

    i, j = N - 1, M - 1
    text_indices = []
    time_indices = []
    while i >= 0 or j >= 0:
        text_indices.append(i)
        time_indices.append(j)
        if i == 0 and j == 0:
            break
        direction = trace[i, j]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1

    text_indices = np.array(text_indices[::-1])
    time_indices = np.array(time_indices[::-1])
    return text_indices, time_indices


def median_filter(x: np.ndarray, filter_width: int = 7) -> np.ndarray:
    """Apply a median filter along the last axis."""
    from scipy.signal import medfilt

    if filter_width <= 1:
        return x
    if filter_width % 2 == 0:
        filter_width += 1

    original_shape = x.shape
    if x.ndim == 1:
        x = x[np.newaxis, np.newaxis, :]
    elif x.ndim == 2:
        x = x[np.newaxis, :]

    pad_width = filter_width // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")
    result = np.zeros_like(padded)
    for b in range(padded.shape[0]):
        for c in range(padded.shape[1]):
            result[b, c] = medfilt(padded[b, c], kernel_size=filter_width)
    result = result[:, :, pad_width:-pad_width]

    return result.reshape(original_shape)


def generate_with_cross_attention(
    model,
    input_values,
    max_new_tokens: int = 448,
) -> tuple[list[int], list, list[float], int]:
    """
    Autoregressive decoding that collects cross-attention weights at each step.

    Returns:
        tokens: list of generated token IDs (including BOS and EOS)
        cross_attentions_per_step: list of tensors, one per decode step,
            each shaped [num_layers, heads, 1, encoder_frames]
        token_probabilities: probability of each generated token
        encoder_frames: number of encoder output frames
    """
    import torch

    device = next(model.parameters()).device
    input_values = input_values.to(device)

    # Run encoder once
    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_values)

    encoder_frames = encoder_outputs.last_hidden_state.shape[1]

    bos_id = model.config.decoder_start_token_id  # 1
    eos_id = model.config.eos_token_id  # 2

    # Start with BOS
    generated_tokens = [bos_id]
    cross_attentions_per_step = []
    token_probabilities = []

    # KV cache for efficient autoregressive decoding
    past_key_values = None

    for step in range(max_new_tokens):
        # On first step, feed BOS and full encoder output.
        # On subsequent steps, feed only the new token + cached KV.
        if past_key_values is None:
            decoder_input_ids = torch.tensor(
                [[bos_id]], dtype=torch.long, device=device
            )
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

        # Update KV cache for next step
        past_key_values = outputs.past_key_values

        # Get logits for the last position
        logits = outputs.logits[0, -1]  # [vocab_size]
        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.argmax(logits))

        # Collect cross-attention: list of [batch, heads, 1, enc_len] per layer
        # Stack into [num_layers, heads, 1, enc_len]
        step_cross_attn = torch.stack(
            [layer_attn[0] for layer_attn in outputs.cross_attentions]
        )
        # step_cross_attn shape: [num_layers, heads, 1, enc_len]
        cross_attentions_per_step.append(step_cross_attn.cpu())

        token_probabilities.append(float(probs[next_token]))

        generated_tokens.append(next_token)

        if next_token == eos_id:
            break

    return generated_tokens, cross_attentions_per_step, token_probabilities, encoder_frames


def build_word_timings(
    tokenizer,
    tokens: list[int],
    cross_attentions_per_step: list,
    token_probabilities: list[float],
    encoder_frames: int,
    audio_duration: float,
) -> list[WordTiming]:
    """
    Build word-level timestamps from per-step cross-attention weights.

    1. Stack per-step attention into a full [tokens x frames] matrix
    2. Normalize and smooth
    3. Apply DTW
    4. Group tokens into words
    """
    import torch

    if not cross_attentions_per_step:
        return []

    # Stack all steps: each is [num_layers, heads, 1, enc_len]
    # Concatenate along the token dim (axis 2) → [num_layers, heads, num_steps, enc_len]
    all_cross_attn = torch.cat(cross_attentions_per_step, dim=2)

    n_layers, n_heads, n_steps, enc_len = all_cross_attn.shape

    # Reshape to [num_layers * heads, n_steps, enc_len]
    weights = all_cross_attn.reshape(n_layers * n_heads, n_steps, enc_len).numpy()

    # Z-score normalize per head
    std = weights.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1e-10, std)
    weights = (weights - weights.mean(axis=-1, keepdims=True)) / std

    # Median filter to smooth
    weights = median_filter(weights, filter_width=7)

    # Average across all heads/layers → [n_steps, enc_len]
    matrix = weights.mean(axis=0)

    # DTW on negated matrix (DTW minimizes, we want to maximize attention)
    text_indices, time_indices = dtw(-matrix)

    # Time per encoder frame
    time_per_frame = audio_duration / encoder_frames

    # --- Group tokens into words ---
    # tokens = [BOS, tok1, tok2, ..., tokN, EOS]
    # cross_attentions_per_step has n_steps entries, one per decode step.
    # Step 0 produced tok1 (from BOS input), step 1 produced tok2, etc.
    # So step i corresponds to token index i+1 in the tokens list.
    # The DTW matrix rows are indexed 0..n_steps-1.
    # text_indices values are row indices in [0, n_steps).
    # We want to map step i → tokens[i+1].

    text_tokens = tokens[1:-1]  # exclude BOS and EOS
    raw_tokens = [tokenizer.convert_ids_to_tokens(t) for t in text_tokens]

    # The number of decode steps that produced text tokens
    # (last step produced EOS, which we exclude from text_tokens)
    n_text_steps = len(text_tokens)

    words = []
    current_word_tokens = []
    current_word_step_indices = []

    for i, (tok_id, raw_tok) in enumerate(zip(text_tokens, raw_tokens)):
        step_idx = i  # row index in DTW matrix

        starts_new_word = raw_tok.startswith("▁") if raw_tok else False

        if starts_new_word and current_word_tokens:
            words.append((current_word_tokens[:], current_word_step_indices[:]))
            current_word_tokens = [tok_id]
            current_word_step_indices = [step_idx]
        else:
            current_word_tokens.append(tok_id)
            current_word_step_indices.append(step_idx)

    if current_word_tokens:
        words.append((current_word_tokens[:], current_word_step_indices[:]))

    # Map each word to time via DTW alignment
    word_timings = []
    for word_tokens, word_step_indices in words:
        word_text = tokenizer.decode(word_tokens).strip()
        if not word_text:
            continue

        mask = np.isin(text_indices, word_step_indices)
        if not mask.any():
            word_timings.append(
                WordTiming(word=word_text, start=0.0, end=0.0, tokens=word_tokens)
            )
            continue

        word_time_indices = time_indices[mask]
        start_frame = int(word_time_indices.min())
        end_frame = int(word_time_indices.max())

        start_time = start_frame * time_per_frame
        end_time = (end_frame + 1) * time_per_frame

        # Probability: average of token probabilities for this word
        word_probs = []
        for si in word_step_indices:
            if si < len(token_probabilities):
                word_probs.append(token_probabilities[si])
        avg_prob = float(np.mean(word_probs)) if word_probs else 0.0

        word_timings.append(
            WordTiming(
                word=word_text,
                start=round(start_time, 3),
                end=round(end_time, 3),
                probability=round(avg_prob, 4),
                tokens=word_tokens,
            )
        )

    # Fix overlapping boundaries
    for i in range(1, len(word_timings)):
        prev = word_timings[i - 1]
        curr = word_timings[i]
        if prev.end > curr.start:
            midpoint = (prev.end + curr.start) / 2
            word_timings[i - 1] = WordTiming(
                word=prev.word,
                start=prev.start,
                end=round(midpoint, 3),
                probability=prev.probability,
                tokens=prev.tokens,
            )
            word_timings[i] = WordTiming(
                word=curr.word,
                start=round(midpoint, 3),
                end=curr.end,
                probability=curr.probability,
                tokens=curr.tokens,
            )

    return word_timings


def transcribe_chunk(
    model,
    tokenizer,
    audio_chunk: np.ndarray,
    time_offset: float,
) -> list[WordTiming]:
    """Transcribe a single audio chunk in a single pass and return word timings."""
    import torch

    input_values = torch.tensor(audio_chunk).unsqueeze(0)

    tokens, cross_attns, token_probs, encoder_frames = generate_with_cross_attention(
        model, input_values
    )

    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if not text:
        return []

    audio_duration = len(audio_chunk) / 16000.0

    word_timings = build_word_timings(
        tokenizer, tokens, cross_attns, token_probs,
        encoder_frames, audio_duration,
    )

    # Apply time offset
    for i, w in enumerate(word_timings):
        word_timings[i] = WordTiming(
            word=w.word,
            start=round(w.start + time_offset, 3),
            end=round(w.end + time_offset, 3),
            probability=w.probability,
            tokens=w.tokens,
        )

    return word_timings


def transcribe_with_word_timestamps(
    model,
    tokenizer,
    audio_path: str,
    chunk_duration: float = 30.0,
) -> TranscriptionResult:
    """
    Full pipeline: load audio -> chunk -> single-pass transcribe+align -> merge.
    """
    import torch

    audio, sr = load_audio(audio_path)
    audio_duration = len(audio) / sr
    samples_per_chunk = int(chunk_duration * sr)

    all_words: list[WordTiming] = []
    all_tokens: list[int] = []
    total_encoder_frames = 0

    if len(audio) <= samples_per_chunk:
        # Short audio: single pass
        input_values = torch.tensor(audio).unsqueeze(0)

        tokens, cross_attns, token_probs, encoder_frames = generate_with_cross_attention(
            model, input_values
        )

        word_timings = build_word_timings(
            tokenizer, tokens, cross_attns, token_probs,
            encoder_frames, audio_duration,
        )
        all_words = word_timings
        all_tokens = tokens
        total_encoder_frames = encoder_frames
    else:
        # Long audio: process in chunks
        stride_duration = chunk_duration * 0.8
        stride_samples = int(stride_duration * sr)
        offset = 0

        chunk_idx = 0
        remaining = len(audio)
        total_chunks = 0
        pos = 0
        while pos < remaining:
            total_chunks += 1
            pos += stride_samples
            if pos < remaining and (remaining - pos) < samples_per_chunk:
                total_chunks += 1
                break

        while offset < len(audio):
            end = min(offset + samples_per_chunk, len(audio))
            chunk = audio[offset:end]
            chunk_dur = len(chunk) / sr

            if chunk_dur < 1.0:
                break

            chunk_idx += 1
            time_offset = offset / sr
            print(
                f"  Chunk {chunk_idx}/{total_chunks}: "
                f"{time_offset:.1f}s - {end/sr:.1f}s",
                file=sys.stderr,
            )
            chunk_words = transcribe_chunk(
                model, tokenizer, chunk, time_offset
            )

            if all_words and chunk_words:
                last_end = all_words[-1].end
                chunk_words = [w for w in chunk_words if w.start >= last_end]

            all_words.extend(chunk_words)

            if end >= len(audio):
                break
            offset += stride_samples

    text = " ".join(w.word for w in all_words)

    return TranscriptionResult(
        text=text,
        words=all_words,
        audio_duration=audio_duration,
        encoder_frames=total_encoder_frames,
        tokens=all_tokens,
    )


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def to_vtt(result: TranscriptionResult, highlight_words: bool = False) -> str:
    lines = ["WEBVTT", ""]
    if highlight_words:
        for word in result.words:
            lines.append(f"{format_timestamp(word.start)} --> {format_timestamp(word.end)}")
            lines.append(word.word)
            lines.append("")
    else:
        if result.words:
            lines.append(f"{format_timestamp(result.words[0].start)} --> {format_timestamp(result.words[-1].end)}")
            lines.append(result.text.strip())
            lines.append("")
    return "\n".join(lines)


def to_srt(result: TranscriptionResult, max_words_per_line: int = 8) -> str:
    lines = []
    idx = 1
    words = result.words
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words_per_line]
        start = format_timestamp(chunk[0].start).replace(".", ",")
        end = format_timestamp(chunk[-1].end).replace(".", ",")
        text = " ".join(w.word for w in chunk)
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
        idx += 1
        i += max_words_per_line
    return "\n".join(lines)


def to_json(result: TranscriptionResult) -> str:
    data = {
        "text": result.text,
        "audio_duration": result.audio_duration,
        "encoder_frames": result.encoder_frames,
        "words": [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for w in result.words
        ],
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Moonshine ASR with word-level timestamps (single-pass)"
    )
    parser.add_argument("audio", nargs="+", help="Path(s) to audio file(s)")
    parser.add_argument(
        "--model", default="UsefulSensors/moonshine-tiny",
        help="HuggingFace model name (default: UsefulSensors/moonshine-tiny)",
    )
    parser.add_argument(
        "--format", choices=["text", "json", "vtt", "srt"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--highlight-words", action="store_true", help="For VTT: one cue per word")
    parser.add_argument("--output-dir", type=str, default=None, help="Write output files to directory")
    parser.add_argument("--chunk-duration", type=float, default=30.0, help="Chunk duration in seconds (default: 30)")
    args = parser.parse_args()

    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    import torch
    from transformers import AutoTokenizer, MoonshineForConditionalGeneration

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = MoonshineForConditionalGeneration.from_pretrained(
        args.model, attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded.\n", file=sys.stderr)

    for audio_path in args.audio:
        print(f"Processing: {audio_path}", file=sys.stderr)
        result = transcribe_with_word_timestamps(
            model, tokenizer, audio_path, chunk_duration=args.chunk_duration
        )

        if args.format == "text":
            output = f"File: {audio_path}\n"
            output += f"Duration: {result.audio_duration:.2f}s\n"
            output += f"Text: {result.text}\n"
            output += f"\nWord-level timestamps:\n"
            for w in result.words:
                output += f"  [{format_timestamp(w.start)} --> {format_timestamp(w.end)}] {w.word}  (conf: {w.probability:.2f})\n"
        elif args.format == "json":
            output = to_json(result)
        elif args.format == "vtt":
            output = to_vtt(result, highlight_words=args.highlight_words)
        elif args.format == "srt":
            output = to_srt(result)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = Path(audio_path).stem
            ext = {"text": "txt", "json": "json", "vtt": "vtt", "srt": "srt"}[args.format]
            out_path = os.path.join(args.output_dir, f"{base}.{ext}")
            with open(out_path, "w") as f:
                f.write(output)
            print(f"  Written to: {out_path}", file=sys.stderr)
        else:
            print(output)

        print("", file=sys.stderr)


if __name__ == "__main__":
    main()

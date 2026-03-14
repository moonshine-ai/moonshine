#!/usr/bin/env python3
"""
Word-level timestamp extraction for Moonshine ASR using cross-attention DTW.

Uses the HuggingFace Transformers implementation of Moonshine to:
1. Transcribe audio
2. Extract cross-attention weights via teacher-forced forward pass
3. Apply Dynamic Time Warping (DTW) to align tokens to audio frames
4. Merge sub-word tokens into word-level timestamps

Based on the technique used by OpenAI Whisper (timing.py).
"""

import argparse
import json
import os
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

    # Resample if needed
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
    # Cumulative cost matrix
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    # Trace matrix for backtracking: 0=diagonal, 1=up, 2=left
    trace = np.zeros((N, M), dtype=np.int32)

    for i in range(N):
        for j in range(M):
            candidates = [D[i, j], D[i, j + 1], D[i + 1, j]]
            argmin = int(np.argmin(candidates))
            trace[i, j] = argmin
            D[i + 1, j + 1] = cost_matrix[i, j] + candidates[argmin]

    # Backtrack
    i, j = N - 1, M - 1
    text_indices = []
    time_indices = []
    while i >= 0 or j >= 0:
        text_indices.append(i)
        time_indices.append(j)
        if i == 0 and j == 0:
            break
        direction = trace[i, j]
        if direction == 0:  # diagonal
            i -= 1
            j -= 1
        elif direction == 1:  # up (came from i-1, j)
            i -= 1
        else:  # left (came from i, j-1)
            j -= 1

    text_indices = np.array(text_indices[::-1])
    time_indices = np.array(time_indices[::-1])
    return text_indices, time_indices


def median_filter(x: np.ndarray, filter_width: int = 7) -> np.ndarray:
    """Apply a median filter along the last axis."""
    from scipy.signal import medfilt

    if filter_width <= 1:
        return x

    # Ensure odd filter width
    if filter_width % 2 == 0:
        filter_width += 1

    original_shape = x.shape
    if x.ndim == 1:
        x = x[np.newaxis, np.newaxis, :]
    elif x.ndim == 2:
        x = x[np.newaxis, :]

    # Pad to handle boundaries
    pad_width = filter_width // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")

    result = np.zeros_like(padded)
    for b in range(padded.shape[0]):
        for c in range(padded.shape[1]):
            result[b, c] = medfilt(padded[b, c], kernel_size=filter_width)

    # Remove padding
    result = result[:, :, pad_width:-pad_width]

    return result.reshape(original_shape)


def find_alignment(
    model,
    tokenizer,
    input_values,
    tokens: list[int],
    encoder_frames: int,
) -> list[WordTiming]:
    """
    Find word-level alignment using cross-attention DTW.

    1. Run teacher-forced forward pass to get cross-attention weights
    2. Normalize and smooth attention weights
    3. Apply DTW to find optimal monotonic alignment
    4. Map token boundaries to time and merge into words
    """
    import torch

    device = next(model.parameters()).device
    decoder_input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_values=input_values.to(device),
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )

    # Stack cross-attention from all layers: [num_layers, batch, heads, dec_len, enc_len]
    cross_attentions = torch.stack(outputs.cross_attentions)
    # Remove batch dim → [num_layers, heads, dec_len, enc_len]
    cross_attentions = cross_attentions[:, 0]

    # Compute token probabilities from logits
    logits = outputs.logits[0]  # [dec_len, vocab_size]
    token_probs = torch.softmax(logits, dim=-1)
    # For each position, get probability of the actual next token
    text_token_ids = tokens[1:]  # shift right (predict next token)
    token_probabilities = []
    for i, tid in enumerate(text_token_ids):
        if i < len(token_probs):
            token_probabilities.append(float(token_probs[i, tid]))
        else:
            token_probabilities.append(0.0)

    # --- Build attention matrix ---
    # Use all layers and heads, then average
    # Shape: [num_layers * heads, dec_len, enc_len]
    n_layers, n_heads, dec_len, enc_len = cross_attentions.shape
    weights = cross_attentions.reshape(n_layers * n_heads, dec_len, enc_len)
    weights = weights.cpu().numpy()

    # Normalize: softmax along time axis (already softmaxed, but re-normalize)
    # Then z-score normalize per head
    std = weights.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1e-10, std)
    weights = (weights - weights.mean(axis=-1, keepdims=True)) / std

    # Apply median filter to smooth
    weights = median_filter(weights, filter_width=7)

    # Average across all heads/layers → [dec_len, enc_len]
    matrix = weights.mean(axis=0)

    # --- DTW alignment ---
    # DTW minimizes cost, so negate the attention matrix
    text_indices, time_indices = dtw(-matrix)

    # --- Map to word timestamps ---
    # Calculate time per encoder frame
    audio_duration = input_values.shape[-1] / 16000.0
    time_per_frame = audio_duration / encoder_frames

    # Find token boundaries: where text_indices changes
    # Skip BOS token (index 0)
    # Tokens: [BOS, tok1, tok2, ..., tokN, EOS]
    # We want timestamps for tok1..tokN (indices 1 to len-2)

    # Split tokens into words using the raw token representation.
    # SentencePiece tokenizers use '▁' prefix to mark word boundaries.
    text_tokens = tokens[1:-1]  # exclude BOS and EOS
    raw_tokens = [tokenizer.convert_ids_to_tokens(t) for t in text_tokens]

    # Group tokens into words: tokens starting with '▁' begin new words
    words = []
    current_word_tokens = []
    current_word_token_indices = []  # indices into the full token sequence

    for i, (tok_id, raw_tok) in enumerate(zip(text_tokens, raw_tokens)):
        token_idx = i + 1  # offset for BOS

        # '▁' prefix means start of a new word in SentencePiece
        starts_new_word = raw_tok.startswith("▁") if raw_tok else False

        if starts_new_word and current_word_tokens:
            words.append(
                (current_word_tokens[:], current_word_token_indices[:])
            )
            current_word_tokens = [tok_id]
            current_word_token_indices = [token_idx]
        else:
            current_word_tokens.append(tok_id)
            current_word_token_indices.append(token_idx)

    # Don't forget the last word
    if current_word_tokens:
        words.append((current_word_tokens[:], current_word_token_indices[:]))

    # For each word, find the time range from DTW alignment
    word_timings = []
    for word_tokens, word_indices in words:
        word_text = tokenizer.decode(word_tokens).strip()
        if not word_text:
            continue

        # Find all DTW path points that correspond to these token indices
        mask = np.isin(text_indices, word_indices)
        if not mask.any():
            # Fallback: interpolate
            word_timings.append(
                WordTiming(word=word_text, start=0.0, end=0.0, tokens=word_tokens)
            )
            continue

        word_time_indices = time_indices[mask]
        start_frame = int(word_time_indices.min())
        end_frame = int(word_time_indices.max())

        start_time = start_frame * time_per_frame
        end_time = (end_frame + 1) * time_per_frame  # +1 for inclusive end

        # Average probability across tokens in this word
        word_probs = []
        for idx in word_indices:
            prob_idx = idx - 1  # offset since token_probabilities is 0-indexed
            if 0 <= prob_idx < len(token_probabilities):
                word_probs.append(token_probabilities[prob_idx])
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

    # Post-process: fix overlapping word boundaries.
    # When consecutive words share DTW frames, snap the boundary so
    # the previous word ends where the next word starts.
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
    """Transcribe a single audio chunk and return word timings with offset applied."""
    import torch

    input_values = torch.tensor(audio_chunk).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(
            input_values=input_values,
            max_new_tokens=448,
        )
        tokens = generated[0].tolist()

    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if not text:
        return []

    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_values)
        encoder_frames = encoder_outputs[0].shape[1]

    word_timings = find_alignment(
        model, tokenizer, input_values, tokens, encoder_frames
    )

    # Apply time offset for this chunk's position in the full audio
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
    Full pipeline: load audio → chunk → transcribe each chunk → merge timestamps.

    For short audio (<= chunk_duration), processes in a single pass.
    For longer audio, splits into overlapping chunks and merges results.
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

        with torch.no_grad():
            generated = model.generate(
                input_values=input_values,
                max_new_tokens=448,
            )
            tokens = generated[0].tolist()
            encoder_outputs = model.model.encoder(input_values)
            encoder_frames = encoder_outputs[0].shape[1]

        word_timings = find_alignment(
            model, tokenizer, input_values, tokens, encoder_frames
        )
        all_words = word_timings
        all_tokens = tokens
        total_encoder_frames = encoder_frames
    else:
        # Long audio: process in chunks
        # Use a stride shorter than chunk_duration to overlap chunks,
        # helping avoid cutting words at boundaries.
        stride_duration = chunk_duration * 0.8  # 80% stride (20% overlap)
        stride_samples = int(stride_duration * sr)
        offset = 0

        chunk_idx = 0
        remaining = len(audio)
        total_chunks = 0
        pos = 0
        while pos < remaining:
            total_chunks += 1
            pos += stride_samples
            # If what's left after this stride is less than a full chunk,
            # the next iteration will grab it all, so count one more and stop.
            if pos < remaining and (remaining - pos) < samples_per_chunk:
                total_chunks += 1
                break

        while offset < len(audio):
            end = min(offset + samples_per_chunk, len(audio))
            chunk = audio[offset:end]
            chunk_duration_s = len(chunk) / sr

            # Skip chunks shorter than 1 second
            if chunk_duration_s < 1.0:
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
                # Remove overlapping words from the new chunk:
                # discard words from the new chunk that start before
                # the last word of the previous chunk ends.
                last_end = all_words[-1].end
                chunk_words = [w for w in chunk_words if w.start >= last_end]

            all_words.extend(chunk_words)

            # If this chunk reached the end of audio, we're done
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
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def to_vtt(result: TranscriptionResult, highlight_words: bool = False) -> str:
    """Generate WebVTT output from transcription result."""
    lines = ["WEBVTT", ""]

    if highlight_words:
        # One cue per word
        for word in result.words:
            start = format_timestamp(word.start)
            end = format_timestamp(word.end)
            lines.append(f"{start} --> {end}")
            lines.append(word.word)
            lines.append("")
    else:
        # Single cue for the full segment
        if result.words:
            start = format_timestamp(result.words[0].start)
            end = format_timestamp(result.words[-1].end)
            lines.append(f"{start} --> {end}")
            lines.append(result.text.strip())
            lines.append("")

    return "\n".join(lines)


def to_srt(result: TranscriptionResult, max_words_per_line: int = 8) -> str:
    """Generate SRT output, grouping words into subtitle lines."""
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
    """Generate JSON output."""
    data = {
        "text": result.text,
        "audio_duration": result.audio_duration,
        "encoder_frames": result.encoder_frames,
        "words": [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            }
            for w in result.words
        ],
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Moonshine ASR with word-level timestamps"
    )
    parser.add_argument("audio", nargs="+", help="Path(s) to audio file(s)")
    parser.add_argument(
        "--model",
        default="UsefulSensors/moonshine-tiny",
        help="HuggingFace model name (default: UsefulSensors/moonshine-tiny)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "vtt", "srt"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--highlight-words",
        action="store_true",
        help="For VTT: one cue per word",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write output files to this directory instead of stdout",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=30.0,
        help="Audio chunk duration in seconds for long files (default: 30)",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, MoonshineForConditionalGeneration

    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

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
            ext = {"text": "txt", "json": "json", "vtt": "vtt", "srt": "srt"}[
                args.format
            ]
            out_path = os.path.join(args.output_dir, f"{base}.{ext}")
            with open(out_path, "w") as f:
                f.write(output)
            print(f"  Written to: {out_path}", file=sys.stderr)
        else:
            print(output)

        print("", file=sys.stderr)


if __name__ == "__main__":
    main()

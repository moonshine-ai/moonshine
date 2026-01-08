"""
Example script for preparing training data for Moonshine.

This script shows how to:
1. Load audio files and transcriptions
2. Create manifest files
3. Filter and preprocess data

Usage:
    python examples/prepare_training_data.py \
        --audio_dir /path/to/audio \
        --transcripts_file /path/to/transcripts.txt \
        --output_manifest train_manifest.jsonl
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np

from moonshine.data_utils import prepare_manifest_from_dataset
from moonshine.tokenizer_utils import TextNormalizer


def load_transcripts(transcript_file: str) -> dict:
    """
    Load transcripts from a text file.

    Expected format: Each line should be "audio_filename.wav|transcription text"

    Args:
        transcript_file: Path to transcript file

    Returns:
        Dictionary mapping audio filenames to transcriptions
    """
    transcripts = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                filename, text = line.split("|", 1)
                transcripts[filename] = text.strip()
    return transcripts


def filter_by_duration(
    audio_paths: List[str],
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    sample_rate: int = 16000,
) -> Tuple[List[str], List[float]]:
    """
    Filter audio files by duration.

    Args:
        audio_paths: List of audio file paths
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        sample_rate: Audio sample rate

    Returns:
        Tuple of (filtered_paths, durations)
    """
    filtered_paths = []
    durations = []

    for audio_path in audio_paths:
        try:
            # Load audio to get duration
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            duration = len(audio) / sr

            # Filter by duration
            if min_duration <= duration <= max_duration:
                filtered_paths.append(audio_path)
                durations.append(duration)
            else:
                print(f"Skipping {audio_path}: duration {duration:.2f}s out of range")

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

    return filtered_paths, durations


def prepare_moonshine_manifest(
    audio_dir: str,
    transcripts: dict,
    output_path: str,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    normalize_text: bool = True,
):
    """
    Prepare a manifest file for Moonshine training.

    Args:
        audio_dir: Directory containing audio files
        transcripts: Dictionary mapping filenames to transcriptions
        output_path: Path to output manifest file
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        normalize_text: Whether to normalize transcription text
    """
    audio_dir = Path(audio_dir)

    # Collect audio files
    audio_paths = []
    texts = []

    for filename, text in transcripts.items():
        audio_path = audio_dir / filename

        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        # Normalize text if requested
        if normalize_text:
            text = TextNormalizer.normalize(text)

        if text:  # Only add if text is not empty
            audio_paths.append(str(audio_path))
            texts.append(text)

    print(f"Found {len(audio_paths)} audio files with transcriptions")

    # Filter by duration
    filtered_paths, durations = filter_by_duration(
        audio_paths, min_duration, max_duration
    )

    print(f"After duration filtering: {len(filtered_paths)} files")

    # Get corresponding texts
    filtered_texts = [texts[audio_paths.index(path)] for path in filtered_paths]

    # Create manifest
    with open(output_path, "w") as f:
        for audio_path, text, duration in zip(filtered_paths, filtered_texts, durations):
            manifest_entry = {
                "audio": audio_path,
                "text": text,
                "duration": duration,
            }
            f.write(json.dumps(manifest_entry) + "\n")

    print(f"Created manifest with {len(filtered_paths)} entries at {output_path}")

    # Print statistics
    avg_duration = np.mean(durations)
    total_hours = np.sum(durations) / 3600

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(durations)}")
    print(f"  Average duration: {avg_duration:.2f}s")
    print(f"  Total hours: {total_hours:.2f}h")
    print(f"  Min duration: {np.min(durations):.2f}s")
    print(f"  Max duration: {np.max(durations):.2f}s")


def combine_segments(
    manifest_path: str,
    output_path: str,
    target_duration: float = 15.0,
    max_gap: float = 2.0,
):
    """
    Combine short audio segments into longer training instances.

    This implements the preprocessing described in the paper where successive
    segments are combined into 4-30 second instances.

    Args:
        manifest_path: Path to input manifest
        output_path: Path to output manifest with combined segments
        target_duration: Target duration for combined segments
        max_gap: Maximum gap between segments
    """
    # Load manifest
    segments = []
    with open(manifest_path, "r") as f:
        for line in f:
            segments.append(json.loads(line.strip()))

    # Sort by audio file to group related segments
    segments.sort(key=lambda x: x["audio"])

    combined = []
    current_batch = []
    current_duration = 0.0

    for segment in segments:
        # Check if we should start a new batch
        if current_duration + segment["duration"] > 30.0:
            # Save current batch if it's long enough
            if current_duration >= 4.0:
                combined.append(current_batch)
            # Start new batch
            current_batch = [segment]
            current_duration = segment["duration"]
        else:
            current_batch.append(segment)
            current_duration += segment["duration"]

    # Save final batch
    if current_duration >= 4.0:
        combined.append(current_batch)

    print(f"Combined {len(segments)} segments into {len(combined)} training instances")

    # Write combined manifest
    # Note: This is simplified - in practice, you'd need to actually concatenate
    # the audio files and combine the transcriptions
    with open(output_path, "w") as f:
        for batch in combined:
            # Combine texts
            combined_text = " ".join([seg["text"] for seg in batch])
            combined_duration = sum([seg["duration"] for seg in batch])

            # For simplicity, just use first audio path
            # In practice, you'd concatenate the audio files
            manifest_entry = {
                "audio": batch[0]["audio"],  # This should be concatenated audio
                "text": combined_text,
                "duration": combined_duration,
                "segments": len(batch),
            }
            f.write(json.dumps(manifest_entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Moonshine")
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--transcripts_file",
        type=str,
        required=True,
        help="File containing transcriptions (format: filename|text)",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        required=True,
        help="Output manifest file path",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--normalize_text", action="store_true", help="Normalize transcription text"
    )
    parser.add_argument(
        "--combine_segments",
        action="store_true",
        help="Combine short segments into longer instances",
    )

    args = parser.parse_args()

    # Load transcripts
    print(f"Loading transcripts from {args.transcripts_file}")
    transcripts = load_transcripts(args.transcripts_file)
    print(f"Loaded {len(transcripts)} transcriptions")

    # Prepare manifest
    prepare_moonshine_manifest(
        audio_dir=args.audio_dir,
        transcripts=transcripts,
        output_path=args.output_manifest,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        normalize_text=args.normalize_text,
    )

    # Optionally combine segments
    if args.combine_segments:
        combined_path = args.output_manifest.replace(".jsonl", "_combined.jsonl")
        print(f"\nCombining segments...")
        combine_segments(args.output_manifest, combined_path)


if __name__ == "__main__":
    main()

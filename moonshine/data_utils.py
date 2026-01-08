"""
Data loading and preprocessing utilities for Moonshine training.

Based on the paper: https://arxiv.org/abs/2410.15608
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import keras
import librosa
import numpy as np


class AudioDataset:
    """Dataset for loading audio and transcriptions for Moonshine training."""

    def __init__(
        self,
        manifest_paths: List[str],
        sample_rate: int = 16000,
        min_duration: float = 4.0,
        max_duration: float = 30.0,
        max_segment_gap: float = 2.0,
        normalize: bool = True,
    ):
        """
        Initialize the AudioDataset.

        Args:
            manifest_paths: List of paths to manifest JSON files.
                Each line should be a JSON with {"audio": path, "text": transcription, "duration": float}
            sample_rate: Target sample rate for audio
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            max_segment_gap: Maximum gap between segments when combining
            normalize: Whether to normalize audio to [-1, 1] range
        """
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_segment_gap = max_segment_gap
        self.normalize = normalize

        # Load manifests
        self.samples = []
        for manifest_path in manifest_paths:
            with open(manifest_path, "r") as f:
                for line in f:
                    sample = json.loads(line.strip())
                    # Filter by duration
                    duration = sample.get("duration", 0)
                    if duration > 0:  # We'll combine short samples later
                        self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from {len(manifest_paths)} manifests")

    def __len__(self) -> int:
        return len(self.samples)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Normalize if requested
        if self.normalize and len(audio) > 0:
            audio = audio / (np.abs(audio).max() + 1e-8)

        return audio

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, str, float]]:
        """
        Get a single sample.

        Returns:
            Dictionary with 'audio', 'text', and 'duration'
        """
        sample = self.samples[idx]

        # Load audio
        audio = self.load_audio(sample["audio"])

        return {
            "audio": audio,
            "text": sample["text"],
            "duration": len(audio) / self.sample_rate,
        }


def collate_fn_variable_length(batch: List[Dict]) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Collate function for variable-length batches (no padding in encoder).

    Since Moonshine supports variable-length sequences, we process each
    sample individually (batch size of 1 per forward pass is most efficient).

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched audio and text
    """
    # For simplicity, we'll just return the batch as-is
    # The training loop will handle individual samples
    audios = [item["audio"] for item in batch]
    texts = [item["text"] for item in batch]
    durations = [item["duration"] for item in batch]

    return {
        "audio": audios,
        "text": texts,
        "duration": durations,
    }


def create_data_loader(
    dataset: AudioDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Create a data loader for the dataset.

    Note: For Moonshine, batch_size=1 is most efficient due to variable-length sequences.

    Args:
        dataset: AudioDataset instance
        batch_size: Batch size (recommend 1 for variable-length)
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        Data loader iterator
    """
    # Simple generator-based data loader
    indices = list(range(len(dataset)))

    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = [dataset[idx] for idx in batch_indices]
        yield collate_fn_variable_length(batch)


def prepare_manifest_from_dataset(
    audio_paths: List[str],
    transcriptions: List[str],
    output_path: str,
    sample_rate: int = 16000,
):
    """
    Create a manifest file from lists of audio paths and transcriptions.

    Args:
        audio_paths: List of paths to audio files
        transcriptions: List of transcription texts
        output_path: Path to output manifest file
        sample_rate: Sample rate to use for duration calculation
    """
    with open(output_path, "w") as f:
        for audio_path, text in zip(audio_paths, transcriptions):
            # Calculate duration
            try:
                audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                duration = len(audio) / sr

                manifest_entry = {
                    "audio": audio_path,
                    "text": text,
                    "duration": duration,
                }
                f.write(json.dumps(manifest_entry) + "\n")
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    print(f"Created manifest with {len(audio_paths)} entries at {output_path}")


class SpecAugment:
    """
    Optional SpecAugment data augmentation for audio features.

    Note: The paper doesn't mention using SpecAugment, but it's included
    as an optional augmentation technique.
    """

    def __init__(
        self,
        time_mask_param: int = 50,
        freq_mask_param: int = 10,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to audio features.

        Args:
            features: Audio features of shape [time, freq]

        Returns:
            Augmented features
        """
        features = features.copy()
        time_len, freq_len = features.shape

        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, time_len - t))
            features[t0 : t0 + t, :] = 0

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, freq_len - f))
            features[:, f0 : f0 + f] = 0

        return features

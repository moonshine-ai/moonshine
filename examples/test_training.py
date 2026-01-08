"""
Test script for Moonshine training setup.

This script creates a small synthetic dataset and runs a few training steps
to verify the training pipeline works correctly.

Usage:
    python examples/test_training.py
"""
import json
import numpy as np
import tempfile
from pathlib import Path

import keras

from moonshine.model import Moonshine
from moonshine.train import MoonshineTrainer
from moonshine.train_config import TrainingConfig, DataConfig
from moonshine.data_utils import AudioDataset


def create_synthetic_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Create synthetic audio data for testing.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        Synthetic audio waveform
    """
    num_samples = int(duration * sample_rate)
    # Create random noise
    audio = np.random.randn(num_samples).astype(np.float32) * 0.1
    return audio


def create_test_dataset(num_samples: int = 10, output_dir: str = None):
    """
    Create a small test dataset with synthetic audio.

    Args:
        num_samples: Number of samples to create
        output_dir: Directory to save test data

    Returns:
        Path to manifest file
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create audio directory
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Sample texts
    sample_texts = [
        "hello world",
        "this is a test",
        "moonshine speech recognition",
        "training example number",
        "quick brown fox jumps",
    ]

    # Create samples
    manifest_path = output_dir / "test_manifest.jsonl"

    with open(manifest_path, "w") as f:
        for i in range(num_samples):
            # Create synthetic audio
            duration = np.random.uniform(4.0, 10.0)
            audio = create_synthetic_audio(duration)

            # Save audio file
            audio_path = audio_dir / f"test_{i:04d}.npy"
            np.save(audio_path, audio)

            # Create manifest entry
            text = sample_texts[i % len(sample_texts)] + f" {i}"
            manifest_entry = {
                "audio": str(audio_path),
                "text": text,
                "duration": duration,
            }
            f.write(json.dumps(manifest_entry) + "\n")

    print(f"Created test dataset with {num_samples} samples in {output_dir}")
    return str(manifest_path)


def test_data_loading():
    """Test data loading functionality."""
    print("\n" + "=" * 50)
    print("Testing Data Loading")
    print("=" * 50)

    # Create test dataset
    manifest_path = create_test_dataset(num_samples=5)

    # Load dataset
    dataset = AudioDataset(
        manifest_paths=[manifest_path],
        sample_rate=16000,
        min_duration=1.0,
        max_duration=30.0,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"Sample audio shape: {sample['audio'].shape}")
    print(f"Sample text: {sample['text']}")
    print(f"Sample duration: {sample['duration']:.2f}s")

    print("✓ Data loading test passed")
    return dataset


def test_model_creation():
    """Test model creation."""
    print("\n" + "=" * 50)
    print("Testing Model Creation")
    print("=" * 50)

    # Create a small model for testing
    model = Moonshine(
        dim=64,  # Small for testing
        inner_dim=64,
        n_head=4,
        enc_n_layers=2,
        dec_n_layers=2,
        enc_ff_mult=2,
        dec_ff_mult=2,
        enc_ff_swiglu=False,
        dec_ff_swiglu=True,
        vocab_size=32768,
    )

    print(f"Model created successfully")
    print(f"  Preprocessor: {model.preprocessor}")
    print(f"  Encoder: {model.encoder}")
    print(f"  Decoder: {model.decoder}")

    # Test forward pass with dummy data
    dummy_audio = np.random.randn(1, 16000, 1).astype(np.float32)
    audio_features = model.preprocessor(dummy_audio)["audio_features"]
    print(f"  Audio features shape: {audio_features.shape}")

    print("✓ Model creation test passed")
    return model


def test_training_step(dataset, model):
    """Test a single training step."""
    print("\n" + "=" * 50)
    print("Testing Training Step")
    print("=" * 50)

    # Create minimal config
    config = TrainingConfig(
        dim=64,
        inner_dim=64,
        n_head=4,
        enc_n_layers=2,
        dec_n_layers=2,
        enc_ff_mult=2,
        dec_ff_mult=2,
        vocab_size=32768,
        num_train_steps=10,
        batch_size_per_gpu=1,
        learning_rate=1e-4,
        warmup_steps=2,
        logging_steps=1,
        save_steps=1000,  # Don't save during test
    )

    data_config = DataConfig()

    # Create trainer
    trainer = MoonshineTrainer(config, data_config, model=model)

    # Get a sample
    sample = dataset[0]

    # Run a training step
    print("Running training step...")
    try:
        metrics = trainer.train_step(sample["audio"], sample["text"])
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Learning rate: {metrics['learning_rate']:.6f}")
        print("✓ Training step test passed")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_training_loop(dataset, model):
    """Test a few iterations of the full training loop."""
    print("\n" + "=" * 50)
    print("Testing Full Training Loop (5 steps)")
    print("=" * 50)

    # Create minimal config
    config = TrainingConfig(
        dim=64,
        inner_dim=64,
        n_head=4,
        enc_n_layers=2,
        dec_n_layers=2,
        enc_ff_mult=2,
        dec_ff_mult=2,
        vocab_size=32768,
        num_train_steps=5,
        batch_size_per_gpu=1,
        learning_rate=1e-4,
        warmup_steps=2,
        logging_steps=1,
        save_steps=1000,
        output_dir=tempfile.mkdtemp(),
    )

    data_config = DataConfig()

    # Create trainer
    trainer = MoonshineTrainer(config, data_config, model=model)

    # Run training
    try:
        trainer.train(train_dataset=dataset, val_dataset=None)
        print("✓ Full training loop test passed")
        return True
    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("MOONSHINE TRAINING TEST SUITE")
    print("=" * 50)

    # Test 1: Data Loading
    try:
        dataset = test_data_loading()
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return

    # Test 2: Model Creation
    try:
        model = test_model_creation()
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return

    # Test 3: Training Step
    try:
        success = test_training_step(dataset, model)
        if not success:
            return
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        return

    # Test 4: Full Training Loop
    try:
        # Create a new model for full training test
        model = test_model_creation()
        success = test_full_training_loop(dataset, model)
        if not success:
            return
    except Exception as e:
        print(f"✗ Full training loop test failed: {e}")
        return

    # Summary
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
    print("\nYou can now train Moonshine models with real data!")
    print("See TRAINING.md for detailed instructions.")


if __name__ == "__main__":
    main()

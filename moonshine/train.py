"""
Training script for Moonshine models.

Based on the paper: https://arxiv.org/abs/2410.15608

Usage:
    python -m moonshine.train --config configs/tiny_config.json
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import keras
from keras import ops
import numpy as np

from moonshine.model import Moonshine
from moonshine.train_config import TrainingConfig, DataConfig, get_tiny_config, get_base_config
from moonshine.data_utils import AudioDataset, create_data_loader
from moonshine.tokenizer_utils import get_tokenizer


class MoonshineTrainer:
    """Trainer class for Moonshine models."""

    def __init__(
        self,
        config: TrainingConfig,
        data_config: DataConfig,
        model: Optional[Moonshine] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            data_config: Data configuration
            model: Optional pre-initialized model
        """
        self.config = config
        self.data_config = data_config

        # Initialize model
        if model is None:
            self.model = Moonshine(
                dim=config.dim,
                inner_dim=config.inner_dim,
                n_head=config.n_head,
                enc_n_layers=config.enc_n_layers,
                dec_n_layers=config.dec_n_layers,
                enc_ff_mult=config.enc_ff_mult,
                dec_ff_mult=config.dec_ff_mult,
                enc_ff_swiglu=config.enc_ff_swiglu,
                dec_ff_swiglu=config.dec_ff_swiglu,
                vocab_size=config.vocab_size,
            )
        else:
            self.model = model

        # Initialize tokenizer
        self.tokenizer = get_tokenizer()

        # Setup optimizer (AdamW with schedule-free, or standard AdamW as fallback)
        self.optimizer = keras.optimizers.AdamW(
            learning_rate=self._create_learning_rate_schedule(),
            weight_decay=config.weight_decay,
            beta_1=config.adam_beta1,
            beta_2=config.adam_beta2,
            epsilon=config.adam_epsilon,
            clipnorm=config.max_grad_norm,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _create_learning_rate_schedule(self):
        """Create learning rate schedule with warmup."""
        # Linear warmup then constant
        def lr_schedule(step):
            if step < self.config.warmup_steps:
                return self.config.learning_rate * (step / self.config.warmup_steps)
            return self.config.learning_rate

        return lr_schedule

    def compute_loss(
        self,
        audio: np.ndarray,
        target_ids: np.ndarray,
    ) -> tuple:
        """
        Compute the training loss for a single example.

        Args:
            audio: Audio waveform of shape [1, num_samples, 1]
            target_ids: Target token IDs of shape [1, seq_len]

        Returns:
            Tuple of (loss, logits)
        """
        # Get audio features from preprocessor
        audio_input = keras.ops.convert_to_tensor(audio, dtype="float32")
        audio_features = self.model.preprocessor(audio_input)["audio_features"]

        # Get encoder output
        seq_len = keras.ops.convert_to_tensor([audio_features.shape[-2]], dtype="int32")
        encoder_output = self.model.encoder(audio_features, seq_len)["last_hidden_state"]

        # Prepare decoder inputs (shifted right)
        # Input: [BOS, token1, token2, ..., tokenN-1]
        # Target: [token1, token2, ..., tokenN, EOS]
        decoder_input_ids = target_ids[:, :-1]  # Remove last token
        decoder_targets = target_ids[:, 1:]  # Remove first token (BOS)

        # Get decoder output
        decoder_seq_len = keras.ops.convert_to_tensor(
            [decoder_input_ids.shape[-1]], dtype="int32"
        )
        decoder_output = self.model.decoder.uncached_call(
            [decoder_input_ids, encoder_output, decoder_seq_len]
        )
        logits = decoder_output[0]  # [batch, seq_len, vocab_size]

        # Compute cross-entropy loss
        loss = keras.losses.sparse_categorical_crossentropy(
            decoder_targets, logits, from_logits=True
        )
        loss = keras.ops.mean(loss)

        return loss, logits

    def train_step(self, audio: np.ndarray, text: str) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            audio: Audio waveform
            text: Transcription text

        Returns:
            Dictionary with loss and other metrics
        """
        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        # Add BOS (1) and EOS (2) tokens
        token_ids = [1] + token_ids + [2]
        target_ids = np.array([token_ids], dtype=np.int32)

        # Prepare audio input [1, num_samples, 1]
        audio_input = audio.reshape(1, -1, 1).astype(np.float32)

        # Forward pass with gradient tracking
        with keras.backend.name_scope("train_step"):
            # Get trainable variables
            trainable_vars = (
                self.model.preprocessor.preprocess.trainable_variables
                + self.model.encoder.encoder.trainable_variables
                + self.model.decoder.uncached_call.trainable_variables
            )

            # Compute gradients
            with keras.backend.GradientTape() as tape:
                loss, logits = self.compute_loss(audio_input, target_ids)

            # Compute gradients
            gradients = tape.gradient(loss, trainable_vars)

            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": float(keras.ops.convert_to_numpy(loss)),
            "learning_rate": float(
                self._create_learning_rate_schedule()(self.global_step)
            ),
        }

    def train(
        self,
        train_dataset: AudioDataset,
        val_dataset: Optional[AudioDataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Main training loop.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            resume_from_checkpoint: Optional path to checkpoint to resume from
        """
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # Create data loader
        train_loader = create_data_loader(
            train_dataset,
            batch_size=1,  # Process one sample at a time for variable-length
            shuffle=True,
        )

        print(f"Starting training from step {self.global_step}")
        print(f"Training for {self.config.num_train_steps} steps")
        print(f"Mixed precision: {self.config.mixed_precision}")

        # Training loop
        running_loss = 0.0
        samples_processed = 0

        while self.global_step < self.config.num_train_steps:
            for batch in train_loader:
                # Get single sample (batch_size=1)
                audio = batch["audio"][0]
                text = batch["text"][0]

                # Train step
                metrics = self.train_step(audio, text)
                running_loss += metrics["loss"]
                samples_processed += 1
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = running_loss / samples_processed
                    print(
                        f"Step {self.global_step}/{self.config.num_train_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {metrics['learning_rate']:.6f}"
                    )
                    running_loss = 0.0
                    samples_processed = 0

                # Validation
                if (
                    val_dataset is not None
                    and self.global_step % self.config.eval_steps == 0
                ):
                    val_loss = self.evaluate(val_dataset)
                    print(f"Validation loss: {val_loss:.4f}")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Check if training is complete
                if self.global_step >= self.config.num_train_steps:
                    break

            self.epoch += 1

        print("Training complete!")
        self.save_checkpoint(final=True)

    def evaluate(self, val_dataset: AudioDataset) -> float:
        """
        Evaluate the model on validation set.

        Args:
            val_dataset: Validation dataset

        Returns:
            Average validation loss
        """
        val_loader = create_data_loader(val_dataset, batch_size=1, shuffle=False)

        total_loss = 0.0
        num_samples = 0

        for i, batch in enumerate(val_loader):
            if i >= 100:  # Evaluate on first 100 samples for speed
                break

            audio = batch["audio"][0]
            text = batch["text"][0]

            # Tokenize
            token_ids = self.tokenizer.encode(text)
            token_ids = [1] + token_ids + [2]
            target_ids = np.array([token_ids], dtype=np.int32)

            # Prepare audio
            audio_input = audio.reshape(1, -1, 1).astype(np.float32)

            # Compute loss
            loss, _ = self.compute_loss(audio_input, target_ids)
            total_loss += float(keras.ops.convert_to_numpy(loss))
            num_samples += 1

        return total_loss / max(num_samples, 1)

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / (
            "final" if final else f"checkpoint-{self.global_step}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self.model.preprocessor.preprocess.save_weights(
            str(checkpoint_dir / "preprocessor.weights.h5")
        )
        self.model.encoder.encoder.save_weights(str(checkpoint_dir / "encoder.weights.h5"))
        self.model.decoder.uncached_call.save_weights(
            str(checkpoint_dir / "decoder.weights.h5")
        )

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f)

        print(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)

        # Load model weights
        self.model.preprocessor.preprocess.load_weights(
            str(checkpoint_dir / "preprocessor.weights.h5")
        )
        self.model.encoder.encoder.load_weights(str(checkpoint_dir / "encoder.weights.h5"))
        self.model.decoder.uncached_call.load_weights(
            str(checkpoint_dir / "decoder.weights.h5")
        )

        # Load training state
        with open(checkpoint_dir / "training_state.json", "r") as f:
            state = json.load(f)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]

        print(f"Checkpoint loaded from {checkpoint_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Moonshine ASR model")
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=["tiny", "base"],
        help="Model size to train",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
        help="Path to training manifest file",
    )
    parser.add_argument(
        "--val_manifest", type=str, default=None, help="Path to validation manifest file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./moonshine_checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--num_steps", type=int, default=250000, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per GPU"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.4e-3, help="Learning rate"
    )

    args = parser.parse_args()

    # Get config
    if args.model == "tiny":
        config = get_tiny_config()
    else:
        config = get_base_config()

    # Override config with args
    config.output_dir = args.output_dir
    config.num_train_steps = args.num_steps
    config.batch_size_per_gpu = args.batch_size
    config.learning_rate = args.learning_rate

    # Data config
    data_config = DataConfig(
        train_manifests=[args.train_manifest],
        val_manifests=[args.val_manifest] if args.val_manifest else [],
    )

    # Create datasets
    train_dataset = AudioDataset(
        manifest_paths=data_config.train_manifests,
        sample_rate=config.sample_rate,
        min_duration=config.min_audio_duration,
        max_duration=config.max_audio_duration,
    )

    val_dataset = None
    if data_config.val_manifests:
        val_dataset = AudioDataset(
            manifest_paths=data_config.val_manifests,
            sample_rate=config.sample_rate,
            min_duration=config.min_audio_duration,
            max_duration=config.max_audio_duration,
        )

    # Create trainer
    trainer = MoonshineTrainer(config, data_config)

    # Start training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resume_from_checkpoint=args.resume_from,
    )


if __name__ == "__main__":
    main()

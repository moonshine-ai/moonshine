"""
Training configuration for Moonshine models.

Based on the paper: https://arxiv.org/abs/2410.15608
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training Moonshine models."""

    # Model architecture
    model_name: str = "moonshine/tiny"  # or "moonshine/base"
    dim: int = 288  # 288 for tiny, 416 for base
    inner_dim: int = 288  # 288 for tiny, 416 for base
    n_head: int = 8
    enc_n_layers: int = 6  # 6 for tiny, 8 for base
    dec_n_layers: int = 6  # 6 for tiny, 8 for base
    enc_ff_mult: int = 4
    dec_ff_mult: int = 4
    enc_ff_swiglu: bool = False
    dec_ff_swiglu: bool = True
    vocab_size: int = 32768

    # Training data
    min_audio_duration: float = 4.0  # seconds
    max_audio_duration: float = 30.0  # seconds
    max_segment_gap: float = 2.0  # seconds between segments
    sample_rate: int = 16000

    # Training hyperparameters (from paper)
    num_train_steps: int = 250000
    batch_size_per_gpu: int = 32
    global_batch_size: int = 1024
    gradient_accumulation_steps: int = 1

    # Optimizer settings (AdamW schedule-free)
    learning_rate: float = 1.4e-3
    warmup_steps: int = 8192
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Mixed precision
    mixed_precision: str = "bf16"  # bf16 as in paper

    # Logging and checkpointing
    logging_steps: int = 100
    eval_steps: int = 5000
    save_steps: int = 5000
    save_total_limit: int = 5

    # Data augmentation (optional)
    use_specaugment: bool = False

    # Paths
    output_dir: str = "./moonshine_checkpoints"
    cache_dir: Optional[str] = None

    # Distributed training
    local_rank: int = -1

    # Generation settings for evaluation
    max_tokens_per_second: int = 6  # Heuristic limit to avoid repetitions

    def __post_init__(self):
        """Calculate derived values."""
        if self.global_batch_size % self.batch_size_per_gpu != 0:
            raise ValueError(
                f"global_batch_size ({self.global_batch_size}) must be divisible by "
                f"batch_size_per_gpu ({self.batch_size_per_gpu})"
            )

        # Calculate gradient accumulation if needed
        num_gpus = self.global_batch_size // self.batch_size_per_gpu
        if num_gpus > 1 and self.gradient_accumulation_steps == 1:
            # If using multiple GPUs, we may need gradient accumulation
            pass


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Dataset paths - users should modify these
    train_manifests: list = None  # List of paths to training manifest files
    val_manifests: list = None  # List of paths to validation manifest files

    # Open datasets to use
    use_common_voice: bool = False
    use_librispeech: bool = False
    use_gigaspeech: bool = False
    use_peoples_speech: bool = False
    use_ami_corpus: bool = False

    # Preprocessing
    normalize_audio: bool = True
    remove_silence: bool = False

    # Workers
    num_workers: int = 4
    prefetch_factor: int = 2

    def __post_init__(self):
        """Initialize default values."""
        if self.train_manifests is None:
            self.train_manifests = []
        if self.val_manifests is None:
            self.val_manifests = []


def get_tiny_config() -> TrainingConfig:
    """Get configuration for Moonshine Tiny model."""
    return TrainingConfig(
        model_name="moonshine/tiny",
        dim=288,
        inner_dim=288,
        n_head=8,
        enc_n_layers=6,
        dec_n_layers=6,
    )


def get_base_config() -> TrainingConfig:
    """Get configuration for Moonshine Base model."""
    return TrainingConfig(
        model_name="moonshine/base",
        dim=416,
        inner_dim=416,
        n_head=8,
        enc_n_layers=8,
        dec_n_layers=8,
    )

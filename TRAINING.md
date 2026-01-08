# Training Moonshine Models

This guide explains how to train Moonshine ASR models from scratch or fine-tune existing models.

## Overview

Moonshine models are encoder-decoder transformers optimized for low-latency speech recognition. The training implementation is based on the methodology described in the paper: [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)

## Key Training Details from Paper

- **Training Steps**: 250,000 steps
- **Batch Size**: 32 per GPU (1024 global batch size on 32x H100 GPUs)
- **Optimizer**: AdamW with schedule-free variant
- **Learning Rate**: 1.4e-3 with 8,192 step warmup
- **Precision**: BF16 mixed precision
- **Audio Duration**: 4-30 seconds per training instance
- **Dataset Size**: ~200K hours (90K open + 100K+ internal)

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Additional dependencies for training:

```bash
pip install librosa numpy
```

## Data Preparation

### 1. Create Manifest Files

Training data should be organized in manifest files (JSONL format). Each line should contain:

```json
{"audio": "/path/to/audio.wav", "text": "transcription text", "duration": 10.5}
```

You can create a manifest file using the provided utility:

```python
from moonshine.data_utils import prepare_manifest_from_dataset

audio_paths = ["/path/to/audio1.wav", "/path/to/audio2.wav"]
transcriptions = ["hello world", "this is a test"]

prepare_manifest_from_dataset(
    audio_paths=audio_paths,
    transcriptions=transcriptions,
    output_path="train_manifest.jsonl"
)
```

### 2. Using Open Datasets

The paper uses the following open datasets:
- Common Voice 16.1
- AMI Corpus
- GigaSpeech
- LibriSpeech
- Multilingual LibriSpeech (English subset)
- People's Speech

You can download these datasets and create manifest files from them.

### 3. Data Preprocessing

According to the paper, the training data should be preprocessed as follows:

1. **Audio Duration**: Combine successive segments into 4-30 second instances
2. **Segment Gap**: No more than 2 seconds between segments
3. **No Zero-Padding**: Variable-length sequences (key difference from Whisper)
4. **Sample Rate**: 16kHz mono audio
5. **Normalization**: Normalize audio amplitude

## Training

### Basic Training

Train a Moonshine Tiny model:

```bash
python -m moonshine.train \
    --model tiny \
    --train_manifest train_manifest.jsonl \
    --val_manifest val_manifest.jsonl \
    --output_dir ./checkpoints \
    --num_steps 250000 \
    --batch_size 32 \
    --learning_rate 1.4e-3
```

Train a Moonshine Base model:

```bash
python -m moonshine.train \
    --model base \
    --train_manifest train_manifest.jsonl \
    --val_manifest val_manifest.jsonl \
    --output_dir ./checkpoints \
    --num_steps 250000
```

### Resume from Checkpoint

```bash
python -m moonshine.train \
    --model tiny \
    --train_manifest train_manifest.jsonl \
    --resume_from ./checkpoints/checkpoint-100000 \
    --output_dir ./checkpoints
```

## Configuration

### Model Configurations

**Moonshine Tiny**:
- Dimension: 288
- Encoder layers: 6
- Decoder layers: 6
- Attention heads: 8
- Parameters: 27.1M

**Moonshine Base**:
- Dimension: 416
- Encoder layers: 8
- Decoder layers: 8
- Attention heads: 8
- Parameters: 61.5M

### Custom Configuration

You can create custom configurations by modifying `train_config.py`:

```python
from moonshine.train_config import TrainingConfig

config = TrainingConfig(
    dim=288,
    inner_dim=288,
    n_head=8,
    enc_n_layers=6,
    dec_n_layers=6,
    learning_rate=1.4e-3,
    num_train_steps=250000,
    # ... other parameters
)
```

## Training on Multiple GPUs

The current implementation uses Keras which supports multi-GPU training. For distributed training across multiple nodes, you would need to integrate with frameworks like Horovod or use Accelerate.

Example with TensorFlow distributed strategy:

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Create model and trainer
    trainer = MoonshineTrainer(config, data_config)
    trainer.train(train_dataset, val_dataset)
```

## Fine-tuning

To fine-tune an existing Moonshine model on your own data:

1. Load the pre-trained weights:

```python
from moonshine.model import load_model

model = load_model("moonshine/tiny")

# Now use this model with the trainer
trainer = MoonshineTrainer(config, data_config, model=model)
```

2. Use a lower learning rate (e.g., 1e-4 to 1e-5)

3. Train for fewer steps (e.g., 10,000 - 50,000)

## Evaluation

The trainer automatically evaluates on the validation set every `eval_steps` (default: 5,000 steps).

To evaluate a trained model manually:

```python
from moonshine.train import MoonshineTrainer

trainer = MoonshineTrainer(config, data_config)
trainer.load_checkpoint("./checkpoints/checkpoint-250000")

val_loss = trainer.evaluate(val_dataset)
print(f"Validation loss: {val_loss}")
```

## Checkpointing

Checkpoints are saved every `save_steps` (default: 5,000 steps) and include:
- Model weights (preprocessor, encoder, decoder)
- Training state (global step, epoch)

Checkpoint structure:
```
checkpoints/
├── checkpoint-5000/
│   ├── preprocessor.weights.h5
│   ├── encoder.weights.h5
│   ├── decoder.weights.h5
│   └── training_state.json
├── checkpoint-10000/
│   └── ...
└── final/
    └── ...
```

## Tips for Training

1. **Variable-Length Sequences**: Moonshine supports variable-length audio, which is key to its efficiency. Don't pad audio to fixed lengths.

2. **Batch Size**: The paper uses batch size 32 per GPU. If you have memory constraints, reduce the batch size and increase gradient accumulation steps.

3. **Learning Rate**: The paper uses 1.4e-3 with 8,192 step warmup. This is quite high, so monitor for instability.

4. **Data Quality**: The paper emphasizes filtering noisy labels using Levenshtein distance. Use the `TextNormalizer` class for this.

5. **Duration Distribution**: Aim for a bimodal distribution of 4-30 seconds as shown in the paper (Figure 4).

6. **Gradient Clipping**: The paper uses gradient norm clipping (max_grad_norm in config).

## Tokenizer

Moonshine uses the same byte-level BPE tokenizer as Llama 1/2 with vocabulary size 32,768 (32,000 + 768 special tokens).

To use the proper Llama tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# Make sure to add 768 special tokens to match Moonshine's vocab size
```

## Troubleshooting

**Out of Memory**: Reduce batch size, use gradient checkpointing, or use smaller audio segments.

**Slow Training**: Variable-length sequences are processed individually for efficiency. Consider batching similar-length sequences together.

**Poor Convergence**: Check learning rate, warmup steps, and data quality. The paper uses a relatively high learning rate.

**Repetitions in Output**: Add the heuristic limit of 6 tokens per second of audio (already implemented in the trainer).

## References

- Paper: [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)
- Code: https://github.com/moonshine-ai/moonshine
- Models: https://huggingface.co/UsefulSensors/moonshine

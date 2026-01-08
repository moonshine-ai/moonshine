# Moonshine Training Examples

This directory contains example scripts for training and fine-tuning Moonshine models.

## Quick Start

### 1. Test Training Setup

Run the test script to verify your training setup works:

```bash
python examples/test_training.py
```

This will:
- Create a small synthetic dataset
- Initialize a tiny model
- Run a few training steps
- Verify the training pipeline works correctly

### 2. Prepare Your Data

Create a manifest file from your audio files and transcriptions:

```bash
python examples/prepare_training_data.py \
    --audio_dir /path/to/audio/files \
    --transcripts_file /path/to/transcripts.txt \
    --output_manifest train_manifest.jsonl \
    --normalize_text
```

The transcripts file should have one line per audio file:
```
audio1.wav|hello world
audio2.wav|this is a test
```

### 3. Train a Model

Train a Moonshine Tiny model:

```bash
python -m moonshine.train \
    --model tiny \
    --train_manifest train_manifest.jsonl \
    --val_manifest val_manifest.jsonl \
    --output_dir ./checkpoints \
    --num_steps 100000 \
    --batch_size 32 \
    --learning_rate 1.4e-3
```

## Example Workflows

### Fine-tuning for a Specific Domain

If you want to fine-tune a pre-trained Moonshine model on domain-specific data:

1. **Prepare domain-specific data**:
```bash
python examples/prepare_training_data.py \
    --audio_dir ./medical_audio \
    --transcripts_file ./medical_transcripts.txt \
    --output_manifest medical_train.jsonl
```

2. **Fine-tune with lower learning rate**:
```bash
python -m moonshine.train \
    --model tiny \
    --train_manifest medical_train.jsonl \
    --output_dir ./medical_checkpoints \
    --num_steps 10000 \
    --learning_rate 1e-5
```

### Training from Scratch

For training from scratch with large datasets:

1. **Prepare combined dataset** (combining multiple sources):
```python
from moonshine.data_utils import prepare_manifest_from_dataset

# Combine Common Voice, LibriSpeech, etc.
# See TRAINING.md for details
```

2. **Train with paper configuration**:
```bash
python -m moonshine.train \
    --model base \
    --train_manifest combined_train.jsonl \
    --val_manifest combined_val.jsonl \
    --output_dir ./checkpoints \
    --num_steps 250000 \
    --batch_size 32 \
    --learning_rate 1.4e-3
```

## Scripts

### `test_training.py`

Tests the training pipeline with synthetic data.

**Usage**:
```bash
python examples/test_training.py
```

### `prepare_training_data.py`

Prepares training data and creates manifest files.

**Usage**:
```bash
python examples/prepare_training_data.py \
    --audio_dir <path> \
    --transcripts_file <path> \
    --output_manifest <path> \
    [--normalize_text] \
    [--combine_segments]
```

**Arguments**:
- `--audio_dir`: Directory containing audio files
- `--transcripts_file`: File with transcriptions (format: `filename|text`)
- `--output_manifest`: Output manifest file path
- `--min_duration`: Minimum audio duration (default: 0.5s)
- `--max_duration`: Maximum audio duration (default: 30s)
- `--normalize_text`: Normalize transcription text (lowercase, unicode, etc.)
- `--combine_segments`: Combine short segments into longer instances

## Tips

1. **Start Small**: Use `test_training.py` to verify your setup before training on large datasets

2. **Monitor Training**: Check the loss values and learning rate during training

3. **Checkpoint Often**: The default saves checkpoints every 5,000 steps

4. **Use Validation**: Always provide a validation set to monitor overfitting

5. **Adjust Batch Size**: If you run out of memory, reduce `--batch_size`

## Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Reduce batch size, use shorter audio segments, or use a smaller model

**Issue**: Training is very slow
- **Solution**: Make sure you're using GPU acceleration (check `keras.backend.backend()`)

**Issue**: Loss is not decreasing
- **Solution**: Check data quality, try different learning rates, verify labels match audio

**Issue**: Model produces repetitions
- **Solution**: The trainer includes a heuristic limit of 6 tokens/second (already implemented)

## Next Steps

- Read [TRAINING.md](../TRAINING.md) for detailed training guide
- Check the [paper](https://arxiv.org/abs/2410.15608) for methodology details
- Join the [Discord](https://discord.gg/27qp9zSRXF) for support

"""Log-mel feature front-end and SpecAugment.

``LogMelSpectrogram`` must stay bit-compatible with the on-device feature
generator (``moonshine-micro/feature-generation``): the exported model expects
the same n_mels / hop / normalization the firmware produces at runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class LogMelSpectrogram(nn.Module):
    """Per-clip mean/std normalized log-mel features.

    Output: ``(B, 1, n_mels, target_frames)``. Slaney mel scale, ``n_fft=512``,
    matching the firmware's feature generator.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 125,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float | None = None,
        target_frames: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.target_frames = target_frames
        self.eps = eps
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2,
            power=2.0,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )

    def _fix_length(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        if T == self.target_frames:
            return x
        if T > self.target_frames:
            return x[..., : self.target_frames]
        return F.pad(x, (0, self.target_frames - T))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        mel = self.melspec(waveform)  # (B, n_mels, T)
        mel = torch.log(mel + self.eps)
        mel = self._fix_length(mel)
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
        mel = (mel - mean) / std
        return mel.unsqueeze(1)  # (B, 1, n_mels, target_frames)


class SpecAugment(nn.Module):
    """Time/frequency masking on ``(B, 1, F, T)`` log-mel features (train only)."""

    def __init__(
        self,
        freq_mask_param: int = 24,
        time_mask_param: int = 24,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq = n_freq_masks
        self.n_time = n_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        for _ in range(self.n_freq):
            x = self.freq_mask(x)
        for _ in range(self.n_time):
            x = self.time_mask(x)
        return x

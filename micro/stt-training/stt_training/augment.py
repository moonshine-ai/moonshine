"""GPU-resident waveform augmentation, applied before the log-mel front-end.

Simulates the channel and environment variation the model sees at deployment:
gain swings, polarity flips, small time shifts, colored noise, band-limiting,
plus optional real background noise (MUSAN) and room reverberation (RIRs).

Everything runs as batched tensor ops on the model's device, and the noise /
RIR corpora are decoded once at construction, so steady-state augmentation does
zero disk I/O and no CPU<->GPU syncs. During eval the module is a no-op.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio


class WaveformAugment(nn.Module):
    """Batched waveform augmentation pipeline.

    Order: Gain -> PolarityInversion -> Shift -> AddColoredNoise ->
    BandPassFilter -> AddBackgroundNoise (MUSAN) -> ApplyImpulseResponse (RIR).

    The MUSAN and RIR stages are only added when their directories exist, so
    the pipeline still runs on a fresh checkout with synthetic noise only.
    Inputs/outputs are 1-D waveforms ``(B, T)``.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        musan_noise_dir: Path | None = None,
        rir_dir: Path | None = None,
        gain_db: float = 12.0,
        noise_snr_min: float = 5.0,
        noise_snr_max: float = 30.0,
        bandpass_p: float = 0.15,
        max_rirs: int = 2048,
        max_noise_seconds: float = 1200.0,
        seed: int = 0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.gain_db = float(gain_db)
        self.noise_snr_min = float(noise_snr_min)
        self.noise_snr_max = float(noise_snr_max)
        self.bandpass_p = float(bandpass_p)
        self.p_gain = 0.7
        self.p_polarity = 0.5
        self.p_shift = 0.5
        self.p_colored = 0.5
        self.p_bg = 0.5
        self.p_rir = 0.3
        self.max_shift = int(round(0.05 * sample_rate))  # +/-50 ms
        self._rng = random.Random(seed)

        noise_buf = self._load_concat_noise(musan_noise_dir, max_noise_seconds)
        self._has_bg = noise_buf is not None and noise_buf.numel() > self.sample_rate
        if not self._has_bg:
            noise_buf = torch.zeros(self.sample_rate, dtype=torch.float32)
        self.register_buffer("noise_concat", noise_buf, persistent=False)

        rir_buf, rir_peak = self._load_rirs(rir_dir, max_rirs)
        self._has_rir = rir_buf is not None
        if not self._has_rir:
            rir_buf = torch.zeros(1, 1, dtype=torch.float32)
            rir_peak = torch.zeros(1, dtype=torch.long)
        self.register_buffer("rirs", rir_buf, persistent=False)       # (N, L)
        self.register_buffer("rir_peak", rir_peak, persistent=False)  # (N,)

        self._n_transforms = 5 + int(self._has_bg) + int(self._has_rir)
        self._has_external_data = self._has_bg or self._has_rir

    # ---- one-time data loading -------------------------------------------
    def _load_concat_noise(self, noise_dir, max_seconds):
        if noise_dir is None or not Path(noise_dir).is_dir():
            return None
        files = sorted(Path(noise_dir).rglob("*.wav"))
        if not files:
            return None
        budget = int(max_seconds * self.sample_rate)
        chunks, total = [], 0
        for f in files:
            try:
                data, sr = sf.read(str(f), dtype="float32", always_2d=True)
            except Exception:  # noqa: BLE001
                continue
            t = torch.from_numpy(data.mean(axis=1))
            if sr != self.sample_rate:
                t = torchaudio.functional.resample(t, sr, self.sample_rate)
            chunks.append(t)
            total += t.numel()
            if total >= budget:
                break
        if not chunks:
            return None
        return torch.cat(chunks)[:budget].contiguous()

    def _load_rirs(self, rir_dir, max_rirs):
        if rir_dir is None or not Path(rir_dir).is_dir():
            return None, None
        files = sorted(Path(rir_dir).rglob("*.wav"))
        if not files:
            return None, None
        if len(files) > max_rirs:
            self._rng.shuffle(files)
            files = files[:max_rirs]
        rirs = []
        for f in files:
            try:
                data, sr = sf.read(str(f), dtype="float32", always_2d=True)
            except Exception:  # noqa: BLE001
                continue
            t = torch.from_numpy(data.mean(axis=1))
            if sr != self.sample_rate:
                t = torchaudio.functional.resample(t, sr, self.sample_rate)
            peak = t.abs().max()
            if peak > 0:
                t = t / peak  # unit peak -> pure reverberation, no gain change
            rirs.append(t)
        if not rirs:
            return None, None
        length = max(t.numel() for t in rirs)
        buf = torch.zeros(len(rirs), length, dtype=torch.float32)
        peaks = torch.zeros(len(rirs), dtype=torch.long)
        for i, t in enumerate(rirs):
            buf[i, : t.numel()] = t
            peaks[i] = int(torch.argmax(t.abs()).item())
        return buf, peaks

    @property
    def n_transforms(self) -> int:
        return self._n_transforms

    @property
    def has_external_data(self) -> bool:
        return self._has_external_data

    # ---- batched transforms ----------------------------------------------
    @staticmethod
    def _mask(p, b, dev):
        return torch.rand(b, device=dev) < p

    @staticmethod
    def _rms(x):
        return torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-12)

    def _gain(self, x, b, dev):
        m = self._mask(self.p_gain, b, dev)
        db = torch.empty(b, device=dev).uniform_(-self.gain_db, self.gain_db / 2)
        factor = torch.where(m, torch.pow(10.0, db / 20.0), torch.ones_like(db))
        return x * factor.unsqueeze(1)

    def _polarity(self, x, b, dev):
        m = self._mask(self.p_polarity, b, dev).float().unsqueeze(1)
        return x * (1.0 - 2.0 * m)

    def _shift(self, x, b, t, dev):
        m = self._mask(self.p_shift, b, dev)
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (b,), device=dev)
        shifts = torch.where(m, shifts, torch.zeros_like(shifts))
        src = torch.arange(t, device=dev).unsqueeze(0) - shifts.unsqueeze(1)
        valid = (src >= 0) & (src < t)
        gathered = torch.gather(x, 1, src.clamp(0, t - 1))
        return gathered * valid.to(x.dtype)  # rollover=False -> zero-pad

    def _snr_scale(self, sig, noise, b, dev):
        snr_db = torch.empty(b, 1, device=dev).uniform_(
            self.noise_snr_min, self.noise_snr_max
        )
        target_noise_rms = self._rms(sig) / torch.pow(10.0, snr_db / 20.0)
        return noise * (target_noise_rms / self._rms(noise))

    def _colored_noise(self, x, b, t, dev):
        m = self._mask(self.p_colored, b, dev)
        white = torch.randn(b, t, device=dev)
        freqs = torch.fft.rfftfreq(t, d=1.0 / self.sample_rate, device=dev)
        f = freqs.clone()
        if f.numel() > 1:
            f[0] = f[1]
        decay = torch.empty(b, 1, device=dev).uniform_(-2.0, 2.0)
        scale = f.unsqueeze(0).pow(-decay / 2.0)
        noise = torch.fft.irfft(torch.fft.rfft(white, dim=1) * scale, n=t, dim=1)
        out = x + self._snr_scale(x, noise, b, dev)
        return torch.where(m.unsqueeze(1), out, x)

    def _bandpass(self, x, b, t, dev):
        m = self._mask(self.bandpass_p, b, dev)
        if not bool(m.any()):
            return x
        freqs = torch.fft.rfftfreq(t, d=1.0 / self.sample_rate, device=dev)
        lo, hi = 300.0, 4000.0
        u = torch.rand(b, 1, device=dev)
        center = torch.exp(u * (math.log(hi) - math.log(lo)) + math.log(lo))
        bw = center * torch.empty(b, 1, device=dev).uniform_(0.5, 1.99)
        low = (center - bw / 2).clamp(min=0.0)
        high = center + bw / 2
        fb = freqs.unsqueeze(0)
        passband = ((fb >= low) & (fb <= high)).to(x.dtype)
        filtered = torch.fft.irfft(torch.fft.rfft(x, dim=1) * passband, n=t, dim=1)
        return torch.where(m.unsqueeze(1), filtered, x)

    def _bg_noise(self, x, b, t, dev):
        m = self._mask(self.p_bg, b, dev)
        n = int(self.noise_concat.numel())
        starts = torch.randint(0, max(1, n - t), (b,), device=dev)
        idx = starts.unsqueeze(1) + torch.arange(t, device=dev).unsqueeze(0)
        noise = self.noise_concat[idx]
        out = x + self._snr_scale(x, noise, b, dev)
        return torch.where(m.unsqueeze(1), out, x)

    def _rir(self, x, b, t, dev):
        m = self._mask(self.p_rir, b, dev)
        if not bool(m.any()):
            return x
        n_rir, length = self.rirs.shape
        sel = torch.randint(0, n_rir, (b,), device=dev)
        h = self.rirs[sel]
        peak = self.rir_peak[sel]
        n = t + length - 1
        nfft = 1 << ((n - 1).bit_length())
        y = torch.fft.irfft(
            torch.fft.rfft(x, n=nfft, dim=1) * torch.fft.rfft(h, n=nfft, dim=1),
            n=nfft, dim=1,
        )
        gidx = peak.unsqueeze(1) + torch.arange(t, device=dev).unsqueeze(0)
        y = torch.gather(y, 1, gidx.clamp(0, nfft - 1))
        y = y * (self._rms(x) / self._rms(y))  # preserve loudness
        return torch.where(m.unsqueeze(1), y, x)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return waveform
        x = waveform
        added_channel = False
        if x.dim() == 3:  # (B, 1, T) -> (B, T)
            x = x.squeeze(1)
            added_channel = True
        b, t = x.shape
        dev = x.device
        x = self._gain(x, b, dev)
        x = self._polarity(x, b, dev)
        x = self._shift(x, b, t, dev)
        x = self._colored_noise(x, b, t, dev)
        x = self._bandpass(x, b, t, dev)
        if self._has_bg:
            x = self._bg_noise(x, b, t, dev)
        if self._has_rir:
            x = self._rir(x, b, t, dev)
        if added_channel:
            x = x.unsqueeze(1)
        return x

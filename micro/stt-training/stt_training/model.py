"""WordCNN: the compact MobileNetV2-style classifier deployed on the RP2350.

The architecture is identical to the ``SpellingCNN`` used by the shipped
moonshine-micro spelling example, so exports produced here are drop-in for the
existing ``moonshine-micro/stt`` firmware (same op set, same arena budget).
Only the number of output classes changes with the vocabulary.
"""

from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: int, divisor: int = 8) -> int:
    """Round channel counts to a multiple of ``divisor`` (mobile-friendly)."""
    new_v = max(divisor, (v + divisor // 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def normalize_stride(v: int | str | Sequence[int]) -> tuple[int, int]:
    """Coerce an int / ``"2,2"`` string / tuple into ``(freq_stride, time_stride)``."""
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, str):
        parts = [p for p in re.split(r"[,x ]+", v.strip()) if p]
    else:
        parts = [str(int(p)) for p in v]
    try:
        vals = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"invalid stem stride spec: {v!r}") from exc
    if len(vals) == 1:
        vals = [vals[0], vals[0]]
    if len(vals) != 2 or any(s < 1 for s in vals):
        raise ValueError(f"stem stride must be one or two positive ints, got {v!r}")
    return (vals[0], vals[1])


class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, kernel=3, stride=1, groups=1, act=True):
        pad = (kernel - 1) // 2
        layers = [
            nn.Conv2d(in_c, out_c, kernel, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual: 1x1 expand -> 3x3 depthwise -> 1x1 project."""

    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        assert stride in (1, 2)
        hidden = in_c * expand_ratio
        self.use_residual = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_c, hidden, kernel=1))
        layers += [
            ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class WordCNN(nn.Module):
    """MobileNetV2-style classifier for single-channel log-mel spectrograms.

    Input:  ``(B, 1, F, T)`` normalized log-mel features.
    Output: ``(B, num_classes)`` logits.
    """

    # (expand_ratio, out_channels, num_blocks, stride_of_first_block)
    CONFIG = [
        (1, 16, 1, 1),
        (4, 24, 2, 2),
        (4, 32, 3, 2),
        (4, 64, 3, 2),
        (4, 96, 2, 1),
        (4, 160, 2, 2),
        (4, 240, 1, 1),
    ]

    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        stem_stride: int | tuple[int, int] = (2, 2),
        pad_to_odd: bool = True,
    ):
        super().__init__()

        # When True, the forward pass pads the (even) input by +1 on each
        # spatial axis so every stride-2 conv sees an ODD input. For an odd
        # input, PyTorch's symmetric padding=1 matches TFLite SAME padding, so
        # the converter fuses it into the conv instead of emitting standalone
        # PAD ops that materialise a full padded copy of the (large, early)
        # activation -- the TFLM arena peak. Trading those PADs for one tiny
        # PAD on the raw input roughly halves peak activation memory. Assumes
        # even input dims (the deployed 64x128 config).
        self.pad_to_odd = bool(pad_to_odd)

        self.stem_stride = normalize_stride(stem_stride)
        stem_c = _make_divisible(int(32 * width_mult))
        # Stem stride is (freq, time). (2, 2) halves both axes early, which
        # quarters the first stride-2 block's expand activation (the TFLM arena
        # bottleneck) so a higher-resolution mel input fits the same budget.
        self.stem = ConvBNAct(1, stem_c, kernel=3, stride=self.stem_stride)

        blocks = []
        in_c = stem_c
        for expand, out_c, n, first_stride in self.CONFIG:
            out_c = _make_divisible(int(out_c * width_mult))
            for i in range(n):
                stride = first_stride if i == 0 else 1
                blocks.append(InvertedResidual(in_c, out_c, stride, expand))
                in_c = out_c
        self.blocks = nn.Sequential(*blocks)

        head_c = _make_divisible(int(640 * width_mult))
        self.head_conv = ConvBNAct(in_c, head_c, kernel=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_c, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.pad_to_odd:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(
    num_classes: int,
    *,
    width_mult: float = 1.0,
    stem_stride: int | tuple[int, int] = (2, 2),
    pad_to_odd: bool = True,
) -> WordCNN:
    return WordCNN(
        num_classes=num_classes,
        width_mult=width_mult,
        stem_stride=stem_stride,
        pad_to_odd=pad_to_odd,
    )

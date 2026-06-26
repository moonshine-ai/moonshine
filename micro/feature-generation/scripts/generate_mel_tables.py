#!/usr/bin/env python3
"""Emit C++ flash tables (periodic Hann window + CSR Slaney mel filterbank) for
the feature-generation module.

The output `mel_tables.{h,cc}` is exactly what LogMelSpectrogram / MelStreamer
adopt via LogMelParams::precomputed_* (so the on-device front-end does no heap
allocation or trig/log work at boot). The maths here is byte-for-byte identical
to feature-generation/src/log_mel.cc (HzToMelSlaney / MakeMelFilterbank /
HannWindowPeriodic).

This generator is intentionally model-independent: pass the front-end config on
the command line. The STT example regenerates its tables from the exported
model's metadata via stt/scripts/generate_embedded_data.py, which uses the same
formulas; this script is the standalone path for reusing the module elsewhere.

Example (the checked-in 64-mel STT front-end config):
  python generate_mel_tables.py \
      --sample-rate 16000 --n-fft 512 --win-length 512 \
      --n-mels 64 --f-min 20 --f-max 8000 \
      --namespace spelling --prefix kMel \
      --out-dir ../../examples/rp2350/generated
"""
import argparse
import math
import os


def _include_guard(stem: str) -> str:
    return f"SPELLING_{stem.upper().replace('-', '_')}_H_"


def hann_window_periodic(length):
    if length <= 0:
        return []
    n = length + 1
    denom = float(n - 1)
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / denom) for i in range(length)]


def hz_to_mel_slaney(hz):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = math.log(6.4) / 27.0
    if hz >= min_log_hz:
        return min_log_mel + math.log(hz / min_log_hz) / log_step
    return hz / f_sp


def mel_to_hz_slaney(mel):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = math.log(6.4) / 27.0
    if mel >= min_log_mel:
        return min_log_hz * math.exp(log_step * (mel - min_log_mel))
    return f_sp * mel


def make_csr_filterbank(n_freq, n_mels, sample_rate, f_min, f_max):
    mel_min = hz_to_mel_slaney(f_min)
    mel_max = hz_to_mel_slaney(f_max)
    hz_pts = [
        mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i / (n_mels + 1))
        for i in range(n_mels + 2)
    ]
    nz_off = [0]
    nz_idx = []
    nz_val = []
    for m in range(n_mels):
        f_left, f_center, f_right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        enorm = 2.0 / (f_right - f_left)
        cnt = 0
        for k in range(n_freq):
            f = sample_rate * 0.5 * k / (n_freq - 1)
            if f <= f_left or f >= f_right:
                continue
            w = ((f - f_left) / (f_center - f_left) if f <= f_center
                 else (f_right - f) / (f_right - f_center))
            val = w * enorm
            if val != 0.0:
                nz_idx.append(k)
                nz_val.append(val)
                cnt += 1
        nz_off.append(nz_off[-1] + cnt)
    return nz_off, nz_idx, nz_val


def fmt_floats(values, per_line=8):
    out = []
    for i in range(0, len(values), per_line):
        chunk = ", ".join(f"{v:.8e}f" for v in values[i:i + per_line])
        out.append("    " + chunk + ",")
    return "\n".join(out)


def fmt_ints(values, per_line=16):
    out = []
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(v) for v in values[i:i + per_line])
        out.append("    " + chunk + ",")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--win-length", type=int, default=512)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--f-min", type=float, default=20.0)
    ap.add_argument("--f-max", type=float, default=8000.0)
    ap.add_argument("--namespace", default="spelling")
    ap.add_argument("--prefix", default="kMel",
                    help="symbol prefix (kMel for STT, kVadMel for VAD)")
    ap.add_argument("--basename", default="mel_tables",
                    help="output file basename (mel_tables / vad_mel_tables)")
    ap.add_argument("--const-prefix", default="kMelTable",
                    help="prefix for the size constants (kMelTable / kVadMelTable)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    n_freq = args.n_fft // 2 + 1
    window = hann_window_periodic(args.win_length)
    # Centre the window inside n_fft (zero pad) -- matches log_mel.cc.
    if args.win_length < args.n_fft:
        left = (args.n_fft - args.win_length) // 2
        window = [0.0] * left + window + [0.0] * (args.n_fft - args.win_length - left)
    nz_off, nz_idx, nz_val = make_csr_filterbank(
        n_freq, args.n_mels, args.sample_rate, args.f_min, args.f_max)
    nz_total = nz_off[-1]

    ns = args.namespace
    p = args.prefix
    cp = args.const_prefix
    win = f"{p}Window"
    base = args.basename
    guard_note = (f"sample_rate={args.sample_rate}, n_fft={args.n_fft}, "
                  f"win_length={args.win_length}, n_mels={args.n_mels}, "
                  f"n_freq={n_freq}, f_min={args.f_min}, f_max={args.f_max}")

    header = f"""// AUTO-GENERATED by feature-generation/scripts/generate_mel_tables.py
// DO NOT EDIT.
//
// Precomputed log-mel front-end tables (periodic Hann window + CSR Slaney mel
// filterbank). These are `const` so they live in flash; LogMelSpectrogram /
// MelStreamer point straight at them and do NO heap allocation or trig/log work.
//
// Generated for: {guard_note}.

#ifndef {_include_guard(base)}
#define {_include_guard(base)}

namespace {ns} {{

constexpr int {cp}SampleRate = {args.sample_rate};
constexpr int {cp}NFft       = {args.n_fft};
constexpr int {cp}WinLength  = {args.win_length};
constexpr int {cp}NMels      = {args.n_mels};
constexpr int {cp}NFreq      = {n_freq};
constexpr int {p}NzTotal     = {nz_total};

// Periodic Hann window, centred in n_fft (length {cp}NFft).
extern const float {win}[{cp}NFft];

// CSR Slaney mel filterbank. Row m occupies [{p}NzOff[m], {p}NzOff[m + 1]) of
// {p}NzIdx (FFT bin column) / {p}NzVal (triangular weight).
extern const int   {p}NzOff[{cp}NMels + 1];
extern const int   {p}NzIdx[{p}NzTotal];
extern const float {p}NzVal[{p}NzTotal];

}}  // namespace {ns}

#endif  // {_include_guard(base)}
"""

    source = f"""// AUTO-GENERATED by feature-generation/scripts/generate_mel_tables.py
// DO NOT EDIT.

#include "{base}.h"

namespace {ns} {{

const float {win}[{cp}NFft] = {{
{fmt_floats(window)}
}};

const int {p}NzOff[{cp}NMels + 1] = {{
{fmt_ints(nz_off)}
}};

const int {p}NzIdx[{p}NzTotal] = {{
{fmt_ints(nz_idx)}
}};

const float {p}NzVal[{p}NzTotal] = {{
{fmt_floats(nz_val)}
}};

}}  // namespace {ns}
"""

    os.makedirs(args.out_dir, exist_ok=True)
    h_path = os.path.join(args.out_dir, f"{base}.h")
    cc_path = os.path.join(args.out_dir, f"{base}.cc")
    with open(h_path, "w") as f:
        f.write(header)
    with open(cc_path, "w") as f:
        f.write(source)
    print(f"wrote {h_path} and {cc_path} "
          f"(n_mels={args.n_mels}, nz_total={nz_total})")


if __name__ == "__main__":
    main()

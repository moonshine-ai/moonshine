"""Generate the compiled-in data blobs for the moonshine-micro Pico build.

The Pico has no filesystem, so the model file and the test wavs are
emitted as C arrays linked into the binary. This script produces three
pairs of files under moonshine-micro/generated/:

    model_data.{h,cc}    -- the int8 mel TFLite classifier as an
                            ``alignas(16) const unsigned char[]`` blob
                            (alignment needed by TFLM's flatbuffer reader)
    classes.{h,cc}       -- the 36 class labels (read from spelling_cnn_meta.json)
    test_clips.{h,cc}    -- N clips per class, decoded with the same
                            stdlib WAV reader the Python predictor uses,
                            pad/cropped to 1 second @ 16 kHz, scaled to
                            int16 PCM. Each clip carries its source path
                            and the integer label index so the on-device
                            test loop can score accuracy directly.

Defaults are picked to be reasonable for the Pico 2 / RP2350 (4 MB QSPI
flash). With ``--clips-per-class 2`` we ship 72 clips x 32 KB = ~2.3 MB
of test data alongside the 1.3 MB model and ~300 KB of TFLM code, which
fits comfortably under the 4 MB ceiling.

Usage::

    python moonshine-micro/scripts/generate_embedded_data.py
    python moonshine-micro/scripts/generate_embedded_data.py --clips-per-class 1
    python moonshine-micro/scripts/generate_embedded_data.py --max-classes 4
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys
import textwrap

# This script lives at moonshine-micro/stt/scripts/.
MOONSHINE_MICRO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = MOONSHINE_MICRO_ROOT.parent
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Use the SAME stdlib-only WAV reader the Python predictor uses, so the
# fp32 samples we quantize to int16 here are byte-for-byte what the
# Python pipeline sees. This is load-bearing for parity: any divergence
# in decoding would show up as a different log-mel and a different
# prediction on-device.
from models.log_mel_pure import (  # noqa: E402
    read_wav_mono,
    resample_linear,
    pad_or_crop,
    hann_window,
    make_mel_filterbank,
)


def _include_guard(stem: str) -> str:
    """Traditional include-guard macro for a generated header stem (no ext)."""
    return f"SPELLING_{stem.upper().replace('-', '_')}_H_"


def _resolve_int8_mel_tflite(explicit: pathlib.Path | None = None) -> pathlib.Path:
    """Return the checked-in SpellingCNN int8 model under moonshine-micro/models/."""
    if explicit is not None:
        p = explicit.expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file():
            return p
        raise FileNotFoundError(f"No TFLite model at {p}")
    for name in (
        "spelling_cnn_mel_int8.tflite",
        "spelling_cnn_letters_digits_mel_int8.tflite",
    ):
        cand = MOONSHINE_MICRO_ROOT / "models" / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "No SpellingCNN int8 model found under moonshine-micro/models/. "
        "Pass --tflite explicitly if using a different export."
    )


def _meta_sidecar(tflite: pathlib.Path) -> pathlib.Path:
    # Prefer the canonical sidecar name; fall back to the legacy
    # letters+digits filename when both exist in moonshine-micro/models/.
    for name in (
        "spelling_cnn_meta.json",
        "spelling_cnn_letters_digits_meta.json",
    ):
        cand = tflite.parent / name
        if cand.is_file():
            return cand
    return tflite.parent / "spelling_cnn_meta.json"


def _load_meta(tflite: pathlib.Path) -> dict:
    """Load the model's audio metadata, or die loudly if it's missing.

    HARD-FAIL design: the on-device log-mel front-end MUST exactly match
    the front-end the model expects. Silently defaulting any of
    n_mels / target_frames / hop_length / n_fft to legacy values was a silent
    corruption mode -- the C++ build would succeed and the firmware would halt
    inside TFLM AllocateTensors() with no visible error.
    """
    sidecar = _meta_sidecar(tflite)
    if not sidecar.is_file():
        raise SystemExit(
            f"ERROR: no metadata sidecar found next to {tflite.name}\n"
            f"       Expected spelling_cnn_letters_digits_meta.json (or "
            f"spelling_cnn_meta.json) in {tflite.parent}.\n"
            f"       The sidecar carries n_mels / target_frames / hop_length / "
            f"n_fft and MUST match the model exactly."
        )
    meta = json.loads(sidecar.read_text())
    required = ("n_mels", "target_frames", "hop_length")
    missing = [k for k in required if k not in meta]
    if missing:
        raise SystemExit(
            f"ERROR: sidecar {sidecar} is missing required key(s) {missing}.\n"
            f"       Regenerate or fix the metadata JSON alongside the model."
        )
    meta.setdefault("classes", [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
        "y", "z", "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine",
    ])
    meta.setdefault("sample_rate", 16000)
    meta.setdefault("clip_seconds", 1.0)
    # n_fft defaults to 512 when omitted from older sidecars.
    meta.setdefault("n_fft", 512)
    return meta


def _format_byte_array(data: bytes, width: int = 16) -> str:
    """Render `data` as comma-separated 0x.. literals, wrapping every `width`."""
    lines = []
    for off in range(0, len(data), width):
        chunk = data[off:off + width]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    return "\n".join(lines)


def _format_int16_array(samples: list[int], width: int = 12) -> str:
    """Render signed int16 samples as comma-separated literals."""
    lines = []
    for off in range(0, len(samples), width):
        chunk = samples[off:off + width]
        lines.append("  " + ", ".join(f"{v:>6}" for v in chunk) + ",")
    return "\n".join(lines)


def _f32(x: float) -> float:
    """Round a Python float (f64) to the nearest IEEE-754 float32 value.

    The on-device tables are ``float`` arrays, so we bake exactly the
    float32 value the compiler will store. ``repr()`` of the result is
    the shortest decimal that round-trips back to that float32, so the
    emitted literal is unambiguous.
    """
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _format_float32_array(vals: list[float], width: int = 8) -> str:
    """Render floats as comma-separated C ``float`` literals (``...f``)."""
    lines = []
    for off in range(0, len(vals), width):
        chunk = vals[off:off + width]
        lines.append("  " + ", ".join(f"{_f32(v)!r}f" for v in chunk) + ",")
    return "\n".join(lines)


def _format_plain_int_array(vals: list[int], width: int = 16) -> str:
    """Render ints as comma-separated literals (no width padding)."""
    lines = []
    for off in range(0, len(vals), width):
        chunk = vals[off:off + width]
        lines.append("  " + ", ".join(str(v) for v in chunk) + ",")
    return "\n".join(lines)


def _fp32_to_int16_pcm(samples: list[float]) -> list[int]:
    """Symmetric 16-bit quantization with saturation.

    Matches the inverse of `read_wav_mono`'s scaling (which divides
    int16 PCM by 32768). We multiply by 32767 (not 32768) so a sample of
    +1.0 maps to INT16_MAX rather than wrapping to -32768, and we clip
    out-of-range values explicitly because audio mastering occasionally
    produces samples slightly above +1.0 in float-WAV captures.
    """
    out = []
    for s in samples:
        v = int(round(s * 32767.0))
        if v >  32767: v =  32767
        if v < -32768: v = -32768
        out.append(v)
    return out


def _is_riff_wav(path: pathlib.Path) -> bool:
    """True when ``path`` looks like a real PCM wav (not a Git LFS pointer)."""
    try:
        with path.open("rb") as f:
            return f.read(4) == b"RIFF"
    except OSError:
        return False


def _load_clip(path: pathlib.Path, sample_rate: int, n_samples: int) -> list[int]:
    """Read a wav, conform to (sample_rate, n_samples), return int16 PCM."""
    samples, src_rate = read_wav_mono(path)
    if src_rate != sample_rate:
        samples = resample_linear(samples, src_rate, sample_rate)
    samples = pad_or_crop(samples, n_samples)
    return _fp32_to_int16_pcm(samples)


# ---------------------------------------------------------------------------
# Code generators

def _write_model_files(out_dir: pathlib.Path, tflite: pathlib.Path) -> None:
    data = tflite.read_bytes()
    name = "g_spelling_model_data"
    size_name = "g_spelling_model_data_size"

    header = textwrap.dedent(f"""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT. Regenerate with:
        //   python moonshine-micro/scripts/generate_embedded_data.py
        //
        // Source: {tflite.relative_to(MOONSHINE_MICRO_ROOT)}
        // Bytes:  {len(data)}

        #ifndef {_include_guard("model_data")}
        #define {_include_guard("model_data")}
        #include <cstdint>

        extern const unsigned int {size_name};
        // 16-byte alignment is what TFLM's flatbuffer reader expects.
        extern const unsigned char {name}[];

        #endif  // {_include_guard("model_data")}
        """)
    (out_dir / "model_data.h").write_text(header)

    body = textwrap.dedent(f"""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT.

        #include "model_data.h"

        const unsigned int {size_name} = {len(data)};

        alignas(16) const unsigned char {name}[] = {{
        """)
    body += _format_byte_array(data) + "\n};\n"
    (out_dir / "model_data.cc").write_text(body)
    print(f"  model_data.{{h,cc}}: {len(data):,} bytes")


def _write_audio_config_file(
    out_dir: pathlib.Path,
    *,
    sample_rate: int,
    clip_seconds: float,
    n_mels: int,
    target_frames: int,
    hop_length: int,
    n_fft: int,
) -> None:
    """Emit ``audio_config.h`` with the model's mel-front-end constants.

    Generated from the model metadata sidecar so dimension changes propagate
    into the firmware without hand-editing ``main.cc``. ``f_max`` is the Nyquist
    frequency (sample_rate / 2).

    ``n_fft`` MUST match the value baked into the model: the Slaney mel
    filterbank is computed from the FFT freq bins, so a mismatch silently
    corrupts predictions even if the rest of the audio config agrees.
    """
    n_samples = int(round(sample_rate * clip_seconds))
    f_max = sample_rate / 2.0

    header = textwrap.dedent(f"""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT. Regenerate with:
        //   python moonshine-micro/scripts/generate_embedded_data.py
        //
        // Audio / mel-front-end constants from the model metadata sidecar.
        // main.cc and log_mel use these instead of hard-coded dimensions.

        #ifndef {_include_guard("audio_config")}
        #define {_include_guard("audio_config")}

        namespace spelling {{

        constexpr int   kSampleRate     = {sample_rate};
        constexpr float kClipSeconds    = {clip_seconds:.6f}f;
        constexpr int   kClipNumSamples = {n_samples};
        constexpr int   kNMels          = {n_mels};
        constexpr int   kTargetFrames   = {target_frames};
        constexpr int   kHopLength      = {hop_length};
        constexpr int   kNFft           = {n_fft};
        constexpr int   kWinLength      = {n_fft};
        constexpr float kFMin           = 20.0f;
        constexpr float kFMax           = {f_max:.1f}f;

        }}  // namespace spelling

        #endif  // {_include_guard("audio_config")}
        """)
    (out_dir / "audio_config.h").write_text(header)
    print(
        f"  audio_config.h: n_mels={n_mels} target_frames={target_frames} "
        f"hop_length={hop_length} n_fft={n_fft} sr={sample_rate} "
        f"clip={clip_seconds}s ({n_samples} samples)"
    )


def _write_mel_tables_file(
    out_dir: pathlib.Path,
    *,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    win_length: int,
    f_min: float,
    f_max: float,
) -> None:
    """Emit ``mel_tables.{h,cc}``: the precomputed Hann window + CSR mel
    filterbank, as ``const`` arrays the linker places in flash.

    This lets the on-device ``LogMelSpectrogram`` SKIP all front-end
    construction work (the triangular Slaney filterbank, the periodic
    Hann window) and instead point straight at these read-only tables.
    The win + filterbank therefore cost ZERO RAM (they live in XIP
    flash) and ZERO boot-time trig/log -- the constructor just stores
    four pointers.

    PARITY: the window and filterbank are computed by the SAME reference
    functions the desktop pipeline uses (``models.log_mel_pure``'s
    ``hann_window`` and ``make_mel_filterbank``), which the on-device
    C++ ``HannWindowPeriodic`` / ``MakeMelFilterbank`` were written to
    mirror. Values are baked as exact float32 literals (see ``_f32``).
    The CSR sparsity pattern (which (m, k) entries are nonzero) matches
    ``PureLogMelSpectrogram._apply_mel``'s ``fb_row[k] != 0.0`` filter,
    so the device sums exactly the same terms as the reference.
    """
    n_freq = n_fft // 2 + 1

    # Periodic Hann of win_length, centred inside a length-n_fft buffer
    # (a no-op when win_length == n_fft, which is our default). Mirrors
    # LogMelSpectrogram's window_ construction exactly.
    raw = hann_window(win_length, periodic=True)
    if len(raw) == n_fft:
        window = list(raw)
    else:
        pad_total = n_fft - len(raw)
        left = pad_total // 2
        window = [0.0] * left + list(raw) + [0.0] * (pad_total - left)
    if len(window) != n_fft:
        raise SystemExit(
            f"internal error: window length {len(window)} != n_fft {n_fft}"
        )

    # Dense Slaney filterbank -> CSR (per-row nonzero index/value lists).
    fb = make_mel_filterbank(n_freq, n_mels, sample_rate, f_min, f_max)
    nz_off = [0] * (n_mels + 1)
    nz_idx: list[int] = []
    nz_val: list[float] = []
    for m in range(n_mels):
        cnt = 0
        row = fb[m]
        for k in range(n_freq):
            v = row[k]
            if v != 0.0:
                nz_idx.append(k)
                nz_val.append(v)
                cnt += 1
        nz_off[m + 1] = nz_off[m] + cnt
    total_nz = nz_off[n_mels]

    header = textwrap.dedent(f"""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT. Regenerate with:
        //   python moonshine-micro/scripts/generate_embedded_data.py
        //
        // Precomputed log-mel front-end tables (periodic Hann window +
        // CSR Slaney mel filterbank) for the exported model's audio
        // config. These are `const` so they live in flash; the on-device
        // LogMelSpectrogram points straight at them and does NO heap
        // allocation or trig/log work for the front-end.
        //
        // Generated for: sample_rate={sample_rate}, n_fft={n_fft},
        //   win_length={win_length}, n_mels={n_mels}, n_freq={n_freq},
        //   f_min={f_min}, f_max={f_max}. Compile-time asserts in main.cc
        //   check these against audio_config.h.

        #ifndef {_include_guard("mel_tables")}
        #define {_include_guard("mel_tables")}

        namespace spelling {{

        constexpr int kMelTableSampleRate = {sample_rate};
        constexpr int kMelTableNFft       = {n_fft};
        constexpr int kMelTableWinLength  = {win_length};
        constexpr int kMelTableNMels      = {n_mels};
        constexpr int kMelTableNFreq      = {n_freq};
        constexpr int kMelNzTotal         = {total_nz};

        // Periodic Hann window, centred in n_fft (length kMelTableNFft).
        extern const float kMelWindow[kMelTableNFft];

        // CSR Slaney mel filterbank. Row m occupies
        //   [kMelNzOff[m], kMelNzOff[m + 1])
        // of kMelNzIdx / kMelNzVal; kMelNzIdx is the FFT bin column and
        // kMelNzVal the (Slaney-area-normalised) triangular weight.
        extern const int   kMelNzOff[kMelTableNMels + 1];
        extern const int   kMelNzIdx[kMelNzTotal];
        extern const float kMelNzVal[kMelNzTotal];

        }}  // namespace spelling

        #endif  // {_include_guard("mel_tables")}
        """)
    (out_dir / "mel_tables.h").write_text(header)

    body = textwrap.dedent("""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT.

        #include "mel_tables.h"

        namespace spelling {

        const float kMelWindow[kMelTableNFft] = {
        """)
    body += _format_float32_array(window) + "\n};\n\n"
    body += "const int kMelNzOff[kMelTableNMels + 1] = {\n"
    body += _format_plain_int_array(nz_off) + "\n};\n\n"
    body += "const int kMelNzIdx[kMelNzTotal] = {\n"
    body += _format_plain_int_array(nz_idx) + "\n};\n\n"
    body += "const float kMelNzVal[kMelNzTotal] = {\n"
    body += _format_float32_array(nz_val) + "\n};\n\n"
    body += "}  // namespace spelling\n"
    (out_dir / "mel_tables.cc").write_text(body)

    # ~ bytes in flash: window (4*n_fft) + off (4*(n_mels+1)) + idx/val
    flash_bytes = 4 * n_fft + 4 * (n_mels + 1) + 8 * total_nz
    print(
        f"  mel_tables.{{h,cc}}: window={n_fft} floats, "
        f"filterbank CSR={total_nz} nonzeros "
        f"({100.0 * total_nz / (n_mels * n_freq):.1f}% dense), "
        f"~{flash_bytes / 1024:.1f} KB flash"
    )


def _write_classes_files(out_dir: pathlib.Path, classes: list[str]) -> None:
    header = textwrap.dedent(f"""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT.

        #ifndef {_include_guard("classes")}
        #define {_include_guard("classes")}

        namespace spelling {{

        constexpr int kNumClasses = {len(classes)};
        extern const char* const kClassLabels[kNumClasses];

        }}  // namespace spelling

        #endif  // {_include_guard("classes")}
        """)
    (out_dir / "classes.h").write_text(header)

    labels = ", ".join(f'"{c}"' for c in classes)
    body = textwrap.dedent(f"""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT.

        #include "classes.h"

        namespace spelling {{

        const char* const kClassLabels[kNumClasses] = {{ {labels} }};

        }}  // namespace spelling
        """)
    (out_dir / "classes.cc").write_text(body)
    print(f"  classes.{{h,cc}}: {len(classes)} labels")


def _write_clips_files(
    out_dir: pathlib.Path,
    decoded: list[tuple[int, str, str, list[int]]],
    sample_rate: int,
    n_samples: int,
) -> None:
    """Write one ``int16`` array per clip plus a small descriptor table.

    Each clip is emitted as a file-scope static array so the linker can
    place individual blobs anywhere in flash. The descriptor table
    carries label index, source path, and the array pointer + length so
    main.cc can iterate without per-clip switch statements.

    ``decoded`` items are ``(label_idx, label, source_display, samples)``
    where ``samples`` is already int16 PCM (decoded from a local wav or a
    Hub clip) so this formatter is source-agnostic.
    """
    selected = decoded
    header = textwrap.dedent(f"""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT.
        //
        // {len(selected)} test clips embedded as int16 PCM @ {sample_rate} Hz,
        // {n_samples} samples per clip ({n_samples / sample_rate:.3f} s).

        #ifndef {_include_guard("test_clips")}
        #define {_include_guard("test_clips")}
        #include <cstdint>

        namespace spelling {{

        struct EmbeddedClip {{
          int           label_index;  // index into kClassLabels
          const char*   label;        // e.g. "a", "zero"
          const char*   source_path;  // e.g. "speech-data/real/captured/a/foo.wav"
          const int16_t* samples;
          unsigned int   num_samples;
        }};

        constexpr int          kEmbeddedClipSampleRate = {sample_rate};
        constexpr unsigned int kEmbeddedClipNumSamples = {n_samples};
        constexpr int          kNumEmbeddedClips       = {len(selected)};

        extern const EmbeddedClip kEmbeddedClips[kNumEmbeddedClips];

        }}  // namespace spelling

        #endif  // {_include_guard("test_clips")}
        """)
    (out_dir / "test_clips.h").write_text(header)

    parts: list[str] = []
    parts.append(textwrap.dedent("""\
        // AUTO-GENERATED by moonshine-micro/scripts/generate_embedded_data.py
        // DO NOT EDIT.

        #include "test_clips.h"

        namespace spelling {

        """))
    # One static array per clip with deterministic naming so the
    # descriptor table can reference it.
    total_bytes = 0
    for i, (label_idx, label, src_display, samples) in enumerate(selected):
        total_bytes += 2 * len(samples)
        parts.append(
            f"// [{i}] label={label!r}  src={src_display}\n"
            f"static const int16_t kClipSamples_{i}[{len(samples)}] = {{\n"
        )
        parts.append(_format_int16_array(samples) + "\n};\n\n")

    parts.append("const EmbeddedClip kEmbeddedClips[kNumEmbeddedClips] = {\n")
    for i, (label_idx, label, src_display, samples) in enumerate(selected):
        parts.append(
            f'  {{ {label_idx:>3}, "{label}", "{src_display}", '
            f'kClipSamples_{i}, {n_samples} }},\n'
        )
    parts.append("};\n\n}  // namespace spelling\n")

    (out_dir / "test_clips.cc").write_text("".join(parts))
    print(f"  test_clips.{{h,cc}}: {len(selected)} clips, "
          f"{total_bytes:,} bytes of PCM ({total_bytes / 1024:.0f} KB)")


def _pick_clips(
    wavs_roots: list[pathlib.Path],
    classes: list[str],
    clips_per_class: int,
    max_classes: int | None,
) -> list[tuple[int, str, pathlib.Path]]:
    """Pick a deterministic per-class subset (alphabetical by filename)."""
    out: list[tuple[int, str, pathlib.Path]] = []
    cls_iter = classes if max_classes is None else classes[:max_classes]
    for label_idx, label in enumerate(classes):
        if max_classes is not None and label_idx >= max_classes:
            break
        wavs: list[pathlib.Path] = []
        for root in wavs_roots:
            cls_dir = root / label
            if not cls_dir.is_dir():
                continue
            found = sorted(
                w for w in cls_dir.glob("*.wav") if _is_riff_wav(w)
            )[:clips_per_class - len(wavs)]
            wavs.extend(found)
            if len(wavs) >= clips_per_class:
                break
        if not wavs:
            print(f"  WARNING: no RIFF clips for class {label!r} under {wavs_roots}")
            continue
        for w in wavs:
            out.append((label_idx, label, w))
    return out


def _pick_clips_hub(
    repo_id: str,
    configs: list[str],
    classes: list[str],
    clips_per_class: int,
    max_classes: int | None,
    sample_rate: int,
    n_samples: int,
    cache_dir: str | None,
) -> list[tuple[int, str, str, list[int]]]:
    """Pick a deterministic per-class subset from packed HF speech shards.

    Each selected clip's wav bytes are staged to a temp file and decoded with
    the SAME stdlib ``read_wav_mono`` path the local branch uses, so the
    embedded int16 PCM is byte-for-byte identical regardless of source.
    Returns ``(label_idx, label, source_display, int16_samples)`` records.
    """
    import shutil
    import tempfile

    from datasets import Audio, load_dataset

    want = set(classes if max_classes is None else classes[:max_classes])
    by_class: dict[str, list[tuple[str, int, int]]] = {c: [] for c in want}
    ds_by_cfg = []
    for ci, cfg in enumerate(configs):
        ds = load_dataset(repo_id, name=cfg, split="train", cache_dir=cache_dir)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds_by_cfg.append(ds)
        labels = ds["label"]
        rels = ds["rel_path"] if "rel_path" in ds.column_names else None
        for ri, lab in enumerate(labels):
            if lab not in want:
                continue
            rel = (rels[ri] if rels else None) or f"row_{ri}.wav"
            by_class[lab].append((rel, ci, ri))

    decoded: list[tuple[int, str, str, list[int]]] = []
    for label_idx, label in enumerate(classes):
        if max_classes is not None and label_idx >= max_classes:
            break
        cands = sorted(by_class.get(label, []))[:clips_per_class]
        if not cands:
            print(f"  WARNING: no clips for class {label!r} in {repo_id} {configs}")
            continue
        for rel, ci, ri in cands:
            payload = ds_by_cfg[ci][ri]["audio"]
            raw = payload.get("bytes")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                if raw is not None:
                    tf.write(raw)
                else:
                    with open(payload["path"], "rb") as src_f:
                        shutil.copyfileobj(src_f, tf)
                tmp = pathlib.Path(tf.name)
            try:
                samples = _load_clip(tmp, sample_rate, n_samples)
            finally:
                tmp.unlink(missing_ok=True)
            decoded.append((label_idx, label, f"hf://{repo_id}/{rel}", samples))
    return decoded


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--clips-per-class", type=int, default=2,
        help="How many clips to embed per class (deterministic sort).",
    )
    ap.add_argument(
        "--max-classes", type=int, default=None,
        help="Cap class count for quick-iteration builds (default: all 36).",
    )
    ap.add_argument(
        "--tflite", default=None,
        help="Path to int8 mel TFLite (default: newest under moonshine-micro/models/).",
    )
    ap.add_argument(
        "--wavs-dirs",
        default="speech-data/real/captured,speech-data/real/peoples_speech",
        help="Comma-separated clip roots (relative to repo root), searched in order. "
        "Ignored when --hub-dataset/--hub-config is given.",
    )
    ap.add_argument(
        "--hub-dataset",
        default="",
        help="Pull test clips from a packed Hugging Face speech dataset (e.g. "
        "petewarden/moonshine-spelling-speech) instead of local wav trees. "
        "Bytes are decoded with the same stdlib WAV reader for parity.",
    )
    ap.add_argument(
        "--hub-config",
        action="append",
        default=None,
        metavar="CONFIG",
        help="HF config(s) to source clips from (repeatable), e.g. captured, an4. "
        "Defaults to 'captured' when --hub-dataset is set without configs.",
    )
    ap.add_argument(
        "--hub-cache-dir",
        default=None,
        help="Optional datasets cache dir for downloaded HF shards.",
    )
    ap.add_argument(
        "--out-dir", default=str(MOONSHINE_MICRO_ROOT / "examples/rp2350/generated"),
        help="Where to write the generated .h/.cc files (relative to repo root).",
    )
    args = ap.parse_args(argv)

    tflite = _resolve_int8_mel_tflite(
        pathlib.Path(args.tflite) if args.tflite else None
    )
    meta = _load_meta(tflite)
    classes = list(meta["classes"])
    sr = int(meta["sample_rate"])
    clip_s = float(meta["clip_seconds"])
    n_mels = int(meta["n_mels"])
    target_frames = int(meta["target_frames"])
    hop_length = int(meta["hop_length"])
    n_fft = int(meta["n_fft"])
    n_samples = int(round(sr * clip_s))

    out_dir = pathlib.Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = MOONSHINE_MICRO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    use_hub = bool(args.hub_dataset or args.hub_config)
    if use_hub:
        repo = args.hub_dataset or "petewarden/moonshine-spelling-speech"
        configs = list(args.hub_config) if args.hub_config else ["captured"]
        print(f"Source:  HF {repo} configs={configs}")
        selected = _pick_clips_hub(
            repo, configs, classes, args.clips_per_class, args.max_classes,
            sr, n_samples, args.hub_cache_dir,
        )
    else:
        wavs_roots = [
            REPO_ROOT / p.strip()
            for p in args.wavs_dirs.split(",")
            if p.strip()
        ]
        picked = _pick_clips(
            wavs_roots, classes, args.clips_per_class, args.max_classes
        )
        selected = [
            (li, lab, str(p.relative_to(REPO_ROOT)), _load_clip(p, sr, n_samples))
            for li, lab, p in picked
        ]

    print(f"Model:   {tflite.relative_to(REPO_ROOT)}")
    print(f"Classes: {len(classes)}  (sr={sr}, clip_seconds={clip_s})")
    print(f"Mel:     n_mels={n_mels}, target_frames={target_frames}, "
          f"hop_length={hop_length}, n_fft={n_fft}")
    print(f"Clips:   {len(selected)} ({args.clips_per_class} per class * "
          f"{len(classes) if args.max_classes is None else args.max_classes} classes)")
    print(f"Output:  {out_dir}/")
    _write_audio_config_file(
        out_dir,
        sample_rate=sr,
        clip_seconds=clip_s,
        n_mels=n_mels,
        target_frames=target_frames,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    # f_min / f_max must match audio_config.h (kFMin=20.0, kFMax=Nyquist)
    # and the LogMelParams main.cc builds from it -- otherwise the baked
    # filterbank triangles wouldn't line up with the runtime config.
    _write_mel_tables_file(
        out_dir,
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=n_fft,
        f_min=20.0,
        f_max=sr / 2.0,
    )
    _write_model_files(out_dir, tflite)
    _write_classes_files(out_dir, classes)
    _write_clips_files(out_dir, selected, sr, n_samples)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

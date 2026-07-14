"""Download optional augmentation assets: MUSAN noise + OpenSLR-26 RIRs.

Training works without these (it falls back to synthetic colored noise), but
real background noise and room impulse responses make the model noticeably more
robust in a real room. This is a convenience wrapper; skip it if you have your
own noise/RIR corpora.

  MUSAN noise  (~1 GB full, https://www.openslr.org/17/)  -> data/musan/noise/
  OpenSLR-26   (~179 MB 16 kHz RIRs, https://www.openslr.org/26/) -> data/rirs/

Only the noise/ subset of MUSAN is streamed and kept. Use --small / --max-files
to keep lighter subsets.

Example::

    python tools/download_musan_rirs.py                 # ~200 noise files + 2k RIRs
    python tools/download_musan_rirs.py --small 0        # full noise subset
"""

from __future__ import annotations

import argparse
import random
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIR_URLS = [
    "https://www.openslr.org/resources/26/sim_rir_16k.zip",
    "https://openslr.trmal.net/resources/26/sim_rir_16k.zip",
]


def download_musan_noise(out_dir: Path, small: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Streaming MUSAN noise from {MUSAN_URL} ...")
    kept = []
    with urllib.request.urlopen(MUSAN_URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = member.name
                if not (member.isfile() and name.endswith(".wav") and "/noise/" in name):
                    continue
                parts = name.split("/")
                source = parts[-2] if len(parts) >= 2 else "noise"
                dst = out_dir / f"{source}__{parts[-1]}"
                if dst.exists():
                    kept.append(dst)
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                dst.write_bytes(f.read())
                kept.append(dst)
    print(f"  kept {len(kept)} noise files in {out_dir}")
    if small and len(kept) > small:
        rng = random.Random(seed)
        rng.shuffle(kept)
        for extra in kept[small:]:
            extra.unlink(missing_ok=True)
        print(f"  subsampled to {small} files")


def download_rirs(out_dir: Path, max_files: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for url in RIR_URLS:
        try:
            print(f"Downloading RIRs from {url} ...")
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
                with urllib.request.urlopen(url) as resp:
                    tmp.write(resp.read())
                tmp.flush()
                with zipfile.ZipFile(tmp.name) as zf:
                    wavs = [n for n in zf.namelist() if n.endswith(".wav")]
                    if max_files and len(wavs) > max_files:
                        rng = random.Random(seed)
                        rng.shuffle(wavs)
                        wavs = wavs[:max_files]
                    for n in wavs:
                        dst = out_dir / Path(n).name
                        if dst.exists():
                            continue
                        dst.write_bytes(zf.read(n))
            print(f"  extracted {len(wavs)} RIRs into {out_dir}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  failed ({exc}); trying next mirror...")
    print("  WARNING: could not download RIRs from any mirror.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--musan-dir", default="data/musan/noise")
    ap.add_argument("--rir-dir", default="data/rirs")
    ap.add_argument("--small", type=int, default=200, help="MUSAN noise file cap (0 = all).")
    ap.add_argument("--max-files", type=int, default=2000, help="RIR file cap (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-musan", action="store_true")
    ap.add_argument("--skip-rirs", action="store_true")
    args = ap.parse_args()

    if not args.skip_musan:
        download_musan_noise(Path(args.musan_dir), args.small, args.seed)
    if not args.skip_rirs:
        download_rirs(Path(args.rir_dir), args.max_files, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

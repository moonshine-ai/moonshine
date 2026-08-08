import base64
import hashlib
import os
import struct
from pathlib import Path
from typing import Callable, Optional

import requests
from tqdm import tqdm
from filelock import FileLock
from platformdirs import user_cache_dir
import platform

try:  # Fast C implementation (small wheel, standard for GCS checksums).
    import google_crc32c  # type: ignore

    _HAVE_CRC32C = True
except Exception:  # pragma: no cover - optional acceleration only
    google_crc32c = None  # type: ignore
    _HAVE_CRC32C = False


def get_cache_dir(app_name: str = "moonshine_voice") -> Path:
    """Get the cache directory, respecting environment override."""
    env_var = f"{app_name.upper()}_CACHE"
    return Path(os.environ.get(env_var, user_cache_dir(app_name)))


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def crc32c_file(path: Path) -> Optional[str]:
    """Return the base64 CRC32C of ``path`` (matching Google Cloud Storage's
    ``x-goog-hash: crc32c=...``), or ``None`` when the ``google-crc32c``
    acceleration package is unavailable (verification is then skipped)."""
    if not _HAVE_CRC32C:
        return None
    checksum = google_crc32c.Checksum()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(struct.pack(">I", int.from_bytes(checksum.digest(), "big"))).decode(
        "ascii"
    )


def _file_matches(
    path: Path,
    expected_sha256: Optional[str],
    expected_size: Optional[int],
    expected_crc32c: Optional[str],
) -> bool:
    """True when ``path`` already satisfies every supplied integrity check."""
    if expected_size is not None and expected_size >= 0:
        if path.stat().st_size != expected_size:
            return False
    if expected_sha256 is not None:
        if hash_file(path) != expected_sha256:
            return False
    if expected_crc32c:
        actual = crc32c_file(path)
        # Only fails when we could compute a digest and it disagreed; if the
        # accelerator is missing (actual is None) we don't force a re-download.
        if actual is not None and actual != expected_crc32c:
            return False
    return True


def download_file(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    resume: bool = True,
    show_progress: bool = True,
    timeout: int = 30,
    expected_size: Optional[int] = None,
    expected_crc32c: Optional[str] = None,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> Path:
    """
    Download a file with progress bar, resume support, and integrity checking.

    Args:
        url: URL to download from
        dest: Destination path for the file
        expected_sha256: Optional SHA256 hash to verify after download
        resume: Whether to attempt resuming partial downloads
        show_progress: Whether to show a progress bar
        timeout: Connection timeout in seconds
        expected_size: Optional expected size in bytes (from the model catalog)
        expected_crc32c: Optional expected base64 CRC32C digest (from the model
            catalog, matching Google Cloud Storage). Verified only when the
            ``google-crc32c`` package is installed.
        on_bytes: Optional sink called with the number of bytes just added to
            this file, including any resumed prefix. Callers that need an
            overall percentage across several files accumulate these deltas
            themselves; see ``download._ProgressTracker``. Not called at all
            when the file is already cached, since no bytes move.

    Returns:
        Path to the downloaded file

    Raises:
        requests.HTTPError: If download fails
        ValueError: If any integrity check fails
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_file = dest.with_suffix(dest.suffix + ".partial")
    lock_file = dest.with_suffix(dest.suffix + ".lock")

    with FileLock(lock_file):
        # Check if already downloaded and valid. For cached files we only verify
        # the (cheap) size and any SHA256, deliberately skipping CRC32C so we
        # don't rehash large multi-hundred-MB models on every load; CRC32C is
        # fully verified on the fresh download below.
        if dest.exists():
            if _file_matches(dest, expected_sha256, expected_size, None):
                return dest
            # Integrity mismatch (or stale file), re-download.
            dest.unlink()

        # Check for partial download
        initial_size = 0
        headers = {}
        if resume and temp_file.exists():
            initial_size = temp_file.stat().st_size
            headers["Range"] = f"bytes={initial_size}-"

        # Start download
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)

        # Handle resume response
        if response.status_code == 416:  # Range not satisfiable
            # File might be complete or server doesn't support range
            temp_file.unlink(missing_ok=True)
            initial_size = 0
            response = requests.get(url, stream=True, timeout=timeout)

        response.raise_for_status()

        # Get total size
        if response.status_code == 206:  # Partial content
            # Content-Range: bytes 1000-9999/10000
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = initial_size + int(
                    response.headers.get("Content-Length", 0)
                )
        else:
            # Full download (server ignored range request or fresh download)
            total_size = int(response.headers.get("Content-Length", 0))
            initial_size = 0  # Reset if server didn't honor range
            temp_file.unlink(missing_ok=True)

        # Download with progress bar
        mode = "ab" if initial_size > 0 else "wb"

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=total_size,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            )
        if on_bytes and initial_size > 0:
            on_bytes(initial_size)

        try:
            with open(temp_file, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        if progress_bar:
                            progress_bar.update(len(chunk))
                        if on_bytes:
                            on_bytes(len(chunk))
        finally:
            if progress_bar:
                progress_bar.close()

        # Verify integrity
        if expected_sha256:
            actual_hash = hash_file(temp_file)
            if actual_hash != expected_sha256:
                temp_file.unlink()
                raise ValueError(
                    f"SHA256 mismatch for {dest.name}: "
                    f"expected {expected_sha256}, got {actual_hash}"
                )
        if expected_size is not None and expected_size >= 0:
            actual_size = temp_file.stat().st_size
            if actual_size != expected_size:
                temp_file.unlink()
                raise ValueError(
                    f"Size mismatch for {dest.name}: "
                    f"expected {expected_size} bytes, got {actual_size}"
                )
        if expected_crc32c:
            actual_crc = crc32c_file(temp_file)
            if actual_crc is not None and actual_crc != expected_crc32c:
                temp_file.unlink()
                raise ValueError(
                    f"CRC32C mismatch for {dest.name}: "
                    f"expected {expected_crc32c}, got {actual_crc}"
                )

        # Atomic rename
        temp_file.rename(dest)

        if platform.system() != "Windows":
            # Clean up lock file
            lock_file.unlink(missing_ok=True)
    return dest


def download_model(
    url: str,
    filename: str,
    expected_sha256: Optional[str] = None,
    app_name: str = "moonshine_voice",
    expected_size: Optional[int] = None,
    expected_crc32c: Optional[str] = None,
    **kwargs,
) -> Path:
    """
    Download a model file to the cache directory.

    Args:
        url: URL to download from
        filename: Name for the cached file
        expected_sha256: Optional SHA256 hash to verify
        app_name: Application name for cache directory
        expected_size: Optional expected size in bytes (from the model catalog)
        expected_crc32c: Optional expected base64 CRC32C digest (from the catalog)
        **kwargs: Additional arguments passed to download_file

    Returns:
        Path to the cached model file
    """
    cache_dir = get_cache_dir(app_name)
    dest = cache_dir / filename
    return download_file(
        url,
        dest,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        expected_crc32c=expected_crc32c,
        **kwargs,
    )


# Example usage
if __name__ == "__main__":
    # Example: download a test file
    model_path = download_model(
        url="https://huggingface.co/openai/whisper-tiny/resolve/main/config.json",
        filename="foo/bar/whisper-tiny-config.json",
        app_name="moonshine_voice",
    )
    print(f"Downloaded to: {model_path}")

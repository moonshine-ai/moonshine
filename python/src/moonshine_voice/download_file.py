import hashlib
import os
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from filelock import FileLock
from platformdirs import user_cache_dir
import platform


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


def download_file(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    resume: bool = True,
    show_progress: bool = True,
    timeout: int = 30,
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

    Returns:
        Path to the downloaded file

    Raises:
        requests.HTTPError: If download fails
        ValueError: If SHA256 verification fails
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_file = dest.with_suffix(dest.suffix + ".partial")
    lock_file = dest.with_suffix(dest.suffix + ".lock")

    with FileLock(lock_file):
        # Check if already downloaded and valid
        if dest.exists():
            if expected_sha256 is None:
                return dest
            if hash_file(dest) == expected_sha256:
                return dest
            # Hash mismatch, re-download
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

        try:
            with open(temp_file, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        if progress_bar:
                            progress_bar.update(len(chunk))
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
    **kwargs,
) -> Path:
    """
    Download a model file to the cache directory.

    Args:
        url: URL to download from
        filename: Name for the cached file
        expected_sha256: Optional SHA256 hash to verify
        app_name: Application name for cache directory
        **kwargs: Additional arguments passed to download_file

    Returns:
        Path to the cached model file
    """
    cache_dir = get_cache_dir(app_name)
    dest = cache_dir / filename
    return download_file(url, dest, expected_sha256=expected_sha256, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example: download a test file
    model_path = download_model(
        url="https://huggingface.co/openai/whisper-tiny/resolve/main/config.json",
        filename="foo/bar/whisper-tiny-config.json",
        app_name="moonshine_voice",
    )
    print(f"Downloaded to: {model_path}")

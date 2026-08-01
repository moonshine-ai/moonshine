"""Tests for the download progress callbacks.

The public promise is that ``on_progress`` gets a fraction that starts at 0,
never goes backwards, and ends at 1, plus the name of the file being fetched.
These check that promise holds for both the byte-weighted path (STT manifests
declare a size per file) and the file-counted fallback (TTS and G2P dependency
lists are bare keys), and that the plumbing underneath actually reports bytes.
"""

import moonshine_voice.download as download
import moonshine_voice.download_file as download_file_mod
from moonshine_voice.download import _ProgressTracker, _download_manifest_group


def _record():
    """Returns a (callback, list-of-calls) pair."""
    calls = []
    return (lambda fraction, name: calls.append((fraction, name))), calls


def _group(*files):
    return {"base_url": "https://example.test/model", "files": list(files)}


def _fractions(calls):
    return [fraction for fraction, _ in calls]


def _assert_well_formed(calls):
    """Every run has to start at 0, rise monotonically, and finish at 1."""
    fractions = _fractions(calls)
    assert fractions, "no progress was reported at all"
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions), f"progress went backwards: {fractions}"
    assert all(0.0 <= f <= 1.0 for f in fractions)


# ---------------------------------------------------------------------------
# _ProgressTracker
# ---------------------------------------------------------------------------


def test_tracker_weights_by_bytes_when_sizes_are_known():
    callback, calls = _record()
    tracker = _ProgressTracker.for_groups(
        callback, [_group({"name": "a", "size": 100}, {"name": "b", "size": 300})]
    )

    tracker.start("a", 100)
    tracker.add_bytes(100)
    tracker.finish()
    tracker.start("b", 300)
    tracker.add_bytes(300)
    tracker.finish()

    _assert_well_formed(calls)
    # The small file is a quarter of the bytes, so finishing it is 25% and not
    # the 50% a file count would have claimed.
    assert (0.25, "a") in calls


def test_tracker_counts_files_when_no_sizes_are_declared():
    callback, calls = _record()
    tracker = _ProgressTracker(callback, total_files=4)

    for name in ("a", "b", "c", "d"):
        tracker.start(name)
        tracker.add_bytes(9999)
        tracker.finish()

    _assert_well_formed(calls)
    assert (0.5, "b") in calls


def test_tracker_counts_files_when_any_single_size_is_missing():
    """One unsized file makes the byte total a lie, so the whole run falls back."""
    callback, calls = _record()
    tracker = _ProgressTracker.for_groups(
        callback, [_group({"name": "a", "size": 100}, {"name": "b"})]
    )

    tracker.start("a", 100)
    tracker.add_bytes(100)
    tracker.finish()
    tracker.start("b", None)
    tracker.finish()

    _assert_well_formed(calls)
    assert (0.5, "a") in calls


def test_tracker_spans_several_groups():
    """get_model_for_language covers the transcriber and the spelling model with
    one tracker, so the bar fills once rather than twice."""
    callback, calls = _record()
    tracker = _ProgressTracker.for_groups(
        callback,
        [_group({"name": "a", "size": 750}), _group({"name": "b", "size": 250})],
    )

    tracker.start("a", 750)
    tracker.add_bytes(750)
    tracker.finish()
    assert calls[-1] == (0.75, "a")

    tracker.start("b", 250)
    tracker.add_bytes(250)
    tracker.finish()
    _assert_well_formed(calls)


def test_tracker_clamps_a_file_that_overruns_its_declared_size():
    callback, calls = _record()
    tracker = _ProgressTracker.for_groups(callback, [_group({"name": "a", "size": 10})])

    tracker.start("a", 10)
    tracker.add_bytes(1000)
    tracker.finish()

    _assert_well_formed(calls)


def test_tracker_throttles_chatter():
    """8KB chunks would otherwise fire tens of thousands of callbacks."""
    callback, calls = _record()
    tracker = _ProgressTracker.for_groups(
        callback, [_group({"name": "a", "size": 1_000_000})]
    )

    tracker.start("a", 1_000_000)
    for _ in range(1000):
        tracker.add_bytes(1000)
    tracker.finish()

    _assert_well_formed(calls)
    assert len(calls) < 600, f"{len(calls)} callbacks for 1000 chunks is too chatty"


# ---------------------------------------------------------------------------
# _download_manifest_group
# ---------------------------------------------------------------------------


def _capture_download_model(monkeypatch, sizes):
    """Replaces download_model with a stub that feeds ``sizes`` bytes back
    through on_bytes, and records the kwargs it was called with."""
    seen = []

    def fake_download_model(url, dest, **kwargs):
        seen.append(kwargs)
        on_bytes = kwargs.get("on_bytes")
        if on_bytes:
            on_bytes(sizes.pop(0))
        return dest

    monkeypatch.setattr(download, "download_model", fake_download_model)
    return seen


def test_manifest_group_reports_every_file(monkeypatch, tmp_path):
    _capture_download_model(monkeypatch, [100, 300])
    callback, calls = _record()
    group = _group({"name": "a", "size": 100}, {"name": "b", "size": 300})
    tracker = _ProgressTracker.for_groups(callback, [group])

    _download_manifest_group(group, tmp_path, tracker)

    _assert_well_formed(calls)
    assert {name for _, name in calls} == {"a", "b"}


def test_manifest_group_silences_tqdm_while_reporting(monkeypatch, tmp_path):
    """An app drawing its own bar does not want tqdm underneath it."""
    seen = _capture_download_model(monkeypatch, [100])
    callback, _ = _record()
    group = _group({"name": "a", "size": 100})

    _download_manifest_group(group, tmp_path, _ProgressTracker.for_groups(callback, [group]))

    assert seen[0]["show_progress"] is False


def test_manifest_group_keeps_tqdm_when_nobody_is_listening(monkeypatch, tmp_path):
    seen = _capture_download_model(monkeypatch, [100])
    group = _group({"name": "a", "size": 100})

    _download_manifest_group(group, tmp_path, None)

    assert seen[0]["show_progress"] is True
    assert seen[0]["on_bytes"] is None


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers if headers is not None else {
            "Content-Length": str(len(body))
        }

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


class _FakeRequests:
    """Stands in for the requests module inside download_file."""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self._response


def test_download_file_reports_byte_deltas(monkeypatch, tmp_path):
    body = b"x" * 20_000
    monkeypatch.setattr(download_file_mod, "requests", _FakeRequests(_FakeResponse(body)))
    deltas = []

    download_file_mod.download_file(
        "https://example.test/a.bin",
        tmp_path / "a.bin",
        show_progress=False,
        on_bytes=deltas.append,
    )

    assert sum(deltas) == len(body)
    assert len(deltas) > 1, "expected several chunks, not one lump"


def test_download_file_counts_a_resumed_prefix(monkeypatch, tmp_path):
    """Bytes already on disk still count towards the total, or the bar would
    start part-filled and then appear to stall."""
    dest = tmp_path / "a.bin"
    partial = tmp_path / "a.bin.partial"
    partial.write_bytes(b"x" * 5_000)
    rest = b"y" * 5_000
    response = _FakeResponse(
        rest,
        status_code=206,
        headers={"Content-Range": "bytes 5000-9999/10000"},
    )
    monkeypatch.setattr(download_file_mod, "requests", _FakeRequests(response))
    deltas = []

    download_file_mod.download_file(
        "https://example.test/a.bin", dest, show_progress=False, on_bytes=deltas.append
    )

    assert sum(deltas) == 10_000
    assert deltas[0] == 5_000


def test_download_file_stays_quiet_for_a_cached_file(monkeypatch, tmp_path):
    """A file already on disk moves no bytes, so the tracker credits it at the
    declared size instead and nothing here fires."""
    dest = tmp_path / "a.bin"
    dest.write_bytes(b"x" * 100)
    requests = _FakeRequests(_FakeResponse(b""))
    monkeypatch.setattr(download_file_mod, "requests", requests)
    deltas = []

    download_file_mod.download_file(
        "https://example.test/a.bin",
        dest,
        expected_size=100,
        show_progress=False,
        on_bytes=deltas.append,
    )

    assert deltas == []
    assert requests.calls == 0


def test_cached_files_still_reach_full_progress(monkeypatch, tmp_path):
    """The everything-already-downloaded case still has to end at 1."""
    monkeypatch.setattr(
        download, "download_model", lambda url, dest, **kwargs: dest
    )
    callback, calls = _record()
    group = _group({"name": "a", "size": 100}, {"name": "b", "size": 300})

    _download_manifest_group(group, tmp_path, _ProgressTracker.for_groups(callback, [group]))

    _assert_well_formed(calls)

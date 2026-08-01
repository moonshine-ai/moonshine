These are built from source by `scripts/build-ort-ios.sh`, not downloaded. They
replaced the prebuilt pod archive
`https://download.onnxruntime.ai/pod-archive-onnxruntime-mobile-c-1.23.2.zip`,
which is still the thing to compare against if you want to know what the
operator-restricted build buys.

Do not drop that pod archive back in here. Everything above the library assumes
a minimal build: only ORT-format models load (`docs/ort-only-models.md`), and
there is no CoreML provider (`docs/execution-providers.md`).

An iOS static library is a poor guide to what an app installs, because the app
linker drops the object files nothing references. Both numbers, for the arm64
device slice:

| | Pod archive | Minimal | Saved |
| --- | ---: | ---: | ---: |
| `libonnxruntime.a` | 36.6 MB | 18.1 MB | 18.5 MB |
| `libmoonshine.a` (merged) | 60.3 MB | 41.8 MB | 18.5 MB |
| Linked app binary | 45.0 MB | 30.6 MB | 14.4 MB |
| `__TEXT` of that binary | 31.1 MB | 22.8 MB | 8.3 MB |

The linked-binary row is the one that matters, and it is what
`scripts/measure-mobile-size.sh ios` reports: it links a small app against the
xcframework with `-dead_strip` and measures the result.

Both Moonshine columns were measured while the diarization models were still
compiled in, so they read 8.2 MB high against a library built today
(`docs/diarization-models.md`); the current minimal linked binary is 22.4 MB.
The saving each row reports is unaffected, since it is the difference between
two builds that both carried those models.

The simulator slice is fat (`x86_64 arm64`), which
`../find-ort-library-path.cmake` expects; it is larger than the device slice
for that reason alone.

# The diarization models are a download

Speaker identification (the `identify_speakers` transcriber option) runs two
models: a segmentation model and a speaker-embedding model, both from
[pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
They used to be compiled into the library as C arrays. They are now fetched from
the CDN like every other model. This is a breaking change for anyone already
using `identify_speakers`.

## Why

The two models are 8.2 MB together, and every application carrying Moonshine
paid for them whether or not it ever identified a speaker. That was most
applications: diarization is off by default and costs a lot of compute, so it
tends to be a deliberate choice rather than the common case.

Making them a download takes 8.2 MB off every platform, as measured by
`scripts/measure-mobile-size.sh`:

| | Before | After |
| --- | ---: | ---: |
| Android arm64-v8a install | 24.6 MB | 16.4 MB |
| Android armeabi-v7a install | 20.7 MB | 12.5 MB |
| Android x86_64 install | 25.2 MB | 17.1 MB |
| iOS linked app binary (arm64) | 30.6 MB | 22.4 MB |

An application that *does* diarize pays the same bytes it always did, just over
the network on first use and into a cache, rather than in the binary.

The clustering parameters are a different matter and are still compiled in. The
PLDA and x-vector arrays are only 261 KB, they are useless without being paired
with exactly these two models, and downloading them separately would buy
nothing.

## What changed

`identify_speakers=true` now needs the models, and a transcriber built without
them fails to load rather than falling back to anything. There are three ways to
supply them, and most callers will not have to do anything:

- **The high-level loaders do it for you.** Python's `MicTranscriber`, Swift's
  `Transcriber.load` and `MicTranscriber.load`, and the WebAssembly
  `Transcriber.load` all notice `identify_speakers` and fetch the models before
  constructing anything.
- **The `diarization_model_dir` option**, for the file-based loaders. Point it
  at a directory holding `segmentation.ort` and `embedding.ort`.
- **In memory**, by passing `segmentation.ort` and `embedding.ort` as keys to
  `moonshine_load_transcriber_from_memory_files`. This is how the browser does
  it, since it has no filesystem.

## Getting the models

Every binding resolves the file list from the library rather than hardcoding it,
so the manifest is the source of truth:

| | |
| --- | --- |
| C | `moonshine_get_diarization_dependencies()` |
| Python | `moonshine_voice.get_diarization_model()`, or `python -m moonshine_voice.download --diarization` |
| Swift | `ModelSpec.diarization`, via `AssetDownloader` or `Moonshine.prepareModels` |
| Android | `ModelSpec.diarization()`, via `Models.ensureOne` or `MoonshineDownloadWorker` |
| WebAssembly | `module.diarizationDependencies()`, fed to `AssetDownloader` |

They live at
`https://download.moonshine.ai/model/diarization-community1/`, and land in the
same per-platform cache as everything else, so the download happens once.

To warm the cache before going offline:

```bash
python -m moonshine_voice.download --diarization
```

## Migrating

If you construct a transcriber directly with `identify_speakers` and a model
directory, download the models and pass the directory:

```python
from moonshine_voice import Transcriber, get_diarization_model

transcriber = Transcriber(
    model_path,
    model_arch=model_arch,
    options={
        "identify_speakers": "true",
        "diarization_model_dir": get_diarization_model(),
    },
)
```

If you use `MicTranscriber`, or Swift's or WebAssembly's `Transcriber.load`,
nothing changes: setting `identify_speakers` is enough, and the first load
downloads what it needs.

A transcriber asked for speaker IDs with no models to hand fails with a message
naming the missing file and pointing back here, rather than a null-path error
out of ONNX Runtime.

## What did not change

- The models themselves. The published `.ort` files are byte-for-byte the arrays
  that used to be compiled in, so results are identical.
- Everything about how diarization behaves once loaded: the same options, the
  same speaker spans, the same clustering.
- Offline use after the first fetch. The cache is on disk (or in the browser's
  Cache API), so only the first run needs the network.

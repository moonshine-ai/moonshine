These are built from source by `scripts/build-ort-android.sh`, not downloaded.
Run that script to replace them; it pins the ONNX Runtime version, restricts
the build to `../../moonshine-required-operators.config`, and strips before
vendoring.

Do not drop a stock ONNX Runtime AAR in here. Everything above the library
assumes a minimal build: only ORT-format models load
(`docs/ort-only-models.md`), and there is no NNAPI provider
(`docs/execution-providers.md`).

What the operator-restricted build is worth, per ABI, against the stock 1.23.2
mobile libraries it replaced:

| ABI | Stock mobile | Minimal | Saved |
| --- | ---: | ---: | ---: |
| arm64-v8a | 18.5 MB | 6.0 MB | 12.5 MB |
| armeabi-v7a | 13.3 MB | 3.7 MB | 9.6 MB |
| x86_64 | 22.1 MB | 6.4 MB | 15.7 MB |

A device installs one ABI, so an arm64 phone sees 12.5 MB of that. Measure the
result with `scripts/measure-mobile-size.sh android`, which reads the built AAR
rather than these files, since Gradle repackages them.

An unstripped arm64 library is around 675 MB, of which 636 MB is `.debug*`
sections Gradle discards at packaging. The build script strips with the NDK's
own `llvm-strip`; check any replacement before committing it, because Git LFS
will happily carry the debug info forever:

    file <abi>/libonnxruntime.so   # expect "stripped"
    strings -a <abi>/libonnxruntime.so | grep -o 'VERS_1\.[0-9.]*' | sort -u

Keep every ABI on the same ORT version, and on the same version as the other
platforms; `scripts/ort-build-common.sh` holds the one that all three build
scripts share.

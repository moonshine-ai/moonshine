# StreamingLatency (iOS)

Host app used by `scripts/test-mobile-latency.sh` to measure English Tiny /
Small / Medium Streaming end-of-phrase latency on a physical iPad/iPhone.

Models are downloaded from `https://download.moonshine.ai` at test time (the
device needs a network connection). `two_cities.wav` is fetched from GitHub if
not already cached.

## Run

```bash
./scripts/test-mobile-latency.sh --ios-only --skip-build-swift
```

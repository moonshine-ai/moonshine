# kaldi-native-fbank

Native C++ implementation of Kaldi-compatible fbank (filter bank) feature
extraction, with no dependency on Kaldi itself.

- **Upstream:** <https://github.com/csukuangfj/kaldi-native-fbank>
- **Version:** v1.22.3
- **License:** Apache-2.0 (see `LICENSE`)

All library source and header files from `kaldi-native-fbank/csrc/` are
included here (test files excluded). This library depends on kissfft at
link time (see `../kissfft/`).

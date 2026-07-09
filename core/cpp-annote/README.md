# cpp-annote (vendored)

Vendored copy of https://github.com/moonshine-ai/cpp-annote, a C++/ONNX Runtime
port of the pyannote community-1 speaker diarization pipeline. Used by the
transcriber for opt-in speaker identification (`identify_speakers` option).

Streaming sessions bound VBx to a sliding window (default 120s via
`diarization_cluster_window_sec`); batch/one-shot diarization uses full history.

Local modifications relative to upstream:

- Removed the cnpy/zlib dependency: the file-based `PldaModel::load()` NPZ
  loader in `src/plda_vbx.cpp` and the `PYANNOTE_CPP_PARITY=2` heavy NPZ dump
  block in `src/clustering_vbx.cpp` are removed. Only the compiled-in
  community-1 model data path is used.
- Only the library sources are vendored (no tests, tools, golden data, or
  cnpy).
- Upstream's bundled dependencies (Eigen, kaldi-native-fbank, kissfft) are
  moved out of this folder into `core/third-party/`, alongside the other
  vendored libraries.

`src/community1_ort_embedded.cpp` (segmentation + embedding ORT models) and
`src/community1_cpp_annote_embedded.cpp` (PLDA/config data) are generated
files tracked with Git LFS.

Licenses: cpp-annote is MIT (see LICENSE). The community-1 diarization models
are released by pyannote.ai under the Creative Commons Attribution 4.0
License. Each dependency under `core/third-party/` keeps its own license
in its folder: Eigen (`COPYING.MPL2`, MPL2, `EIGEN_MPL2_ONLY`), kaldi-native-fbank
(`LICENSE`, Apache 2.0), kissfft (`COPYING`, BSD 3-clause).

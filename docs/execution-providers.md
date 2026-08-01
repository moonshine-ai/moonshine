# Execution providers: why Moonshine ships CPU-only

Moonshine runs its models on ONNX Runtime's CPU execution provider on every
platform. The `ort_providers` option still accepts `CoreML` and `NNAPI`, but no
library we ship contains either, and asking for one returns an error pointing
here rather than silently doing something slow.

This is a measured decision, not an omission, and the measurement is worth
repeating whenever ONNX Runtime or our models change.

## What a compiling provider needs

CoreML and NNAPI do not execute a graph node by node. Each one asks ONNX
Runtime which nodes it recognises, takes those in contiguous groups, compiles
every group into a graph of its own, and leaves the rest to the CPU provider.
Each boundary between a compiled group and the CPU costs a synchronisation, and
on the Neural Engine a copy across memory the CPU cannot see.

So the number that decides whether one of these providers helps is not how many
nodes it supports. It is how few pieces the graph lands in. A model taken whole
is a win; a model taken in fifty scraps is slower than never having asked.

## What we measured

We convert our models to `.ort` at full graph optimization, which is what makes
them load fast and run fast on the CPU (see
[`scripts/convert-models-to-ort.py`](../scripts/convert-models-to-ort.py)).
That optimization fuses whole regions into ONNX Runtime's own `com.microsoft`
operators — `FusedConv`, `MultiHeadAttention`, `MatMulNBits`,
`SkipLayerNormalization` and the rest. They are CPU kernels. No compiling
provider recognises any of them, and because they sit in the middle of the
graph rather than at its edges, they cut everything else into fragments.

Asking CoreML what it would take, on the models as we ship them:

| Model | Nodes | Taken | Partitions | Nodes per partition |
| --- | ---: | ---: | ---: | ---: |
| `tiny-en` encoder | 582 | 156 | 59 | 2.6 |
| `tiny-en` decoder (merged) | 2 | 0 | 0 | — |
| `spelling_cnn` | 174 | 123 | 18 | 6.8 |
| Kokoro | 2,994 | 891 | 148 | 6.0 |
| Piper `en_US-saikat` | 2,549 | 891 | 141 | 6.3 |
| Piper `en_US-amy-low` | 2,528 | 958 | 220 | 4.4 |
| ZipVoice `fm_decoder` | 3,996 | 84 | 63 | 1.3 |
| ZipVoice `vocoder` | 120 | 63 | 14 | 4.5 |
| English OOV G2P | 1,076 | 368 | 116 | 3.2 |

Nothing reaches seven nodes per partition. The Piper voices, the largest and
most latency-sensitive models we run, are cut into 141 and 220 pieces.

Reproduce it with:

```bash
python scripts/check-ep-partitioning.py core/moonshine-tts/data/kokoro/model.ort
```

CoreML stands in for NNAPI in that script because both accept `ai.onnx`
operators only, so a graph fragmented for one is fragmented for the other, and
CoreML runs on a development Mac.

## What we do about it

**Android** builds without NNAPI. It costs 0.55 MB of a 6.5 MB library, 9% of
everything the minimal build worked to remove, and buys the partitioning above.
[`scripts/build-ort-android.sh`](../scripts/build-ort-android.sh) takes
`with-nnapi` to put it back.

**iOS** builds without CoreML, and here we have no choice: ONNX Runtime 1.23's
CoreML provider does not compile in a minimal build at all. It calls
`Graph::GetModel`, which
`include/onnxruntime/core/graph/graph.h` compiles out under `ORT_MINIMAL_BUILD`,
and `--minimal_build extended` still defines that. The build fails with "no
member named 'GetModel'". `scripts/build-ort-ios.sh with-coreml` attempts it
anyway, which is how to check whether a later ONNX Runtime has fixed it.

The `ort_providers` and `coreml_cache_dir` options keep working and keep their
meaning. `cpu` is the only value any shipped build accepts.

## What would change the answer

Two things, either of which is worth rerunning the measurement for:

- **Runtime-style conversion.** ONNX Runtime can save an `.ort` with the
  level-2 and level-3 fusions held back as *runtime optimizations*, applied at
  load time only to the nodes a compiling provider did not take
  (`--optimization_style Runtime`). A graph converted that way keeps its
  `ai.onnx` operators where the provider can see them. The cost is a second set
  of model artifacts for every model in the catalog, downloaded by every
  client, and a CPU path that must be shown not to have regressed. That was too
  much to spend on a provider we had not yet shown would win.
- **A provider that handles the fused operators.** If CoreML or NNAPI grows
  support for the `com.microsoft` domain, or ONNX Runtime stops fusing into it
  by default, the partition counts collapse and the trade changes.

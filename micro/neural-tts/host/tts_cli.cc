// Native (desktop) driver for the on-device neural TTS engine.
//
// Runs the EXACT C++ pipeline the RP2350 firmware runs -- shared g2p front end
// -> Klatt rule durations -> diphone/word unit selection -> int8 RVQ decode
// through the s16x8 TFLM graph -> WORLD-lite vocoder -- but on the host, using
// the portable TFLM reference kernels (see host/tflm_ref/) instead of CMSIS-NN.
// The flash pack is loaded from a .bin file rather than linked from flash.
//
// This is the fast, flash-free equivalent of the `moonshine_micro_tts` firmware
// + tts_speak.py: same bytes in, same PCM out, no board required.
//
// Usage:
//   tts_cli "hello there" -o out.wav
//   tts_cli --ipa "h@loU" -o hi.wav
//   tts_cli --pack path/to/neural_tts_pack.bin "text" -o out.wav
//   tts_cli "text" -o -            # raw int16 LE PCM to stdout

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "neural_tts/neural_tts.h"

namespace {

// Default pack: the same blob the firmware embeds (generated/neural_tts_pack.bin).
constexpr const char* kDefaultPack =
    "examples/rp2350/generated/neural_tts_pack.bin";

std::vector<uint8_t> ReadFile(const char* path) {
  std::vector<uint8_t> buf;
  FILE* f = std::fopen(path, "rb");
  if (!f) return buf;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    buf.resize(static_cast<size_t>(n));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) buf.clear();
  }
  std::fclose(f);
  return buf;
}

void WriteWavHeader(FILE* f, int rate, int nsamples) {
  const uint32_t data_bytes = static_cast<uint32_t>(nsamples) * 2u;
  const uint32_t byte_rate = static_cast<uint32_t>(rate) * 2u;
  auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
  auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
  std::fwrite("RIFF", 1, 4, f);
  w32(36u + data_bytes);
  std::fwrite("WAVE", 1, 4, f);
  std::fwrite("fmt ", 1, 4, f);
  w32(16u);          // fmt chunk size
  w16(1);            // PCM
  w16(1);            // mono
  w32(static_cast<uint32_t>(rate));
  w32(byte_rate);
  w16(2);            // block align
  w16(16);           // bits/sample
  std::fwrite("data", 1, 4, f);
  w32(data_bytes);
}

struct Sink {
  std::vector<int16_t> pcm;
};

void Emit(void* user, const int16_t* samples, int n) {
  auto* s = static_cast<Sink*>(user);
  s->pcm.insert(s->pcm.end(), samples, samples + n);
}

}  // namespace

int main(int argc, char** argv) {
  std::string text;
  std::string out = "out.wav";
  std::string pack_path = kDefaultPack;
  bool ipa = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-o" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--pack" && i + 1 < argc) {
      pack_path = argv[++i];
    } else if (a == "--ipa") {
      ipa = true;
    } else if (a == "-h" || a == "--help") {
      std::fprintf(stderr,
                   "usage: %s [--pack PACK.bin] [--ipa] [-o OUT.wav|-] TEXT\n",
                   argv[0]);
      return 2;
    } else if (!a.empty() && a[0] == '-' && a != "-") {
      std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
      return 2;
    } else {
      if (!text.empty()) text += " ";
      text += a;
    }
  }
  if (text.empty()) {
    std::fprintf(stderr, "error: no text given\n");
    return 2;
  }

  std::vector<uint8_t> pack = ReadFile(pack_path.c_str());
  if (pack.empty()) {
    std::fprintf(stderr, "error: could not read pack '%s'\n", pack_path.c_str());
    return 1;
  }

  // Arena: the engine's transient working set (TFLM tensors + planning). The
  // firmware lends it ~340 KiB; give the host build generous headroom.
  std::vector<uint8_t> arena(1u << 20);  // 1 MiB

  neural_tts::NeuralTts tts(pack.data(), arena.data(), arena.size());
  if (!tts.ok()) {
    std::fprintf(stderr, "error: neural TTS init failed (pack or arena)\n");
    return 1;
  }

  Sink sink;
  const int rc = ipa ? tts.SynthesizeIpa(text.c_str(), Emit, &sink)
                     : tts.Synthesize(text.c_str(), Emit, &sink);
  if (rc < 0) {
    std::fprintf(stderr, "error: synth failed (%d)\n", rc);
    return 1;
  }

  const int rate = neural_tts::NeuralTts::kSampleRate;
  const int n = static_cast<int>(sink.pcm.size());
  if (out == "-") {
    std::fwrite(sink.pcm.data(), sizeof(int16_t), sink.pcm.size(), stdout);
  } else {
    FILE* f = std::fopen(out.c_str(), "wb");
    if (!f) {
      std::fprintf(stderr, "error: could not open '%s' for writing\n",
                   out.c_str());
      return 1;
    }
    WriteWavHeader(f, rate, n);
    std::fwrite(sink.pcm.data(), sizeof(int16_t), sink.pcm.size(), f);
    std::fclose(f);
  }

  const neural_tts::NeuralTts::Stats& st = tts.stats();
  std::fprintf(stderr,
               "[tts_cli] \"%s\" -> %s  (%d samples, %.2f s @ %d Hz)\n"
               "[tts_cli] chunks=%d tiles=%d\n",
               text.c_str(), out.c_str(), n,
               static_cast<double>(n) / rate, rate, st.chunks, st.tiles);
  return 0;
}

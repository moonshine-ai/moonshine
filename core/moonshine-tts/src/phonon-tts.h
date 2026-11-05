#ifndef MOONSHINE_TTS_PHONON_TTS_H
#define MOONSHINE_TTS_PHONON_TTS_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "chunk-policy.h"
#include "file-information.h"
#include "moonshine-g2p-options.h"
#include "tts-stream.h"

namespace moonshine_tts {

/// Options for the Phonon autoregressive ONNX TTS engine.
///
/// Phonon is Gradium's family of small streaming text-to-speech models
/// (https://gradium.ai/phonon), built on Kyutai's Pocket TTS and Mimi codec.
/// This engine runs ``phonon-31m-en``
/// (https://huggingface.co/gradium/phonon-31m-en) and its Mimi decoder
/// (https://huggingface.co/gradium/phonon-mimi), both Apache 2.0, as ORT graphs
/// cut by ``scripts/export-phonon-model.py``, with the generation loop of
/// Gradium's ``xn-ptts`` runtime ported to C++.
///
/// The voice is a prompt of precomputed speaker latents rather than a model:
/// ``voice_id`` names one of the catalog voices shipped under
/// ``phonon/voices/`` (from https://huggingface.co/gradium/phonon-voices,
/// Apache 2.0), or ``voice_latents`` supplies one directly. The encoder that
/// turns a recording into those latents is not published; Gradium offers
/// cloning and voice design as a hosted service (Gradium Studio,
/// https://studio.gradium.ai) whose downloaded ``.safetensors`` files load
/// here unchanged. On-device cloning in Moonshine stays with ZipVoice.
struct PhononTTSOptions {
  /// Locale tag. The shipped model is English only.
  std::string lang = "en_us";
  /// Speaking rate. Phonon has no time-stretch control; speed is mapped onto
  /// the model's ``duration_delta`` conditioning (see ``duration_delta``), so
  /// the effect is approximate and saturates past about 0.7x–1.4x.
  double speed = 1.0;
  /// Direct control of the duration conditioning, when finite. Positive values
  /// slow speech down, negative speed it up; the useful range is about
  /// ``[-1, 1]``. Overrides ``speed`` when set.
  float duration_delta = kUnsetDurationDelta;
  MoonshineG2POptions g2p_options{};
  std::vector<std::string> ort_provider_names{};
  std::string coreml_cache_dir{};
  /// Peak-normalize the final waveform. Off by default: the model's output
  /// level is already the speaker's. Clipping to ``[-1, 1]`` always applies.
  bool normalize_audio = false;
  float output_volume = 1.F;

  /// Sampling temperature for the flow head's Gaussian noise; the latent is
  /// ``noise * sqrt(temperature) + velocity``. Negative means the model's
  /// default (0.3). Zero is deterministic.
  float temperature = -1.F;
  /// Seed for the noise, so a given text renders the same way every call.
  unsigned int seed = 0U;
  /// Run the reference runtime's text normalization (typography folding,
  /// numbers read out) before tokenizing.
  bool normalize_text = true;

  /// Catalog voice id (the ``voices.json`` ``id``, e.g. ``zoey``), used when
  /// ``voice_latents`` is empty.
  std::string voice_id{};
  /// A voice supplied directly: ``voice_frames`` frames of
  /// ``audio_prompt_dim`` (512) floats, frame-major, as the ``speaker_wavs``
  /// tensor of a Phonon voice file holds them once transposed.
  std::vector<float> voice_latents{};
  int voice_frames = 0;

  /// In-memory Phonon assets keyed by ``phonon/backbone.ort`` etc. (see
  /// moonshine-tts-options.h).
  FileInformationMap tts_asset_files{};

  static constexpr float kUnsetDurationDelta = -1000.F;
};

/// One catalog voice, from ``phonon/voices.json``.
struct PhononVoiceInfo {
  std::string id{};
  std::string name{};
  std::string locale{};
  std::string gender{};
  std::string description{};
  /// Path of the voice file, relative to the ``phonon/`` directory.
  std::string file{};
};

class PhononTTS {
 public:
  explicit PhononTTS(const PhononTTSOptions& opt);
  PhononTTS(const PhononTTS&) = delete;
  PhononTTS& operator=(const PhononTTS&) = delete;
  PhononTTS(PhononTTS&&) noexcept;
  PhononTTS& operator=(PhononTTS&&) noexcept;
  ~PhononTTS();

  static constexpr int kSampleRateHz = 24000;
  /// Samples per generated frame (80 ms at 24 kHz).
  static constexpr int kFrameSamples = 1920;

  void set_speed(double speed);
  double speed() const;
  void set_duration_delta(float delta);
  float duration_delta() const;
  void set_temperature(float temperature);
  float temperature() const;
  void set_seed(unsigned int seed);
  unsigned int seed() const;
  bool normalize_audio() const;
  void set_normalize_audio(bool on);
  float output_volume() const;
  void set_output_volume(float volume);

  /// Text -> prompts -> autoregressive latents -> Mimi -> mono float waveform
  /// at ``kSampleRateHz``, with the output effects applied.
  std::vector<float> synthesize(std::string_view text);

  /// Normalize ``text`` and group it into the prompts ``begin`` takes, the
  /// way ``synthesize`` does internally.
  std::vector<std::string> plan_prompts(std::string_view text) const;

  /// One prompt being generated, a frame at a time. Produced by ``begin``;
  /// the engine must outlive it and only one may be driven at once.
  class Generation {
   public:
    ~Generation();
    Generation(const Generation&) = delete;
    Generation& operator=(const Generation&) = delete;

    /// Whether ``next_frame`` has anything left to give. Known without
    /// running the model, which is what lets a stream mark its final chunk.
    bool done() const;
    /// The next ``kFrameSamples`` samples with ``output_volume`` and clipping
    /// applied, or empty once ``done``. Peak normalization needs the whole
    /// utterance and is not applied here.
    std::vector<float> next_frame();

   private:
    friend class PhononTTS;
    struct State;
    explicit Generation(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
  };

  /// Start generating one prompt of plain text (already normalized and
  /// grouped; see ``plan_prompts``). The prefill is paid here.
  std::unique_ptr<Generation> begin(std::string_view prompt);

  /// Voices listed in ``phonon/voices.json`` under ``root`` (or in the
  /// in-memory assets). Empty when the catalog is missing.
  static std::vector<PhononVoiceInfo> list_voices(
      const std::filesystem::path& root, const FileInformationMap& files);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Streams a Phonon utterance as it is generated.
///
/// Phonon is autoregressive, so unlike the sliced sources there is nothing to
/// analyze up front and no frame count to plan against: `begin` pays the
/// prompt prefill, then each `next` runs the model for as many 80 ms frames as
/// the chunk schedule asks for and hands back exactly those samples. Frames
/// are decoded with the state a whole render would have had, so chunks join
/// without crossfades, and `has_more` is answered from the end-of-speech
/// countdown rather than by generating ahead.
///
/// Chunk lengths follow `ChunkPolicyOptions::first_chunk_seconds` and
/// `growth`; `tolerance_seconds`, `crossfade_seconds` and `pad_frames` are
/// unused. Text over one prompt's budget is generated as several prompts in
/// turn, each a sentence group (see `PhononTTS::plan_prompts`).
class PhononChunkSource : public ChunkSource {
 public:
  PhononChunkSource(PhononTTS& tts, ChunkPolicyOptions policy);

  void begin(std::string_view text) override;
  std::vector<float> next() override;
  bool has_more() const override;
  int sample_rate() const override;

 private:
  /// Move to the next prompt that has frames to give, or clear `gen_`.
  void advance();

  PhononTTS& tts_;
  ChunkPolicyOptions policy_;
  std::vector<std::string> prompts_{};
  size_t next_prompt_ = 0;
  std::unique_ptr<PhononTTS::Generation> gen_{};
  int chunks_emitted_ = 0;
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_PHONON_TTS_H

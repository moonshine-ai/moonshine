// Measures the reference peak of every installed Kokoro voice, offline.
//
// One-shot synthesis peak-normalizes, which needs the finished waveform.
// Streaming cannot wait for that, so it needs a gain chosen before the decoder
// runs or its output comes out quieter than the same text spoken in one go, and
// by a different amount for each voice.
//
// That gain has to be predicted from something. Measurement ruled out the
// candidates that would have been free: the prosody stage's energy curve tracks
// the shape of an utterance's level but not its scale (within-voice correlation
// about -0.2), and the style vector predicts nothing (leave-one-out residual
// worse than assuming the median). What does predict it is which voice is
// speaking, which accounts for most of the variation and is known before any
// text arrives. Hence a static table, measured here and read at synthesis time.
//
// Writes the generated header that kokoro-voice-levels.h declares. Run it after
// changing the Kokoro model or the shipped voice set:
//
//   tts-voice-level-calibrate --model-root core/moonshine-tts/data \
//       --out core/moonshine-tts/src/kokoro-voice-levels-data.h
//
// Without --out it prints the measurements and the accuracy they imply, which
// is the useful mode when checking whether a change moved the levels.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "moonshine-c-api.h"

namespace {

// Deliberately varied in length, punctuation and loudness: a voice's peak moves
// with what it is saying, so a narrow set would measure the phrases rather than
// the voice.
//
// Each language needs its own, because a voice fed text its phonemizer does not
// handle produces silence, or worse, something with a level that has nothing to
// do with how the voice really sounds.
struct PhraseSet {
  std::string_view language;
  std::vector<std::string_view> phrases;
};

const std::vector<PhraseSet>& calibration_phrases() {
  static const std::vector<PhraseSet> sets = {
      {"en",
       {"Yes.", "I can do that for you.",
        "The old lighthouse stood alone against the crashing waves of the "
        "north sea.",
        "Let me check that for you, it should only take a moment.",
        "Your appointment is confirmed for Tuesday the fourteenth at half "
        "past two.",
        "I'm sorry, I didn't catch that. Could you say it again?",
        "According to the latest figures, revenue grew by twelve percent "
        "this quarter.",
        "Turn left at the next junction, then continue for about two miles.",
        "It is a truth universally acknowledged that a single man in "
        "possession of a good fortune must be in want of a wife.",
        "That's a great question, and the answer depends on what you're "
        "trying to do.",
        "Rain is expected later this afternoon, so you may want to take an "
        "umbrella.",
        "There were three of them waiting by the door, and none said a word.",
        "Wait! Stop right there!",
        "The quick brown fox jumps over the lazy dog."}},
      {"es",
       {"Sí.", "Puedo hacer eso por usted.",
        "El viejo faro se alzaba solo frente a las olas del mar del norte.",
        "Déjame comprobarlo, solo tardará un momento.",
        "Su cita está confirmada para el martes catorce a las dos y media.",
        "Lo siento, no le he entendido. ¿Puede repetirlo?",
        "Según las últimas cifras, los ingresos crecieron un doce por ciento.",
        "Gire a la izquierda en el próximo cruce y siga unos tres kilómetros.",
        "¡Espera! ¡Detente ahí mismo!",
        "Se espera lluvia por la tarde, así que lleve un paraguas."}},
      {"fr",
       {"Oui.", "Je peux faire cela pour vous.",
        "Le vieux phare se dressait seul face aux vagues de la mer du nord.",
        "Laissez-moi vérifier, cela ne prendra qu'un instant.",
        "Votre rendez-vous est confirmé pour mardi quatorze à quatorze "
        "heures trente.",
        "Je suis désolé, je n'ai pas compris. Pouvez-vous répéter ?",
        "Selon les derniers chiffres, les revenus ont augmenté de douze pour "
        "cent.",
        "Tournez à gauche au prochain carrefour, puis continuez trois "
        "kilomètres.",
        "Attends ! Arrête-toi tout de suite !",
        "De la pluie est prévue cet après-midi, prenez un parapluie."}},
      {"it",
       {"Sì.", "Posso farlo per lei.",
        "Il vecchio faro si ergeva solo contro le onde del mare del nord.",
        "Lasci che controlli, ci vorrà solo un momento.",
        "Il suo appuntamento è confermato per martedì quattordici alle due e "
        "mezza.",
        "Mi scusi, non ho capito. Può ripetere?",
        "Secondo gli ultimi dati, i ricavi sono cresciuti del dodici per "
        "cento.",
        "Giri a sinistra al prossimo incrocio e prosegua per tre chilometri.",
        "Aspetta! Fermati subito!",
        "Nel pomeriggio è prevista pioggia, quindi prenda un ombrello."}},
      {"pt_br",
       {"Sim.", "Posso fazer isso para você.",
        "O velho farol erguia-se sozinho diante das ondas do mar do norte.",
        "Deixe-me verificar, vai levar apenas um momento.",
        "Sua consulta está confirmada para terça-feira, dia catorze, às duas "
        "e meia.",
        "Desculpe, não entendi. Pode repetir?",
        "Segundo os últimos números, a receita cresceu doze por cento.",
        "Vire à esquerda no próximo cruzamento e siga por três quilômetros.",
        "Espere! Pare aí mesmo!",
        "Há previsão de chuva à tarde, então leve um guarda-chuva."}},
      {"hi",
       {"हाँ।", "मैं आपके लिए यह कर सकता हूँ।",
        "पुराना प्रकाशस्तंभ उत्तरी समुद्र की लहरों के सामने अकेला खड़ा था।",
        "मुझे जाँचने दीजिए, इसमें बस एक पल लगेगा।",
        "आपका समय मंगलवार चौदह तारीख को ढाई बजे तय है।",
        "क्षमा करें, मैं समझ नहीं पाया। क्या आप दोहरा सकते हैं?",
        "नवीनतम आंकड़ों के अनुसार, आय में बारह प्रतिशत की वृद्धि हुई।",
        "अगले चौराहे से बाएँ मुड़ें और तीन किलोमीटर आगे चलें।", "रुको! वहीं रुक जाओ!",
        "दोपहर बाद बारिश की संभावना है, छाता साथ ले जाइए।"}},
      {"ja",
       {"はい。", "それはこちらで対応できます。",
        "古い灯台は北の海の荒波に向かって一つだけ立っていた。",
        "確認しますので、少々お待ちください。",
        "ご予約は十四日火曜日の二時半で確定しています。",
        "すみません、聞き取れませんでした。もう一度お願いします。",
        "最新の数字によると、売上は十二パーセント増加しました。",
        "次の交差点を左に曲がって、三キロほど進んでください。",
        "待って！そこで止まって！",
        "午後から雨が降る予報なので、傘を持って行ってください。"}},
      {"zh",
       {"是的。", "我可以为您处理这件事。",
        "那座旧灯塔独自矗立在北方海浪之前。", "让我确认一下，只需要一会儿。",
        "您的预约已确认，时间是十四号星期二两点半。",
        "抱歉，我没有听清。您可以再说一遍吗？",
        "根据最新数据，收入增长了百分之十二。",
        "在下一个路口左转，然后继续走三公里。", "等一下！马上停下！",
        "下午预计有雨，请带上雨伞。"}},
  };
  return sets;
}

/// The phrases to measure `language` with, falling back to English so a new
/// language still gets a level rather than being skipped.
const std::vector<std::string_view>& phrases_for(std::string_view language) {
  const std::vector<PhraseSet>& sets = calibration_phrases();
  for (const PhraseSet& set : sets) {
    if (set.language == language) {
      return set.phrases;
    }
  }
  return sets.front().phrases;
}

struct VoiceLevels {
  std::string id;
  std::vector<float> peaks;
  /// Per utterance, the sample magnitudes above half the peak, sorted. Enough
  /// to work out how much of an utterance a given gain would push past the
  /// clamp without keeping every sample of every render.
  std::vector<std::vector<float>> loud_samples;
  size_t total_samples = 0;
};

/// The peak the table stores for a voice.
///
/// A high quantile rather than the median, because the two directions of error
/// are not equal: underestimating makes the gain too large and clips, while
/// overestimating only makes that voice slightly quiet.
float reference_peak(std::vector<float> peaks) {
  if (peaks.empty()) {
    return 0.F;
  }
  std::sort(peaks.begin(), peaks.end());
  const size_t index = static_cast<size_t>(
      std::llround(0.9 * static_cast<double>(peaks.size() - 1)));
  return peaks[index];
}

std::vector<std::string> installed_voices(
    const std::filesystem::path& model_root) {
  std::vector<std::string> ids;
  const std::filesystem::path dir = model_root / "kokoro" / "voices";
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (entry.path().extension() == ".kokorovoice") {
      ids.push_back(entry.path().stem().string());
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

/// Kokoro voice ids start with a language and gender code, and a synthesizer
/// only accepts voices matching the language it was built for.
std::string_view language_for_voice(std::string_view id) {
  if (id.size() < 2) {
    return "en_us";
  }
  switch (id[0]) {
    case 'b':
      return "en_gb";
    case 'j':
      return "ja";
    case 'z':
      return "zh";
    case 'e':
      return "es";
    case 'f':
      return "fr";
    case 'h':
      return "hi";
    case 'i':
      return "it";
    case 'p':
      return "pt_br";
    default:
      return "en_us";
  }
}

bool measure_voice(const std::string& id, const std::string& model_root,
                   VoiceLevels& out) {
  const std::string language(language_for_voice(id));
  const moonshine_option_t options[] = {
      {"model_root", model_root.c_str()},
      {"lang", language.c_str()},
      {"voice", id.c_str()},
      // The whole point is to measure what the decoder produces, before any
      // level shaping is applied on top.
      {"normalize_audio", "false"},
      {"output_volume", "1.0"},
  };
  const int32_t handle = moonshine_create_tts_synthesizer_from_files(
      language.c_str(), nullptr, 0, options,
      sizeof(options) / sizeof(options[0]), MOONSHINE_HEADER_VERSION);
  if (handle < 0) {
    std::fprintf(stderr, "    (%s: %s)\n", language.c_str(),
                 moonshine_error_to_string(handle));
    return false;
  }
  out.id = id;
  // en_us and en_gb share one set; the rest match the synthesizer's language.
  for (const std::string_view phrase :
       phrases_for(language.rfind("en", 0) == 0 ? "en" : language)) {
    float* audio = nullptr;
    uint64_t count = 0;
    int32_t rate = 0;
    const std::string text(phrase);
    const int32_t status = moonshine_text_to_speech(
        handle, text.c_str(), nullptr, 0, &audio, &count, &rate);
    if (status != MOONSHINE_ERROR_NONE) {
      std::fprintf(stderr, "    (%s: %s)\n", text.substr(0, 20).c_str(),
                   moonshine_error_to_string(status));
      continue;
    }
    // The C API hands back a malloc'd buffer, so tie it to this scope.
    const std::unique_ptr<float, void (*)(void*)> owned(audio, &std::free);
    float peak = 0.F;
    for (uint64_t i = 0; i < count; ++i) {
      peak = std::max(peak, std::fabs(audio[i]));
    }
    if (peak > 1e-6F) {
      std::vector<float> loud;
      for (uint64_t i = 0; i < count; ++i) {
        if (std::fabs(audio[i]) > 0.5F * peak) {
          loud.push_back(std::fabs(audio[i]));
        }
      }
      std::sort(loud.begin(), loud.end());
      out.peaks.push_back(peak);
      out.loud_samples.push_back(std::move(loud));
      out.total_samples += count;
    }
  }
  moonshine_free_tts_synthesizer(handle);
  return !out.peaks.empty();
}

void report(const std::vector<VoiceLevels>& measured) {
  std::vector<double> errors;
  std::printf("%-14s %8s %8s %8s %9s\n", "voice", "median", "max", "stored",
              "worst dB");
  for (const VoiceLevels& voice : measured) {
    std::vector<float> sorted = voice.peaks;
    std::sort(sorted.begin(), sorted.end());
    const float stored = reference_peak(voice.peaks);
    double voice_worst = -1e9;
    for (const float peak : voice.peaks) {
      // How far a real utterance lands from what the table predicts. Positive
      // means louder than predicted, which is the direction that clips.
      const double error_db = 20.0 * std::log10(peak / stored);
      voice_worst = std::max(voice_worst, error_db);
      errors.push_back(error_db);
    }
    std::printf("%-14s %8.3f %8.3f %8.3f %+9.2f\n", voice.id.c_str(),
                sorted[sorted.size() / 2], sorted.back(), stored, voice_worst);
  }
  if (errors.empty()) {
    return;
  }
  std::sort(errors.begin(), errors.end());
  const auto quantile = [&errors](double q) {
    return errors[static_cast<size_t>(
        std::llround(q * static_cast<double>(errors.size() - 1)))];
  };
  // Both tails matter and they cost different things. Above the stored level an
  // utterance risks the clamp; below it, it simply comes out quieter than the
  // one-shot path would have made it, which is the more common outcome and the
  // one a listener actually notices.
  std::printf(
      "\n%zu utterances over %zu voices\n"
      "utterance peak against its voice's stored level, in dB:\n"
      "  min %+.2f  p05 %+.2f  p25 %+.2f  median %+.2f  p75 %+.2f  p95 %+.2f  "
      "max %+.2f\n",
      errors.size(), measured.size(), errors.front(), quantile(0.05),
      quantile(0.25), quantile(0.5), quantile(0.75), quantile(0.95),
      errors.back());

  // The stored level is a high quantile, so most utterances come in under it
  // and the gain has room to spare. What matters is where the loudest ones
  // land: a target leaves 20*log10(1/target) dB before the clamp bites.
  //
  // An utterance "clipping" is not by itself interesting, because clamping the
  // two samples at a waveform's crest is inaudible. The honest cost is how many
  // samples the clamp touches, so that is reported too.
  std::printf("\n%-8s %10s %12s %14s %16s\n", "target", "headroom",
              "utterances", "worst samples", "typical level");
  for (const double target : {1.0, 0.95, 0.9, 0.85, 0.8, 0.7}) {
    const double headroom = -20.0 * std::log10(target);
    size_t utterances = 0;
    size_t worst_samples = 0;
    for (const VoiceLevels& voice : measured) {
      const double stored = reference_peak(voice.peaks);
      for (size_t i = 0; i < voice.peaks.size(); ++i) {
        const double gain = target / stored;
        if (voice.peaks[i] * gain <= 1.0) {
          continue;
        }
        ++utterances;
        // The loud samples are sorted, so the count above the clamp is a
        // partition point rather than a scan.
        const auto& loud = voice.loud_samples[i];
        const auto first = std::upper_bound(loud.begin(), loud.end(),
                                            static_cast<float>(1.0 / gain));
        worst_samples =
            std::max(worst_samples, static_cast<size_t>(loud.end() - first));
      }
    }
    std::printf("%-8.2f %9.2fdB %7zu/%-5zu %15zu %+15.2fdB\n", target, headroom,
                utterances, errors.size(), worst_samples,
                quantile(0.5) - headroom);
  }
}

/// Saves the raw measurements so the choice of stored statistic and target can
/// be revisited without spending another few minutes rendering.
void write_measurements(const std::filesystem::path& path,
                        const std::vector<VoiceLevels>& measured) {
  std::ofstream out(path);
  if (!out) {
    return;
  }
  out << "voice\tpeak\n";
  for (const VoiceLevels& voice : measured) {
    for (const float peak : voice.peaks) {
      out << voice.id << '\t' << peak << '\n';
    }
  }
}

bool write_header(const std::filesystem::path& path,
                  const std::vector<VoiceLevels>& measured) {
  std::ofstream out(path);
  if (!out) {
    return false;
  }
  out << "// Generated by tts-voice-level-calibrate. Do not edit.\n"
      << "//\n"
      << "// The peak each voice reaches when it speaks, measured without\n"
      << "// normalization, so streaming can pick a gain before it has any\n"
      << "// audio to measure. See kokoro-voice-levels.h.\n"
      << "\n"
      << "#ifndef MOONSHINE_TTS_KOKORO_VOICE_LEVELS_DATA_H\n"
      << "#define MOONSHINE_TTS_KOKORO_VOICE_LEVELS_DATA_H\n"
      << "\n"
      << "namespace moonshine_tts {\n"
      << "\n"
      << "// Sorted by id so lookup can binary search.\n"
      << "inline constexpr KokoroVoiceLevel kKokoroVoiceLevels[] = {\n";
  for (const VoiceLevels& voice : measured) {
    char line[128];
    std::snprintf(line, sizeof(line), "    {\"%s\", %.4ff},\n",
                  voice.id.c_str(), reference_peak(voice.peaks));
    out << line;
  }
  out << "};\n"
      << "\n"
      << "}  // namespace moonshine_tts\n"
      << "\n"
      << "#endif  // MOONSHINE_TTS_KOKORO_VOICE_LEVELS_DATA_H\n";
  return out.good();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_root = "core/moonshine-tts/data";
  std::filesystem::path out_path;
  std::string only_voice;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model-root" && i + 1 < argc) {
      model_root = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--voice" && i + 1 < argc) {
      only_voice = argv[++i];
    } else {
      std::fprintf(
          stderr,
          "usage: %s [--model-root DIR] [--out HEADER] [--voice ID]\n"
          "  Without --out, prints the measurements instead of\n"
          "  writing the generated table. --voice measures one voice,\n"
          "  for checking why a voice was skipped.\n",
          argv[0]);
      return 2;
    }
  }

  std::vector<std::string> ids = installed_voices(model_root);
  if (!only_voice.empty()) {
    ids.erase(std::remove_if(ids.begin(), ids.end(),
                             [&only_voice](const std::string& id) {
                               return id != only_voice;
                             }),
              ids.end());
  }
  if (ids.empty()) {
    std::fprintf(stderr, "no .kokorovoice files under %s/kokoro/voices\n",
                 model_root.c_str());
    return 1;
  }

  std::vector<VoiceLevels> measured;
  for (const std::string& id : ids) {
    VoiceLevels levels;
    if (measure_voice(id, model_root, levels)) {
      measured.push_back(std::move(levels));
      std::fprintf(stderr, "  %s\n", id.c_str());
    } else {
      std::fprintf(stderr, "  %s: skipped\n", id.c_str());
    }
  }
  if (measured.empty()) {
    std::fprintf(stderr, "nothing measured\n");
    return 1;
  }

  write_measurements("/tmp/kokoro-voice-peaks.tsv", measured);

  if (out_path.empty()) {
    report(measured);
    return 0;
  }
  if (!write_header(out_path, measured)) {
    std::fprintf(stderr, "could not write %s\n", out_path.string().c_str());
    return 1;
  }
  report(measured);
  std::fprintf(stderr, "\nwrote %s\n", out_path.string().c_str());
  return 0;
}

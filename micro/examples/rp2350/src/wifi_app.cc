#include "wifi_app.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "app_common.h"     // g_tensor_arena, kTensorArenaSize, g_waveform
#include "audio_config.h"   // kClipNumSamples, kNFft
#include "audio_service.h"  // RecognizerInit / RecognizeOne / Speak / kMinResultProb
#include "classes.h"        // kClassLabels
#include "kiss_fftr.h"
#include "spelling_labels.h"  // SpokenForLabel
#include "usb_audio_io.h"

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"

#include "lwip/netif.h"

namespace spelling {

namespace {

// WPA2 password max is 63 chars; SSID max 32. One buffer size covers both.
constexpr std::size_t kFieldMax = 64;

// How long to wait for the link to come up and for DHCP to hand out an address.
constexpr int kConnectTimeoutMs = 30000;
constexpr int kDhcpTimeoutMs = 15000;

enum class State {
  kIdle,         // waiting for "wifi" / "ip"
  kSsid,         // spelling the network name
  kSsidConfirm,  // "yes" to accept the name, "no" to redo
  kPassword,     // spelling the password
};

// What a recognized class label means during credential entry.
enum class Tok {
  kChar,     // produced a credential character (see out_char)
  kCapital,  // capitalize the next letter
  kDelete,   // erase the last character
  kDone,     // finish the current field
  kCancel,   // abort back to idle
  kWifi,     // start the setup flow
  kIp,       // report the current IP
  kYes,
  kNo,
  kIgnore,  // wake word / unmapped
};

const char* DigitWord(char c) {
  static const char* const kWords[10] = {"zero", "one", "two",   "three", "four",
                                         "five", "six", "seven", "eight", "nine"};
  return (c >= '0' && c <= '9') ? kWords[c - '0'] : "";
}

// Symbol class label -> the character it inserts.
bool SymbolChar(const char* label, char* out) {
  if (std::strcmp(label, "star") == 0) { *out = '*'; return true; }
  if (std::strcmp(label, "dollar") == 0) { *out = '$'; return true; }
  if (std::strcmp(label, "underscore") == 0) { *out = '_'; return true; }
  if (std::strcmp(label, "exclamation") == 0) { *out = '!'; return true; }
  if (std::strcmp(label, "percent") == 0) { *out = '%'; return true; }
  return false;
}

// Symbol character -> the word the TTS should say for it.
const char* SymbolWord(char c) {
  switch (c) {
    case '*': return "star";
    case '$': return "dollar";
    case '_': return "underscore";
    case '!': return "exclamation";
    case '%': return "percent";
    default: return "";
  }
}

Tok Classify(const char* label, char* out_char) {
  // Single lowercase letter 'a'..'z'.
  if (label[0] >= 'a' && label[0] <= 'z' && label[1] == '\0') {
    *out_char = label[0];
    return Tok::kChar;
  }
  // Digit words.
  for (char d = '0'; d <= '9'; ++d) {
    if (std::strcmp(label, DigitWord(d)) == 0) {
      *out_char = d;
      return Tok::kChar;
    }
  }
  // Symbol words.
  if (SymbolChar(label, out_char)) return Tok::kChar;

  if (std::strcmp(label, "capital") == 0 ||
      std::strcmp(label, "uppercase") == 0)
    return Tok::kCapital;
  if (std::strcmp(label, "delete") == 0) return Tok::kDelete;
  if (std::strcmp(label, "done") == 0) return Tok::kDone;
  if (std::strcmp(label, "cancel") == 0) return Tok::kCancel;
  if (std::strcmp(label, "wifi") == 0) return Tok::kWifi;
  if (std::strcmp(label, "ip") == 0) return Tok::kIp;
  if (std::strcmp(label, "yes") == 0) return Tok::kYes;
  if (std::strcmp(label, "no") == 0) return Tok::kNo;
  return Tok::kIgnore;  // "hey rp" and anything unmapped
}

// Append the spoken word(s) for one credential character to `dst` (a phrase
// builder), separated by a leading space when `dst` is non-empty.
void AppendSpokenForChar(char* dst, std::size_t cap, char c) {
  char word[24];
  if (c >= 'A' && c <= 'Z') {
    const char low[2] = {static_cast<char>(c - 'A' + 'a'), '\0'};
    std::snprintf(word, sizeof(word), "capital %s", SpokenForLabel(low));
  } else if (c >= 'a' && c <= 'z') {
    const char lab[2] = {c, '\0'};
    std::snprintf(word, sizeof(word), "%s", SpokenForLabel(lab));
  } else if (c >= '0' && c <= '9') {
    std::snprintf(word, sizeof(word), "%s", DigitWord(c));
  } else {
    std::snprintf(word, sizeof(word), "%s", SymbolWord(c));
  }
  const std::size_t len = std::strlen(dst);
  if (len > 0 && len + 1 < cap) {
    dst[len] = ' ';
    dst[len + 1] = '\0';
  }
  std::strncat(dst, word, cap - std::strlen(dst) - 1);
}

// Speak a credential back, spelled out ("see ay tee one"), with an optional
// prefix ("the name is ...").
void SpeakSpelled(const char* prefix, const char* text, AudioOutput& out,
                  AudioInput& in, uint8_t* arena, std::size_t arena_size) {
  static char phrase[640];
  phrase[0] = '\0';
  std::strncat(phrase, prefix, sizeof(phrase) - 1);
  for (const char* p = text; *p != '\0'; ++p) {
    AppendSpokenForChar(phrase, sizeof(phrase), *p);
  }
  Speak(phrase, out, in, arena, arena_size);
}

// Speak the current STA IPv4 address digit-by-digit ("one nine two point ...").
void SpeakIp(AudioOutput& out, AudioInput& in, uint8_t* arena,
             std::size_t arena_size) {
  struct netif* nif = &cyw43_state.netif[CYW43_ITF_STA];
  const ip4_addr_t* a = netif_ip4_addr(nif);
  if (a == nullptr || ip4_addr_get_u32(a) == 0) {
    Speak("no address yet", out, in, arena, arena_size);
    return;
  }
  const int octet[4] = {ip4_addr1(a), ip4_addr2(a), ip4_addr3(a),
                        ip4_addr4(a)};
  printf("[wifi] ip %d.%d.%d.%d\n", octet[0], octet[1], octet[2], octet[3]);
  fflush(stdout);

  static char phrase[256];
  phrase[0] = '\0';
  std::strncat(phrase, "the address is", sizeof(phrase) - 1);
  for (int o = 0; o < 4; ++o) {
    char dec[4];
    std::snprintf(dec, sizeof(dec), "%d", octet[o]);
    for (const char* d = dec; *d != '\0'; ++d) {
      std::size_t len = std::strlen(phrase);
      if (len + 1 < sizeof(phrase)) {
        phrase[len] = ' ';
        phrase[len + 1] = '\0';
      }
      std::strncat(phrase, DigitWord(*d), sizeof(phrase) - std::strlen(phrase) - 1);
    }
    if (o < 3) std::strncat(phrase, " point", sizeof(phrase) - std::strlen(phrase) - 1);
  }
  Speak(phrase, out, in, arena, arena_size);
}

// Join `ssid`/`pw` (WPA2-PSK), waiting for the link and a DHCP lease, pumping
// the CYW43 poll context throughout. Speaks success or failure.
void DoConnect(const char* ssid, const char* pw, AudioOutput& out,
               AudioInput& in, uint8_t* arena, std::size_t arena_size) {
  printf("[wifi] connecting to '%s' (%u-char key)\n", ssid,
         static_cast<unsigned>(std::strlen(pw)));
  fflush(stdout);
  Speak("connecting", out, in, arena, arena_size);

  const int err = cyw43_arch_wifi_connect_timeout_ms(
      ssid, pw, CYW43_AUTH_WPA2_AES_PSK, kConnectTimeoutMs);
  if (err != 0) {
    printf("[wifi] connect failed (err=%d)\n", err);
    fflush(stdout);
    Speak("could not connect", out, in, arena, arena_size);
    return;
  }

  // Link is up; wait for DHCP to assign an address (poll mode: pump the chip).
  struct netif* nif = &cyw43_state.netif[CYW43_ITF_STA];
  const absolute_time_t deadline = make_timeout_time_ms(kDhcpTimeoutMs);
  while (!time_reached(deadline)) {
    cyw43_arch_poll();
    const ip4_addr_t* a = netif_ip4_addr(nif);
    if (a != nullptr && ip4_addr_get_u32(a) != 0) break;
    sleep_ms(50);
  }

  Speak("connected", out, in, arena, arena_size);
  SpeakIp(out, in, arena, arena_size);
}

// Append one recognized credential character to `buf`, applying a pending caps
// flag, and echo it back. Returns the new length.
std::size_t AppendChar(char* buf, std::size_t len, char c, bool* caps,
                       AudioOutput& out, AudioInput& in, uint8_t* arena,
                       std::size_t arena_size) {
  if (len + 1 >= kFieldMax) {
    Speak("full", out, in, arena, arena_size);
    return len;
  }
  if (*caps && c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  *caps = false;
  buf[len++] = c;
  buf[len] = '\0';

  char echo[24];
  echo[0] = '\0';
  AppendSpokenForChar(echo, sizeof(echo), c);
  Speak(echo, out, in, arena, arena_size);
  return len;
}

}  // namespace

void RunWifiApp() {
  uint8_t* arena = g_tensor_arena;
  const std::size_t arena_size = kTensorArenaSize;
  float* window = g_waveform;
  const int WS = kClipNumSamples;

  // One shared 512-pt real-FFT twiddle state for the VAD front-end and the STT
  // log-mel (same n_fft, never concurrent), exactly as the spelling app does.
  kiss_fftr_state* fft = kiss_fftr_alloc(kNFft, /*inverse_fft=*/0, nullptr, nullptr);
  if (fft == nullptr) {
    printf("[wifi] kiss_fftr_alloc failed\n");
    while (true) { /* halt */
    }
  }

  printf("[wifi] bringing up CYW43...\n");
  fflush(stdout);
  if (cyw43_arch_init() != 0) {
    printf("[wifi] cyw43_arch_init failed\n");
    while (true) { /* halt */
    }
  }
  cyw43_arch_enable_sta_mode();
  printf("[wifi] CYW43 up (STA mode)\n");
  fflush(stdout);

  RecognizerInit(fft);

  UsbAudioInput in;
  UsbAudioOutput out;

  static char ssid[kFieldMax];
  static char pw[kFieldMax];
  std::size_t ssid_len = 0;
  std::size_t pw_len = 0;
  bool caps = false;
  State st = State::kIdle;

  printf("\n[wifi] ready: say \"wifi\" to set up a network, or \"ip\".\n");
  fflush(stdout);
  Speak("say wifi to set up a network", out, in, arena, arena_size);

  for (;;) {
    float prob = 0.0f;
    const int pred = RecognizeOne(in, arena, arena_size, window, WS, &prob);
    if (prob < kMinResultProb) {
      // Low-confidence: RESULT already logged by RecognizeOne; just re-listen.
      continue;
    }
    const char* label = spelling::kClassLabels[pred];
    char ch = '\0';
    const Tok tok = Classify(label, &ch);

    switch (st) {
      case State::kIdle:
        if (tok == Tok::kWifi) {
          ssid_len = 0;
          ssid[0] = '\0';
          caps = false;
          Speak("spell the network name then say done", out, in, arena,
                arena_size);
          st = State::kSsid;
        } else if (tok == Tok::kIp) {
          SpeakIp(out, in, arena, arena_size);
        }
        break;

      case State::kSsid:
      case State::kPassword: {
        char* buf = (st == State::kSsid) ? ssid : pw;
        std::size_t& len = (st == State::kSsid) ? ssid_len : pw_len;
        switch (tok) {
          case Tok::kChar:
            len = AppendChar(buf, len, ch, &caps, out, in, arena, arena_size);
            break;
          case Tok::kCapital:
            caps = true;
            break;
          case Tok::kDelete:
            if (len > 0) {
              buf[--len] = '\0';
              Speak("delete", out, in, arena, arena_size);
            }
            break;
          case Tok::kCancel:
            caps = false;
            Speak("cancelled", out, in, arena, arena_size);
            st = State::kIdle;
            Speak("say wifi to set up a network", out, in, arena, arena_size);
            break;
          case Tok::kDone:
            if (st == State::kSsid) {
              if (ssid_len == 0) {
                Speak("the name is empty, please spell it", out, in, arena,
                      arena_size);
                break;
              }
              SpeakSpelled("the name is", ssid, out, in, arena, arena_size);
              Speak("say yes to confirm or no to try again", out, in, arena,
                    arena_size);
              st = State::kSsidConfirm;
            } else {
              // Password complete -> attempt the join right away.
              DoConnect(ssid, pw, out, in, arena, arena_size);
              st = State::kIdle;
              Speak("say wifi to set up another network", out, in, arena,
                    arena_size);
            }
            break;
          default:
            break;  // ignore wifi/ip/yes/no/wake mid-entry
        }
        break;
      }

      case State::kSsidConfirm:
        if (tok == Tok::kYes) {
          pw_len = 0;
          pw[0] = '\0';
          caps = false;
          Speak("spell the password then say done", out, in, arena,
                arena_size);
          st = State::kPassword;
        } else if (tok == Tok::kNo) {
          ssid_len = 0;
          ssid[0] = '\0';
          caps = false;
          Speak("spell the network name then say done", out, in, arena,
                arena_size);
          st = State::kSsid;
        }
        break;
    }
  }
}

}  // namespace spelling

// Daemon: Moonshine TTS streaming over Unix domain sockets → PipeWire (Linux only).
#include "moonshine-tts.h"
#include "moonshine-g2p.h"

// PipeWire — Linux only
#include <pipewire/pipewire.h>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct DaemonOptions {
  std::string input_socket_path    = "/tmp/moonshine-tts-input.sock";
  std::string phoneme_socket_path  = "/tmp/moonshine-tts-phonemes.sock";
  std::string pipewire_target_name;  // empty → default sink
  std::string lang                 = "en_us";
  moonshine_tts::MoonshineTTSOptions tts_options;
};

// ---------------------------------------------------------------------------
// PCM transfer types
// ---------------------------------------------------------------------------

struct PcmChunk {
  std::vector<int16_t> samples;
  bool end_of_utterance = false;  // set on the last chunk of an utterance
};

// Thread-safe queue that transfers int16_t PCM chunks from the worker thread
// to the PipeWire on_process callback.  Uses std::mutex + std::condition_variable.
// Chunks are moved (not copied) to avoid heap allocation on the hot path.
class PcmQueue {
 public:
  // Move chunk into the queue and notify one waiter.
  void push(PcmChunk chunk) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(std::move(chunk));
    }
    cv_.notify_one();
  }

  // Non-blocking pop for the PipeWire callback thread.
  // Returns nullopt immediately when the queue is empty.
  std::optional<PcmChunk> try_pop() {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    PcmChunk chunk = std::move(queue_.front());
    queue_.pop_front();
    return chunk;
  }

  // Blocking pop with timeout.
  // Returns nullopt on timeout or when shutdown() has been called and the
  // queue is empty.
  std::optional<PcmChunk> pop_wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; });
    if (queue_.empty()) {
      return std::nullopt;
    }
    PcmChunk chunk = std::move(queue_.front());
    queue_.pop_front();
    return chunk;
  }

  // Signal all waiters to unblock.  After this call, pop_wait() will return
  // nullopt once the queue is drained.
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      shutdown_ = true;
    }
    cv_.notify_all();
  }

  // Returns true when the queue contains no chunks.
  bool empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.empty();
  }

 private:
  std::deque<PcmChunk>     queue_;
  mutable std::mutex       mu_;
  std::condition_variable  cv_;
  bool                     shutdown_ = false;
};

// ---------------------------------------------------------------------------
// PCM conversion helper
// ---------------------------------------------------------------------------

// Convert a floating-point audio sample in [-1, 1] to signed 16-bit PCM.
// Values outside [-1, 1] are clamped before conversion.
// Requirement 2.1
inline int16_t float_to_s16(float s) {
  float clamped = std::clamp(s, -1.0f, 1.0f);
  return static_cast<int16_t>(std::lroundf(clamped * 32767.0f));
}

// ---------------------------------------------------------------------------
// Utterance framing helpers
// ---------------------------------------------------------------------------

// Returns true for empty strings and strings containing only ASCII whitespace
// (space, tab, carriage return, newline, form feed, vertical tab).
// Requirement 1.7
bool is_whitespace_only(std::string_view text) {
  for (char c : text) {
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\f' && c != '\v') {
      return false;
    }
  }
  return true;
}

// Scans buf for a '\n' within the first max_len bytes:
//   - If '\n' is found: extracts the text up to (not including) the '\n',
//     strips a trailing '\r' if present, and returns it as a std::string
//     (even if the resulting string is empty).
//   - If no '\n' is found and buf.size() > max_len: returns std::nullopt —
//     caller should treat this as an oversized-buffer condition and close
//     the connection.
//   - If no '\n' is found and buf.size() <= max_len: returns std::nullopt —
//     incomplete line; caller should wait for more data.
// Requirement 1.5
std::optional<std::string> parse_utterance_line(std::string_view buf,
                                                size_t max_len) {
  // Only scan up to max_len bytes for the newline.
  std::string_view search_region = buf.substr(0, std::min(buf.size(), max_len));
  auto pos = search_region.find('\n');

  if (pos != std::string_view::npos) {
    // Found a newline within the allowed region — extract the line.
    std::string_view line = buf.substr(0, pos);
    // Strip trailing '\r' for '\r\n' line endings.
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    return std::string(line);
  }

  // No newline found. Check whether the buffer has grown beyond the limit.
  if (buf.size() > max_len) {
    // Oversized buffer with no newline — signal the caller to discard and close.
    return std::nullopt;
  }

  // Incomplete line; need more data.
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--input-socket PATH] [--phoneme-socket PATH]"
         " [--pipewire-target-name NAME]\n"
         "       [--model-root DIR] [--lang LANG] [--voice ID] [--speed N]\n\n"
      << "  --input-socket PATH        Unix socket path for text utterance input\n"
         "                             (default: /tmp/moonshine-tts-input.sock)\n"
      << "  --phoneme-socket PATH      Unix socket path for IPA phoneme output\n"
         "                             (default: /tmp/moonshine-tts-phonemes.sock)\n"
      << "  --pipewire-target-name NAME  PipeWire sink node name to connect to\n"
         "                             (default: PipeWire default audio sink)\n"
      << "  --model-root DIR           Root directory for TTS model assets\n"
         "                             (default: process current working directory)\n"
      << "  --lang LANG                Language tag, e.g. en_us, de, ja\n"
         "                             (default: en_us)\n"
      << "  --voice ID                 Voice ID, optionally prefixed with kokoro_ or piper_\n"
      << "  --speed N                  Speech speed multiplier; must be a positive finite number\n"
         "                             (default: 1.0)\n";
}

}  // anonymous namespace

// When MOONSHINE_TTS_STREAMING_TESTABLE is defined, testable helpers are
// placed at file scope so that the test translation unit can call them directly
// after #including this file (or linking against an object compiled with the
// flag). Otherwise they live inside the anonymous namespace.
#ifdef MOONSHINE_TTS_STREAMING_TESTABLE
using DaemonOptionsForTest = DaemonOptions;
#define MOONSHINE_STREAMING_HELPER_SCOPE  /* file scope */
#else
namespace {
#define MOONSHINE_STREAMING_HELPER_SCOPE  /* anonymous namespace */
#endif

// Parses command-line arguments and returns a populated DaemonOptions struct.
// Recognised daemon-specific flags: --input-socket, --phoneme-socket,
// --pipewire-target-name.  TTS flags --model-root, --lang, --voice, --speed
// are delegated to MoonshineTTSOptions::parse_options.
// Unrecognised arguments: print error + usage to stderr, exit(2).
// Invalid --speed (zero, negative, NaN, infinity): print descriptive error
// including the value and "must be a positive finite number", exit(2).
//
// Requirements: 1.1, 1.2, 2.2, 3.1, 4.1–4.9
DaemonOptions parse_cli_args(int argc, char** argv) {
  using moonshine_tts::MoonshineTTSOptions;

  // Collect all --key value pairs from the command line.
  // Daemon-specific keys are consumed here; remaining pairs are forwarded to
  // MoonshineTTSOptions::parse_options.
  std::vector<std::pair<std::string, std::string>> pairs;

  for (int i = 1; i < argc;) {
    const std::string a = argv[i];

    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      std::exit(0);
    }

    if (a.rfind("--", 0) == 0) {
      const std::string key = a.substr(2);
      if (key.empty()) {
        std::cerr << "Error: empty option name \"--\".\n";
        usage(argv[0]);
        std::exit(2);
      }
      if (i + 1 >= argc) {
        std::cerr << "Error: missing value for --" << key << '\n';
        usage(argv[0]);
        std::exit(2);
      }
      pairs.emplace_back(key, argv[i + 1]);
      i += 2;
      continue;
    }

    // Positional arguments are not supported by this daemon.
    std::cerr << "Error: unrecognised argument \"" << a << "\".\n";
    usage(argv[0]);
    std::exit(2);
  }

  DaemonOptions opts;

  // Separate daemon-specific flags from TTS flags.
  std::vector<std::pair<std::string, std::string>> tts_pairs;
  for (const auto& [key, value] : pairs) {
    if (key == "input-socket") {
      opts.input_socket_path = value;
    } else if (key == "phoneme-socket") {
      opts.phoneme_socket_path = value;
    } else if (key == "pipewire-target-name") {
      opts.pipewire_target_name = value;
    } else if (key == "model-root" || key == "lang" || key == "language" ||
               key == "voice"      || key == "speed") {
      // Forward TTS-recognised keys.
      tts_pairs.emplace_back(key, value);
    } else {
      std::cerr << "Error: unrecognised argument \"--" << key << "\".\n";
      usage(argv[0]);
      std::exit(2);
    }
  }

  // Delegate TTS-specific options to MoonshineTTSOptions::parse_options.
  try {
    opts.tts_options.parse_options(tts_pairs, &opts.lang, nullptr);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    usage(argv[0]);
    std::exit(2);
  }

  // Validate speed: must be a positive finite number.
  if (!std::isfinite(opts.tts_options.speed) || opts.tts_options.speed <= 0.0) {
    std::cerr << "Error: invalid --speed value " << opts.tts_options.speed
              << ": must be a positive finite number.\n";
    std::exit(2);
  }

  return opts;
}

#ifndef MOONSHINE_TTS_STREAMING_TESTABLE
}  // anonymous namespace (closed — was re-opened above for parse_cli_args)
#endif

struct SocketServer {
  int fd = -1;         // bound+listening socket fd
  std::string path;    // filesystem path, for unlink on exit

  // Bind and listen on a SOCK_STREAM Unix domain socket at socket_path.
  // If path already exists (any file type): attempt unlink().
  //   - If unlink fails with errno != ENOENT: print descriptive error to stderr
  //     (include path and strerror(errno)) and exit(1)
  // Create AF_UNIX SOCK_STREAM socket; bind; listen(backlog=5).
  // Set O_NONBLOCK on the listening fd.
  // On any other failure: throw std::runtime_error with descriptive message.
  void bind_and_listen(const std::string& socket_path) {
    // sockaddr_un.sun_path is limited to 108 chars on Linux
    if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
      throw std::runtime_error(
          "Socket path too long (max " +
          std::to_string(sizeof(sockaddr_un::sun_path) - 1) + " chars): " +
          socket_path);
    }

    // Remove stale socket file if it exists
    if (::unlink(socket_path.c_str()) != 0) {
      if (errno != ENOENT) {
        std::cerr << "moonshine-tts-streaming: failed to unlink stale socket "
                  << socket_path << ": " << std::strerror(errno) << "\n";
        std::exit(1);
      }
    }

    int sock_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd < 0) {
      throw std::runtime_error(
          std::string("socket() failed: ") + std::strerror(errno));
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    // Safe: length checked above
    socket_path.copy(addr.sun_path, socket_path.size());
    addr.sun_path[socket_path.size()] = '\0';

    if (::bind(sock_fd, reinterpret_cast<const sockaddr*>(&addr),
               sizeof(sa_family_t) + socket_path.size() + 1) != 0) {
      ::close(sock_fd);
      throw std::runtime_error(
          std::string("bind() failed on ") + socket_path + ": " +
          std::strerror(errno));
    }

    if (::listen(sock_fd, 5) != 0) {
      ::close(sock_fd);
      throw std::runtime_error(
          std::string("listen() failed on ") + socket_path + ": " +
          std::strerror(errno));
    }

    // Set O_NONBLOCK on the listening fd
    int flags = ::fcntl(sock_fd, F_GETFL, 0);
    if (flags < 0 || ::fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
      ::close(sock_fd);
      throw std::runtime_error(
          std::string("fcntl(O_NONBLOCK) failed on ") + socket_path + ": " +
          std::strerror(errno));
    }

    fd   = sock_fd;
    path = socket_path;
  }

  // Non-blocking accept. Uses accept4 with SOCK_NONBLOCK.
  // Returns -1 if no connection pending (EAGAIN or EWOULDBLOCK).
  // Returns the new client fd on success.
  int try_accept() {
    int client_fd = ::accept4(fd, nullptr, nullptr, SOCK_NONBLOCK);
    if (client_fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return -1;
      }
      throw std::runtime_error(
          std::string("accept4() failed: ") + std::strerror(errno));
    }
    return client_fd;
  }

  // Close the listening fd and unlink the socket file.
  // Errors in close/unlink are silently ignored (safe for destructor).
  void close_and_unlink() {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
    if (!path.empty()) {
      ::unlink(path.c_str());
      path.clear();
    }
  }

  // Destructor calls close_and_unlink().
  ~SocketServer() {
    close_and_unlink();
  }
};

}  // namespace

int main(int argc, char** argv) {
  return 0;
}

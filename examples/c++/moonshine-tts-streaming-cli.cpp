// moonshine-tts-streaming-cli.cpp
//
// Linux-only streaming TTS CLI.
//
// Architecture
// ============
//
//   TextSocketServer  ──text lines──►  SynthesisWorker  ──PCM chunks──►  AudioRing
//   LineSocketServer  ◄──IPA lines──  (synthesis thread)               ▼
//                                                                  PipeWireSink
//                                                            (PipeWire data-loop)
//
// Sockets
// -------
//   --text-sock   (default /tmp/moonshine-tts.sock)
//       Send a UTF-8 text line terminated by '\n'.  Each line is synthesised
//       in order and played back through PipeWire.
//
//   --phoneme-sock  (default /tmp/moonshine-phonemes.sock)
//       After each synthesis the IPA string is broadcast to every connected
//       client as a '\n'-terminated line.
//
// Build (Linux)
// -------------
//   g++ moonshine-tts-streaming-cli.cpp \
//       -std=c++17 \
//       -Imoonshine-voice-linux-x86_64/include \
//       -Lmoonshine-voice-linux-x86_64/lib \
//       -lmoonshine \
//       $(pkg-config --cflags --libs libpipewire-0.3) \
//       -lpthread \
//       -o moonshine-tts-streaming-cli
//   export LD_LIBRARY_PATH="$(pwd)/moonshine-voice-linux-x86_64/lib"

#ifdef __linux__

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/utils/ringbuffer.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <poll.h>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "moonshine-cpp.h"

// ─────────────────────────────────────────────────────────────────────────────
// Global shutdown flag – set by SIGINT / SIGTERM.
// ─────────────────────────────────────────────────────────────────────────────

static std::atomic<bool> g_shutdown{false};

static void handleSignal(int /*sig*/) {
  g_shutdown.store(true, std::memory_order_relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// ThreadSafeQueue
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
class ThreadSafeQueue {
 public:
  /// Push an item; always succeeds.
  void push(T item) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.push_back(std::move(item));
    }
    cv_.notify_one();
  }

  /// Block until an item is available or shutdown is signalled.
  /// Returns true and fills *out on success; returns false on shutdown.
  bool pop(T &out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return !q_.empty() || g_shutdown.load(); });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  /// Unblock any waiting pop() during shutdown.
  void notifyShutdown() { cv_.notify_all(); }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> q_;
};

// ─────────────────────────────────────────────────────────────────────────────
// AudioRing – lock-free SPSC ring buffer wrapping spa_ringbuffer.
//
// The synthesis thread writes (producer); the PipeWire data callback reads
// (consumer).  spa_ringbuffer is safe for this SPSC pattern without
// additional locks when acquire/release ordering is respected, which the
// PipeWire implementation guarantees via __atomic builtins.
// ─────────────────────────────────────────────────────────────────────────────

class AudioRing {
 public:
  // Must be a power of two.
  static constexpr uint32_t kCapacity = 1u << 17;  // 131072 floats ~= 3 s @44.1 kHz

  AudioRing() {
    spa_ringbuffer_init(&rb_);
    buf_.resize(kCapacity, 0.0f);
  }

  // Producer: called from the synthesis thread.
  void write(const float *src, uint32_t n_frames) {
    uint32_t written = 0;
    while (written < n_frames) {
      uint32_t index;
      int32_t space =
          static_cast<int32_t>(kCapacity) -
          static_cast<int32_t>(spa_ringbuffer_get_write_index(&rb_, &index));
      if (space <= 0) {
        // Exit spin during shutdown so SynthesisWorker can be joined.
        if (g_shutdown.load(std::memory_order_relaxed)) return;
        // Ring is full – sleep briefly and retry rather than dropping audio.
        struct timespec ts = {0, 500000};  // 0.5 ms
        nanosleep(&ts, nullptr);
        continue;
      }
      uint32_t chunk =
          std::min(static_cast<uint32_t>(space), n_frames - written);
      uint32_t pos  = index & (kCapacity - 1);
      uint32_t tail = kCapacity - pos;
      if (chunk <= tail) {
        std::memcpy(&buf_[pos], src + written, chunk * sizeof(float));
      } else {
        std::memcpy(&buf_[pos], src + written, tail * sizeof(float));
        std::memcpy(&buf_[0], src + written + tail,
                    (chunk - tail) * sizeof(float));
      }
      spa_ringbuffer_write_update(&rb_, index + chunk);
      written += chunk;
    }
  }

  // Consumer: called from the PipeWire RT data callback.
  // Fills dst with up to n_frames frames; pads with silence if underrun.
  void read(float *dst, uint32_t n_frames) {
    uint32_t index;
    int32_t avail = spa_ringbuffer_get_read_index(&rb_, &index);
    uint32_t readable = (avail > 0) ? static_cast<uint32_t>(avail) : 0;
    uint32_t chunk = std::min(readable, n_frames);
    if (chunk > 0) {
      uint32_t pos  = index & (kCapacity - 1);
      uint32_t tail = kCapacity - pos;
      if (chunk <= tail) {
        std::memcpy(dst, &buf_[pos], chunk * sizeof(float));
      } else {
        std::memcpy(dst, &buf_[pos], tail * sizeof(float));
        std::memcpy(dst + tail, &buf_[0], (chunk - tail) * sizeof(float));
      }
      spa_ringbuffer_read_update(&rb_, index + chunk);
    }
    if (chunk < n_frames)
      std::memset(dst + chunk, 0, (n_frames - chunk) * sizeof(float));
  }

 private:
  spa_ringbuffer rb_;
  std::vector<float> buf_;
};

// ─────────────────────────────────────────────────────────────────────────────
// PipeWireSink
//
// Opens a PipeWire F32 mono playback stream.  Drains AudioRing in the
// real-time data callback.  Runs pw_main_loop on a dedicated thread.
// ─────────────────────────────────────────────────────────────────────────────

class PipeWireSink {
 public:
  PipeWireSink(AudioRing &ring, uint32_t sample_rate, uint32_t channels)
      : ring_(ring), sample_rate_(sample_rate), channels_(channels) {
    loop_ = pw_main_loop_new(nullptr);
    if (!loop_) throw std::runtime_error("pw_main_loop_new failed");

    pw_loop *pw_loop_ptr = pw_main_loop_get_loop(loop_);

    stream_ = pw_stream_new_simple(
        pw_loop_ptr,
        "moonshine-tts",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE,     "Audio",
            PW_KEY_MEDIA_CATEGORY, "Playback",
            PW_KEY_MEDIA_ROLE,     "Music",
            nullptr),
        &kStreamEvents,
        this);
    if (!stream_) {
      pw_main_loop_destroy(loop_);
      throw std::runtime_error("pw_stream_new_simple failed");
    }

    uint8_t buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    struct spa_audio_info_raw info = {};
    info.format   = SPA_AUDIO_FORMAT_F32;
    info.rate     = sample_rate_;
    info.channels = channels_;

    const struct spa_pod *params[1];
    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

    int ret = pw_stream_connect(
        stream_,
        PW_DIRECTION_OUTPUT,
        PW_ID_ANY,
        static_cast<pw_stream_flags>(PW_STREAM_FLAG_AUTOCONNECT |
                                     PW_STREAM_FLAG_MAP_BUFFERS |
                                     PW_STREAM_FLAG_RT_PROCESS),
        params, 1);
    if (ret < 0) {
      pw_stream_destroy(stream_);
      pw_main_loop_destroy(loop_);
      throw std::runtime_error(
          std::string("pw_stream_connect failed: ") + spa_strerror(ret));
    }

    loop_thread_ = std::thread([this] { pw_main_loop_run(loop_); });
  }

  ~PipeWireSink() {
    if (loop_) pw_main_loop_quit(loop_);
    if (loop_thread_.joinable()) loop_thread_.join();
    if (stream_) pw_stream_destroy(stream_);
    if (loop_) pw_main_loop_destroy(loop_);
  }

  PipeWireSink(const PipeWireSink &) = delete;
  PipeWireSink &operator=(const PipeWireSink &) = delete;

 private:
  static void onProcess(void *userdata) {
    auto *self = static_cast<PipeWireSink *>(userdata);

    struct pw_buffer *b = pw_stream_dequeue_buffer(self->stream_);
    if (!b) return;

    struct spa_buffer *buf = b->buffer;
    float *dst = static_cast<float *>(buf->datas[0].data);
    if (!dst) {
      pw_stream_queue_buffer(self->stream_, b);
      return;
    }

    uint32_t stride   = sizeof(float) * self->channels_;
    uint32_t n_frames = buf->datas[0].maxsize / stride;
    if (b->requested > 0 &&
        static_cast<uint32_t>(b->requested) < n_frames)
      n_frames = static_cast<uint32_t>(b->requested);

    self->ring_.read(dst, n_frames * self->channels_);

    buf->datas[0].chunk->offset = 0;
    buf->datas[0].chunk->stride = static_cast<int32_t>(stride);
    buf->datas[0].chunk->size   = n_frames * stride;

    pw_stream_queue_buffer(self->stream_, b);
  }

  static const struct pw_stream_events kStreamEvents;

  AudioRing   &ring_;
  uint32_t     sample_rate_;
  uint32_t     channels_;
  pw_main_loop *loop_   = nullptr;
  pw_stream    *stream_ = nullptr;
  std::thread   loop_thread_;
};

const struct pw_stream_events PipeWireSink::kStreamEvents = {
    .version = PW_VERSION_STREAM_EVENTS,
    .process = PipeWireSink::onProcess,
};

// ─────────────────────────────────────────────────────────────────────────────
// LineSocketServer
//
// Listens on a Unix-domain SOCK_STREAM socket and broadcasts '\n'-terminated
// lines to every connected client.  Used for the phoneme notification socket.
// ─────────────────────────────────────────────────────────────────────────────

class LineSocketServer {
 public:
  explicit LineSocketServer(const std::string &path) : path_(path) {
    listen_fd_ =
        ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (listen_fd_ < 0)
      throw std::runtime_error(
          std::string("LineSocketServer socket(): ") + strerror(errno));

    ::unlink(path_.c_str());

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (path_.size() >= sizeof(addr.sun_path))
      throw std::runtime_error("LineSocketServer: socket path too long");
    std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(listen_fd_,
               reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
      ::close(listen_fd_);
      throw std::runtime_error(
          std::string("LineSocketServer bind(): ") + strerror(errno));
    }
    if (::listen(listen_fd_, 8) < 0) {
      ::close(listen_fd_);
      throw std::runtime_error(
          std::string("LineSocketServer listen(): ") + strerror(errno));
    }

    accept_thread_ = std::thread(&LineSocketServer::acceptLoop, this);
  }

  ~LineSocketServer() {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;

    if (accept_thread_.joinable()) accept_thread_.join();

    {
      std::lock_guard<std::mutex> lk(clients_mu_);
      for (int fd : client_fds_) ::close(fd);
      client_fds_.clear();
    }
    ::unlink(path_.c_str());
  }

  LineSocketServer(const LineSocketServer &) = delete;
  LineSocketServer &operator=(const LineSocketServer &) = delete;

  /// Broadcast a line (appends '\n') to every connected client.
  /// Silently removes clients that have disconnected.
  void broadcast(const std::string &line) {
    std::string msg = line + "\n";
    std::lock_guard<std::mutex> lk(clients_mu_);
    auto it = client_fds_.begin();
    while (it != client_fds_.end()) {
      // Loop until all bytes are written or an error occurs.
      bool ok = true;
      size_t total = 0;
      while (total < msg.size()) {
        ssize_t sent = ::send(*it, msg.data() + total,
                              msg.size() - total, MSG_NOSIGNAL);
        if (sent <= 0) { ok = false; break; }
        total += static_cast<size_t>(sent);
      }
      if (!ok) {
        ::close(*it);
        it = client_fds_.erase(it);
      } else {
        ++it;
      }
    }
  }

 private:
  void acceptLoop() {
    while (!g_shutdown.load(std::memory_order_relaxed)) {
      int fd = ::accept4(listen_fd_, nullptr, nullptr,
                         SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (fd >= 0) {
        // Switch accepted fd to blocking for send(); reads are not performed.
        int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0) {
          ::close(fd);
          continue;
        }
        std::lock_guard<std::mutex> lk(clients_mu_);
        client_fds_.push_back(fd);
      } else {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          struct pollfd pfd = {listen_fd_, POLLIN, 0};
          ::poll(&pfd, 1, 100 /*ms*/);
        } else if (errno == EINTR) {
          continue;
        } else {
          break;  // listen_fd_ closed or unrecoverable error
        }
      }
    }
  }

  std::string      path_;
  int              listen_fd_ = -1;
  std::mutex       clients_mu_;
  std::vector<int> client_fds_;
  std::thread      accept_thread_;
};

// ─────────────────────────────────────────────────────────────────────────────
// TextSocketServer
//
// Accepts connections on a Unix-domain SOCK_STREAM socket, reads
// '\n'-delimited lines from each client, and pushes them onto the synthesis
// queue.  Each client is serviced on a tracked thread that is joined
// in the destructor to prevent use-after-free on the queue reference.
// ─────────────────────────────────────────────────────────────────────────────

class TextSocketServer {
 public:
  TextSocketServer(const std::string &path,
                   ThreadSafeQueue<std::string> &queue)
      : path_(path), queue_(queue) {
    listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listen_fd_ < 0)
      throw std::runtime_error(
          std::string("TextSocketServer socket(): ") + strerror(errno));

    ::unlink(path_.c_str());

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (path_.size() >= sizeof(addr.sun_path))
      throw std::runtime_error("TextSocketServer: socket path too long");
    std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(listen_fd_,
               reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
      ::close(listen_fd_);
      throw std::runtime_error(
          std::string("TextSocketServer bind(): ") + strerror(errno));
    }
    if (::listen(listen_fd_, 8) < 0) {
      ::close(listen_fd_);
      throw std::runtime_error(
          std::string("TextSocketServer listen(): ") + strerror(errno));
    }

    listen_thread_ = std::thread(&TextSocketServer::listenLoop, this);
  }

  ~TextSocketServer() {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
    if (listen_thread_.joinable()) listen_thread_.join();
    // g_shutdown is already set before destruction reaches here;
    // client threads will exit their poll loop within 200 ms.
    // Join them explicitly to prevent use-after-free on queue_.
    {
      std::lock_guard<std::mutex> lk(client_threads_mu_);
      for (auto &t : client_threads_)
        if (t.joinable()) t.join();
      client_threads_.clear();
    }
    ::unlink(path_.c_str());
  }

  TextSocketServer(const TextSocketServer &) = delete;
  TextSocketServer &operator=(const TextSocketServer &) = delete;

 private:
  void listenLoop() {
    while (!g_shutdown.load(std::memory_order_relaxed)) {
      struct pollfd pfd = {listen_fd_, POLLIN, 0};
      int ret = ::poll(&pfd, 1, 200 /*ms*/);
      if (ret <= 0) continue;

      int client_fd =
          ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
      if (client_fd < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
          continue;
        break;  // listen_fd_ was closed
      }

      {
        std::lock_guard<std::mutex> lk(client_threads_mu_);
        client_threads_.emplace_back(
            [this, client_fd] { clientLoop(client_fd); });
      }
    }
  }

  void clientLoop(int fd) {
    std::string buf;
    char tmp[4096];
    while (!g_shutdown.load(std::memory_order_relaxed)) {
      struct pollfd pfd = {fd, POLLIN, 0};
      int ret = ::poll(&pfd, 1, 200 /*ms*/);
      if (ret < 0) break;
      if (ret == 0) continue;

      ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
      if (n <= 0) break;  // EOF or error

      buf.append(tmp, static_cast<size_t>(n));
      std::string::size_type pos;
      while ((pos = buf.find('\n')) != std::string::npos) {
        std::string line = buf.substr(0, pos);
        buf.erase(0, pos + 1);
        if (!line.empty()) queue_.push(std::move(line));
      }
    }
    ::close(fd);
  }

  std::string                    path_;
  ThreadSafeQueue<std::string>  &queue_;
  int                            listen_fd_ = -1;
  std::mutex                     client_threads_mu_;
  std::vector<std::thread>       client_threads_;
  std::thread                    listen_thread_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SynthesisWorker
//
// Pulls text lines from the queue, synthesises audio with Moonshine TTS,
// writes PCM into the AudioRing, and broadcasts the IPA string to the
// phoneme socket.
// ─────────────────────────────────────────────────────────────────────────────

class SynthesisWorker {
 public:
  SynthesisWorker(moonshine::TextToSpeech &tts,
                  moonshine::GraphemeToPhonemizer &g2p,
                  ThreadSafeQueue<std::string> &text_queue,
                  AudioRing &ring,
                  LineSocketServer &phoneme_server)
      : tts_(tts),
        g2p_(g2p),
        text_queue_(text_queue),
        ring_(ring),
        phoneme_server_(phoneme_server) {
    thread_ = std::thread(&SynthesisWorker::run, this);
  }

  ~SynthesisWorker() {
    text_queue_.notifyShutdown();
    if (thread_.joinable()) thread_.join();
  }

  SynthesisWorker(const SynthesisWorker &) = delete;
  SynthesisWorker &operator=(const SynthesisWorker &) = delete;

 private:
  void run() {
    while (true) {
      std::string text;
      if (!text_queue_.pop(text)) break;  // shutdown

      // Convert to IPA and broadcast phonemes.
      try {
        std::string ipa = g2p_.toIpa(text);
        std::cout << "[g2p] " << ipa << "\n" << std::flush;
        phoneme_server_.broadcast(ipa);
      } catch (const moonshine::MoonshineException &e) {
        std::cerr << "[g2p] error: " << e.what() << "\n";
      }

      // Synthesise and push audio.
      try {
        moonshine::TtsSynthesisResult result = tts_.synthesize(text);
        std::cout << "[tts] " << result.samples.size() << " samples @ "
                  << result.sampleRateHz << " Hz\n" << std::flush;
        ring_.write(result.samples.data(),
                    static_cast<uint32_t>(result.samples.size()));
      } catch (const moonshine::MoonshineException &e) {
        std::cerr << "[tts] error: " << e.what() << "\n";
      }
    }
  }

  moonshine::TextToSpeech        &tts_;
  moonshine::GraphemeToPhonemizer &g2p_;
  ThreadSafeQueue<std::string>   &text_queue_;
  AudioRing                      &ring_;
  LineSocketServer               &phoneme_server_;
  std::thread                     thread_;
};

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

static void printUsage(const char *prog) {
  std::cerr
      << "Usage: " << prog << " [OPTIONS]\n"
      << "\n"
      << "Options:\n"
      << "  -r, --asset-root PATH    Path to Moonshine TTS data directory\n"
      << "                           (default: ../../core/moonshine-tts/data)\n"
      << "  -l, --language LANG      Language tag (default: en_us)\n"
      << "  -v, --voice VOICE        Voice name (default: engine default)\n"
      << "      --text-sock PATH     Unix socket for text input\n"
      << "                           (default: /tmp/moonshine-tts.sock)\n"
      << "      --phoneme-sock PATH  Unix socket for IPA output\n"
      << "                           (default: /tmp/moonshine-phonemes.sock)\n"
      << "  -h, --help               Show this help\n"
      << "\n"
      << "Send newline-terminated text lines to --text-sock to synthesise\n"
      << "and play them through PipeWire.  The IPA transcription of each\n"
      << "utterance is broadcast to every client of --phoneme-sock.\n";
}

int main(int argc, char *argv[]) {
  std::string asset_root       = "../../core/moonshine-tts/data";
  std::string language         = "en_us";
  std::string voice;
  std::string text_sock_path    = "/tmp/moonshine-tts.sock";
  std::string phoneme_sock_path = "/tmp/moonshine-phonemes.sock";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-r" || arg == "--asset-root") {
      if (i + 1 >= argc) { printUsage(argv[0]); return 1; }
      asset_root = argv[++i];
    } else if (arg == "-l" || arg == "--language") {
      if (i + 1 >= argc) { printUsage(argv[0]); return 1; }
      language = argv[++i];
    } else if (arg == "-v" || arg == "--voice") {
      if (i + 1 >= argc) { printUsage(argv[0]); return 1; }
      voice = argv[++i];
    } else if (arg == "--text-sock") {
      if (i + 1 >= argc) { printUsage(argv[0]); return 1; }
      text_sock_path = argv[++i];
    } else if (arg == "--phoneme-sock") {
      if (i + 1 >= argc) { printUsage(argv[0]); return 1; }
      phoneme_sock_path = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  // Install signal handlers before creating any threads.
  {
    struct sigaction sa = {};
    sa.sa_handler = handleSignal;
    sigemptyset(&sa.sa_mask);
    ::sigaction(SIGINT,  &sa, nullptr);
    ::sigaction(SIGTERM, &sa, nullptr);
  }
  // Broken-pipe from socket writes must not kill the process.
  ::signal(SIGPIPE, SIG_IGN);

  // PipeWire is a process-global library; init/deinit must be called
  // exactly once and must outlive all pw_* objects.
  struct PwInitGuard {
    PwInitGuard()  { pw_init(nullptr, nullptr); }
    ~PwInitGuard() { pw_deinit(); }
  } pw_guard;

  try {
    std::vector<moonshine_option_t> tts_options = {
        {"g2p_root", asset_root.c_str()},
    };
    if (!voice.empty()) tts_options.push_back({"voice", voice.c_str()});

    std::cout << "Loading TTS synthesizer (language='" << language << "') ...\n";
    moonshine::TextToSpeech tts(language, tts_options);

    std::cout << "Loading G2P phonemizer ...\n";
    moonshine::GraphemeToPhonemizer g2p(
        language, {{"g2p_root", asset_root.c_str()}});

    // Probe the sample rate via a cheap throwaway synthesis call.
    moonshine::TtsSynthesisResult probe = tts.synthesize(".");
    const uint32_t sample_rate =
        static_cast<uint32_t>(probe.sampleRateHz > 0 ? probe.sampleRateHz : 22050);
    const uint32_t channels = 1;

    std::cout << "Sample rate: " << sample_rate << " Hz\n";

    // Shared ring buffer.
    AudioRing ring;

    // Seed ring with probe audio to avoid leading silence.
    if (!probe.samples.empty())
      ring.write(probe.samples.data(),
                 static_cast<uint32_t>(probe.samples.size()));

    // Open PipeWire playback stream.
    std::cout << "Opening PipeWire stream ...\n";
    PipeWireSink pw_sink(ring, sample_rate, channels);

    // Phoneme socket (output to connected clients).
    std::cout << "Phoneme socket: " << phoneme_sock_path << "\n";
    LineSocketServer phoneme_server(phoneme_sock_path);

    // Text input queue and socket.
    ThreadSafeQueue<std::string> text_queue;

    std::cout << "Text socket:    " << text_sock_path << "\n";
    TextSocketServer text_server(text_sock_path, text_queue);

    // Synthesis worker.
    SynthesisWorker worker(tts, g2p, text_queue, ring, phoneme_server);

    std::cout << "Ready.  Send text lines to " << text_sock_path << "\n";
    std::cout << "Press Ctrl-C to quit.\n";

    // Main thread: sleep until SIGINT / SIGTERM.
    while (!g_shutdown.load(std::memory_order_relaxed)) {
      struct timespec ts = {0, 50000000};  // 50 ms
      nanosleep(&ts, nullptr);
    }

    std::cout << "\nShutting down ...\n";

    // Objects are destroyed in reverse declaration order:
    //   worker          -> notifies queue; joins synthesis thread
    //   text_server     -> closes listen fd; client threads drain naturally
    //   phoneme_server  -> closes all client fds; joins accept thread
    //   pw_sink         -> quits pw_main_loop; joins loop thread; destroys stream
    //   ring            -> trivially destroyed
    //   g2p, tts        -> Moonshine handles freed via RAII
  } catch (const moonshine::MoonshineException &e) {
    std::cerr << "Moonshine error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << "\n";
    return 1;
  }

  std::cout << "Done.\n";
  return 0;
}

#else  // !__linux__

#include <cstdio>
int main() {
  std::fprintf(stderr,
               "moonshine-tts-streaming-cli is Linux-only "
               "(requires PipeWire and Unix domain sockets).\n");
  return 1;
}

#endif  // __linux__

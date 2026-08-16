#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <unistd.h>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <pipewire/pipewire.h>
#include <pipewire/stream.h>

#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>

#include "moonshine-g2p.h"
#include "moonshine-tts.h"

static constexpr const char* PHONEME_FIFO =
    "/tmp/moonshine_phonemes";

static constexpr const char* TEXT_FIFO =
    "/tmp/moonshine_text";

struct AudioData {
    pw_stream* stream = nullptr;

    std::vector<float> samples;
    size_t position = 0;

    static constexpr uint32_t sample_rate = 24000;
};


static void on_process(void* userdata) {
    auto* data = static_cast<AudioData*>(userdata);

    pw_buffer* pw_buf =
        pw_stream_dequeue_buffer(data->stream);

    if (pw_buf == nullptr) {
        return;
    }

    spa_buffer* buffer = pw_buf->buffer;

    if (buffer == nullptr || buffer->n_datas < 1) {
        pw_stream_queue_buffer(data->stream, pw_buf);
        return;
    }

    spa_data* audio_data = &buffer->datas[0];

    if (audio_data->data == nullptr ||
        audio_data->chunk == nullptr) {
        pw_stream_queue_buffer(data->stream, pw_buf);
        return;
    }

    auto* output =
        static_cast<float*>(audio_data->data);

    uint32_t max_samples =
        audio_data->maxsize / sizeof(float);

    uint32_t requested =
        static_cast<uint32_t>(pw_buf->requested);

    uint32_t sample_count = max_samples;

    if (requested > 0 &&
        requested < sample_count) {
        sample_count = requested;
    }

    /*
     * Copy synthesized Moonshine audio into PipeWire buffer.
     */
    uint32_t written = 0;

    while (written < sample_count) {

        if (data->position >= data->samples.size()) {
            /*
             * Speech has finished.
             *
             * Fill the remaining buffer with silence.
             */
            output[written++] = 0.0f;
            continue;
        }

        output[written++] =
            data->samples[data->position++];
    }

    audio_data->chunk->offset = 0;

    audio_data->chunk->size =
        sample_count * sizeof(float);

    audio_data->chunk->stride =
        sizeof(float);

    pw_buf->size = sample_count;

    pw_stream_queue_buffer(data->stream, pw_buf);
}


static void on_state_changed(
    void*,
    enum pw_stream_state,
    enum pw_stream_state state,
    const char* error) {

    std::cout
        << "PipeWire stream state: "
        << pw_stream_state_as_string(state)
        << "\n";

    if (error != nullptr) {
        std::cerr
            << "PipeWire error: "
            << error
            << "\n";
    }
}


int main(int argc, char* argv[]) {

    /*
     * ---------------------------------------------------------
     * 1. Moonshine CLI-style argument handling
     * ---------------------------------------------------------
     */

    using moonshine_tts::MoonshineTTS;
    using moonshine_tts::MoonshineTTSOptions;

    std::vector<std::pair<std::string, std::string>> pairs;
    std::vector<std::string> positionals;
    std::string text_flag;

    for (int i = 1; i < argc;) {

        const std::string a = argv[i];

        if (a == "-h" || a == "--help") {
            std::cout
                << "Usage: " << argv[0]
                << " [--lang LANG] [--voice ID] [--speed N] "
                   "[--text \"...\"] [TEXT...]\n";

            return 0;
        }

        if (a == "--text" && i + 1 < argc) {
            text_flag = argv[i + 1];
            i += 2;
            continue;
        }

        if (a.rfind("--", 0) == 0) {

            const std::string key =
                a.substr(2);

            if (i + 1 >= argc) {
                std::cerr
                    << "Missing value for --"
                    << key
                    << "\n";

                return 2;
            }

            pairs.emplace_back(
                key,
                argv[i + 1]);

            i += 2;
            continue;
        }

        positionals.push_back(a);
        ++i;
    }


    /*
     * ---------------------------------------------------------
     * 2. Build text exactly like moonshine-tts CLI
     * ---------------------------------------------------------
     */

    std::string text = text_flag;

    if (text.empty()) {

        for (const auto& p : positionals) {

            if (!text.empty()) {
                text += ' ';
            }

            text += p;
        }
    }

    if (text.empty()) {

    /*
     * No --text / positional argument was supplied.
     * Read the text from the named FIFO.
     */

    if (mkfifo(TEXT_FIFO, 0666) < 0) {

        if (errno != EEXIST) {

            std::cerr
                << "Failed to create text FIFO: "
                << TEXT_FIFO
                << "\n";

            return 1;
        }
    }

    std::cout
        << "Waiting for text on "
        << TEXT_FIFO
        << "...\n";

    int text_fd =
        open(TEXT_FIFO, O_RDONLY);

    if (text_fd < 0) {

        std::cerr
            << "Failed to open text FIFO\n";

        return 1;
    }

    char buffer[4096];

    ssize_t bytes_read =
        read(
            text_fd,
            buffer,
            sizeof(buffer) - 1);

    close(text_fd);

    if (bytes_read <= 0) {

        std::cerr
            << "No text received from FIFO\n";

        return 1;
    }

    buffer[bytes_read] = '\0';

    text = buffer;

    /*
     * Remove trailing newline(s) from echo/cat input.
     */
     while (!text.empty() &&
           (text.back() == '\n' ||
            text.back() == '\r')) {

        text.pop_back();
      }

      if (text.empty()) {

        std::cerr
            << "Received empty text\n";

        return 1;
      }
    }


    /*
     * ---------------------------------------------------------
     * 3. Create Moonshine TTS
     * ---------------------------------------------------------
     */

    MoonshineTTSOptions opt;

    std::string lang = "en_us";
    bool lang_set = false;

    try {

        opt.parse_options(
            pairs,
            &lang,
            &lang_set);

    } catch (const std::exception& e) {

        std::cerr
            << "TTS option error: "
            << e.what()
            << "\n";

        return 2;
    }


    /*
     * ---------------------------------------------------------
     * 4. SYNTHESIZE BEFORE starting PipeWire
     * ---------------------------------------------------------
     */

    std::cout
        << "Synthesizing: \""
        << text
        << "\"\n";

    std::vector<float> wav;

    try {

        MoonshineTTS tts(
            lang,
            opt);

        wav =
            tts.synthesize(text);

    } catch (const std::exception& e) {

        std::cerr
            << "TTS synthesis error: "
            << e.what()
            << "\n";

        return 1;
    }

    try {

    moonshine_tts::MoonshineG2POptions g2p_opt;
	g2p_opt.g2p_root = "core/moonshine-tts/data";

	moonshine_tts::MoonshineG2P g2p(lang, g2p_opt);

    std::string phonemes =
        g2p.text_to_ipa(text);

    std::cout
        << "Phonemes: "
        << phonemes
        << "\n";

    /*
     * Create the phoneme FIFO if it does not exist.
     */
    if (mkfifo(PHONEME_FIFO, 0666) < 0) {
       if (errno != EEXIST) {
         std::cerr
         << "Failed to create phoneme FIFO: "
         << PHONEME_FIFO
         << "\n";

        return 1;}
}

    /*
     * Open FIFO for writing.
     *
     * This will wait until a reader connects.
     */
    int phoneme_fd =
        open(PHONEME_FIFO, O_WRONLY | O_NONBLOCK);

    if (phoneme_fd < 0) {

    if (errno == ENXIO) {
        std::cerr
            << "No phoneme FIFO reader connected; "
               "continuing without phoneme IPC.\n";
    } else {
        std::cerr
            << "Failed to open phoneme FIFO for writing: "
            << std::strerror(errno)
            << "\n";
    }

  } else {

      std::string output =
          phonemes + "\n";

    ssize_t written =
        write(
            phoneme_fd,
            output.data(),
            output.size());

    if (written < 0) {
        if (errno == EPIPE) {
            std::cerr
                << "Phoneme FIFO reader disconnected; "
                   "continuing.\n";
        } else {
            std::cerr
                << "Failed to write phonemes: "
                << std::strerror(errno)
                << "\n";
        }
    }

    close(phoneme_fd);
  }

    } catch (const std::exception& e) {

        std::cerr
        << "G2P error: "
        << e.what()
        << "\n";

        return 1;
    }

    if (wav.empty()) {

        std::cerr
            << "Error: empty waveform\n";

        return 1;
    }

    std::cout
        << "Generated "
        << wav.size()
        << " samples at "
        << MoonshineTTS::kSampleRateHz
        << " Hz\n";

    /*
     * ---------------------------------------------------------
     * 5. Put synthesized audio into PipeWire userdata
     * ---------------------------------------------------------
     */

    AudioData audio;

    audio.samples =
        std::move(wav);

    audio.position = 0;


    /*
     * ---------------------------------------------------------
     * 6. Initialize PipeWire
     * ---------------------------------------------------------
     */

    pw_init(
        &argc,
        &argv);

    std::cout
        << "PipeWire initialized\n";


    pw_thread_loop* loop =
        pw_thread_loop_new(
            "moonshine-tts-stream",
            nullptr);

    if (loop == nullptr) {

        std::cerr
            << "Failed to create PipeWire thread loop\n";

        pw_deinit();

        return 1;
    }


    /*
     * ---------------------------------------------------------
     * 7. PipeWire events
     * ---------------------------------------------------------
     */

    pw_stream_events events{};

    events.version =
        PW_VERSION_STREAM_EVENTS;

    events.state_changed =
        on_state_changed;

    events.process =
        on_process;


    /*
     * ---------------------------------------------------------
     * 8. PipeWire properties
     * ---------------------------------------------------------
     */

    pw_properties* props =
        pw_properties_new(
            PW_KEY_MEDIA_TYPE,
            "Audio",

            PW_KEY_MEDIA_CATEGORY,
            "Playback",

            PW_KEY_MEDIA_ROLE,
            "Music",

            nullptr);


    if (props == nullptr) {

        std::cerr
            << "Failed to create PipeWire properties\n";

        pw_thread_loop_destroy(loop);

        pw_deinit();

        return 1;
    }


    /*
     * ---------------------------------------------------------
     * 9. Create PipeWire stream
     * ---------------------------------------------------------
     */

    pw_stream* stream =
        pw_stream_new_simple(
            pw_thread_loop_get_loop(loop),

            "Moonshine TTS Stream",

            props,

            &events,

            &audio);


    if (stream == nullptr) {

        std::cerr
            << "Failed to create PipeWire stream\n";

        pw_thread_loop_destroy(loop);

        pw_deinit();

        return 1;
    }


    audio.stream = stream;


    /*
     * ---------------------------------------------------------
     * 10. PipeWire audio format
     *
     * Moonshine = 24000 Hz F32 mono
     * ---------------------------------------------------------
     */

    uint8_t buffer[1024];

    spa_pod_builder builder =
        SPA_POD_BUILDER_INIT(
            buffer,
            sizeof(buffer));


    spa_audio_info_raw audio_format{};

    audio_format.format =
        SPA_AUDIO_FORMAT_F32;

    audio_format.rate =
        MoonshineTTS::kSampleRateHz;

    audio_format.channels =
        1;


    const spa_pod* params[1];

    params[0] =
        spa_format_audio_raw_build(
            &builder,

            SPA_PARAM_EnumFormat,

            &audio_format);


    /*
     * ---------------------------------------------------------
     * 11. Connect PipeWire stream
     * ---------------------------------------------------------
     */

    int res =
        pw_stream_connect(
            stream,

            PW_DIRECTION_OUTPUT,

            PW_ID_ANY,

            static_cast<pw_stream_flags>(
                PW_STREAM_FLAG_AUTOCONNECT |
                PW_STREAM_FLAG_MAP_BUFFERS |
                PW_STREAM_FLAG_RT_PROCESS),

            params,

            1);


    if (res < 0) {

        std::cerr
            << "Failed to connect PipeWire stream: "
            << res
            << "\n";

        pw_stream_destroy(stream);

        pw_thread_loop_destroy(loop);

        pw_deinit();

        return 1;
    }


    std::cout
        << "Starting Moonshine audio stream...\n";


    /*
     * ---------------------------------------------------------
     * 12. Start PipeWire
     * ---------------------------------------------------------
     */

    res =
        pw_thread_loop_start(loop);


    if (res < 0) {

        std::cerr
            << "Failed to start PipeWire thread loop: "
            << res
            << "\n";

        pw_stream_destroy(stream);

        pw_thread_loop_destroy(loop);

        pw_deinit();

        return 1;
    }


    std::cout
        << "Streaming synthesized speech.\n";


    /*
     * ---------------------------------------------------------
     * 13. Keep process alive
     * ---------------------------------------------------------
     */

    while (
        audio.position <
        audio.samples.size()) {

        sleep(1);
    }
     std::cout
        << "Speech finished.\n";



    /*
     * ---------------------------------------------------------
     * 14. Cleanly shut down PipeWire
     * ---------------------------------------------------------
     */

    pw_thread_loop_stop(loop);

    std::cout
        << "PipeWire stopped.\n";

    pw_stream_destroy(stream);

    std::cout
        << "PipeWire stream destroyed.\n";

    pw_thread_loop_destroy(loop);

    std::cout
        << "PipeWire loop destroyed.\n";

    pw_deinit();

    std::cout
        << "PipeWire deinitialized.\n";

    return 0;
}


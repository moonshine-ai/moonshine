# Moonshine Streaming TTS over PipeWire (Linux)

A streaming, long-running version of
[`examples/c++/text-to-speech.cpp`](../../c++/text-to-speech.cpp). Instead of
synthesizing one string and writing a `.wav`, it stays running and accepts
an arbitrary number of utterances over sockets, playing audio live through
PipeWire instead of writing a file.

This example is Linux-only, since it depends on PipeWire.

## Features

- Accepts UTF-8 text over a Unix domain socket, one utterance per line
- Streams synthesized PCM audio directly to a PipeWire playback sink
- Broadcasts IPA phonemes for each utterance to all connected clients over
  a second Unix domain socket
- Pipelines utterances: synthesis of the next utterance can proceed while
  the current one is still playing

## Requirements

- Linux (x86_64)
- `pkg-config` and PipeWire dev headers:
  ```bash
  sudo apt-get install libpipewire-0.3-dev pkg-config
  ```
- A running PipeWire session (standard on most current desktop Linux
  distros; check with `pw-cli info`)

## Build

```bash
./build.sh
```

This downloads the prebuilt `moonshine-voice-linux-x86_64` library (skipped
if already present in this directory) and compiles with:

```bash
g++ -std=c++17 -O2 moonshine-tts-streaming-cli.cpp \
  -Imoonshine-voice-linux-x86_64/include \
  -Lmoonshine-voice-linux-x86_64/lib \
  $(pkg-config --cflags libpipewire-0.3) \
  -lmoonshine \
  $(pkg-config --libs libpipewire-0.3) \
  -lpthread \
  -o moonshine-tts-streaming-cli

export LD_LIBRARY_PATH=$(pwd)/moonshine-voice-linux-x86_64/lib
```

## Run

```bash
./moonshine-tts-streaming-cli -r ../../../core/moonshine-tts/data -l en_us
```

| Flag | Default | Meaning |
|---|---|---|
| `-r`, `--asset-root` | `../../../core/moonshine-tts/data` | Path to Moonshine asset data |
| `-l`, `--language` | `en_us` | Language tag |
| `-v`, `--voice` | (engine default) | Voice name |
| `--text-sock` | `/tmp/moonshine-tts.sock` | Socket path for text input |
| `--phoneme-sock` | `/tmp/moonshine-phonemes.sock` | Socket path for phoneme output |

## Socket usage

Once the tool prints `Ready.`:

```bash
# terminal 1: listen for phonemes
nc -U /tmp/moonshine-phonemes.sock

# terminal 2: send text to synthesize
echo "Hello, this is a streaming test." | nc -U /tmp/moonshine-tts.sock
```

Text socket: one UTF-8 line per utterance, newline-terminated. Multiple
clients can connect and send text concurrently.

Phoneme socket: one line per utterance, containing just the IPA string
(no utterance ID). Broadcast to every currently connected client — any
number of readers can attach at once.

Audio plays to your default sink automatically. To check the node and
capture what it's emitting:

```bash
pw-cli ls Node | grep -A2 moonshine-tts-streaming
pw-record --target moonshine-tts-streaming test.wav   # Ctrl+C after a few seconds
```

`Ctrl+C` shuts the process down gracefully.

## Implementation notes

`moonshine::TextToSpeech::synthesize()` is a single blocking call that
returns a complete utterance's samples at once — the library has no
incremental synthesis API. So streaming here happens at the utterance
boundary, not within a single utterance's audio generation.

In practice this means: phonemes for an utterance are sent as soon as
`toIpa()` returns, ahead of audio; and synthesis of the next utterance can
start while the current one is still draining out to PipeWire. That
overlap is what makes this faster than calling the batch tool repeatedly,
even though no single `synthesize()` call itself streams.

See comments in `moonshine-tts-streaming-cli.cpp` for the rest of the
implementation detail (socket design, ring buffer, PipeWire callback
constraints, thread ownership).

## Limitations

- No mid-synthesis audio streaming (see above) — this is a property of the
  current TTS API, not something this example works around.
- Phoneme messages carry no utterance ID, so a client can't correlate a
  given phoneme line back to a specific request if it sent several in
  quick succession.
- Ring buffer backpressure blocks the synthesis thread rather than
  dropping audio, so a burst of very long utterances queued faster than
  they can play will delay phoneme delivery for later utterances too.
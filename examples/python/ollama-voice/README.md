# Ollama Voice Example

Example of using the **Moonshine Voice** library to transcribe speech from your microphone and send it to an [Ollama](https://ollama.com/) LLM chat interface. As you speak, your words are transcribed in real time; when you finish a segment, it is sent to the selected Ollama model and the streamed reply is printed to the console.

## Prerequisites

- **Ollama** installed and running locally (e.g. `ollama serve`), with at least one model pulled (e.g. `ollama pull qwen3.5:9b`)
- **Python 3** with the Moonshine Voice package and its dependencies
- A working microphone

## Installation

Install the Moonshine Voice Python package (if you haven’t already):

```bash
pip install moonshine-voice
```

Install the Ollama Python client:

```bash
pip install ollama
```

## Usage

Start Ollama (if not already running):

```bash
ollama serve
```

Run the example in another terminal:

```bash
python ollama_voice.py
```

Speak into the microphone. When a segment is finalized, it is sent to the default Ollama model and the response is streamed to the terminal. Press **Ctrl+C** to stop.

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--ollama-model` | `qwen3.5:9b` | Ollama model name for chat responses |
| `--language` | `en` | Language for Moonshine transcription |
| `--moonshine-model-arch` | (auto) | Moonshine model architecture to use |

Examples:

```bash
python ollama_voice.py --ollama-model llama3.2
python ollama_voice.py --language en --ollama-model qwen3.5:9b
```

## How it works

- **Moonshine Voice** (`MicTranscriber`) captures audio from the default microphone and runs on-device speech-to-text. It emits:
  - **Live updates** (`on_line_text_changed`) for the current phrase, shown on one line with `\r`.
  - **Finalized segments** (`on_line_completed`) when a phrase is done.
- The example implements `TranscriptEventListener`. On each finalized segment it:
  1. Appends the text as a user message to the conversation.
  2. Calls Ollama’s chat API with the current message history.
  3. Streams the model’s reply to stdout.

No audio or transcript is sent to Moonshine servers; only the finalized text is sent to your local Ollama instance.

"""Example of using the Moonshine Voice library to transcribe speech and send it to an Ollama LLM chat interface."""

import argparse
import time

import numpy as np
import ollama
import sounddevice as sd
from moonshine_voice import (
    MicTranscriber,
    TextToSpeech,
    TranscriptEventListener,
    get_model_for_language,
    get_tts_model,
)


class Spinner:
    """Simple spinner class that prints a rotating character to the console."""

    def __init__(self):
        self.frames = ["-", "\\", "|", "/"]

    def spin(self):
        index = int(time.time() * 4) % len(self.frames)
        print(self.frames[index], end="\r", flush=True)


class OllamaVoice(TranscriptEventListener):
    """
    Transcript event listener that links speech transcription to Ollama chat.

    As the microphone transcriber detects new speech, this class handles intermediate
    transcript updates for live feedback and sends finalized transcript segments to
    an Ollama LLM chat interface, outputting its streamed completion.
    """

    def __init__(
        self,
        ollama_model: str,
        system_prompt: str,
        tts: TextToSpeech | None = None,
        output_device: int | None = None,
    ):
        """
        Initialize the OllamaVoice listener.

        Args:
            ollama_model (str): The Ollama model name to use for chat responses.
            tts: An optional TextToSpeech instance for speaking responses aloud.
            output_device: The sounddevice output device index for playback.
        """
        self.tts = tts
        self.output_device = output_device
        # Which Ollama model to use for chat responses.
        self.ollama_model = ollama_model
        # Download the Ollama model if it's not already on-disk.
        listed = ollama.list()
        model_names = [m.model for m in (listed.models or [])]
        if self.ollama_model not in model_names:
            print(f"Downloading Ollama model '{self.ollama_model}'...")
            ollama.pull(self.ollama_model)
        # Preload the Ollama model to minimize latency on first use.
        print(f"Preloading Ollama model {self.ollama_model}...")
        ollama.generate(model=self.ollama_model, prompt="", keep_alive="10m")
        # Whether Ollama is currently processing a chat request.
        self.is_ollama_running = False
        # The system prompt to use for the chat context.
        self.system_prompt_message = {"role": "system", "content": system_prompt}
        # Accumulates messages forming the conversation context.
        self.messages = [self.system_prompt_message]

    def on_line_text_changed(self, event):
        """
        Called whenever the transcription of the current line changes (live updates).

        Args:
            event: Transcript event containing the current line state.
        """
        # Print live transcript with carriage return for inline feedback
        print(event.line.text, end="\r", flush=True)

    def on_line_completed(self, event):
        """
        Called when a line (segment) of speech has been fully transcribed.

        Args:
            event: Transcript event with the finalized line.
        """
        # Print the completed transcription segment
        print(event.line.text)
        # Add the completed utterance as a user message for the chat context
        self.messages.append({"role": "user", "content": event.line.text})

        # Prevent overlapping chat completions
        if not self.is_ollama_running:
            self.is_ollama_running = True
            # Use a copy of the current conversation to send to the model
            messages = self.messages.copy()
            # Clear out the messages accumulator for the next user turn
            self.messages = [self.system_prompt_message]
            # Request a streamed response from Ollama's chat API
            stream = ollama.chat(
                model=self.ollama_model, messages=messages, stream=True
            )
            spinner = Spinner()
            has_content = False
            full_response = ""
            for chunk in stream:
                # Extract and print each chunk of model output as it streams in
                content = chunk["message"]["content"]
                # We get a stream of empty chunks when the model is thinking.
                if len(content) == 0:
                    if not has_content:
                        spinner.spin()
                else:
                    has_content = True
                    full_response += content
                    print(content, end="", flush=True)
            # Print newline to finish the response segment cleanly
            print("\n", end="", flush=True)

            # Speak the response aloud via TTS.
            if self.tts is not None and full_response.strip():
                try:
                    result = self.tts.generate(full_response.strip())
                    audio_float = result.audio_data.astype(np.float32) / 32768.0
                    sd.play(
                        audio_float,
                        samplerate=result.sample_rate,
                        device=self.output_device,
                    )
                    sd.wait()
                except Exception as e:
                    print(f"[TTS error: {e}]")

            self.is_ollama_running = False


if __name__ == "__main__":
    # Set up the main components for Ollama voice interaction:
    # - Parse command-line arguments for the Ollama and Moonshine model configurations.
    # - Determine model path/architecture for transcription based on language settings.
    # - Instantiate microphone transcriber and Ollama voice response handler.
    # - Attach the response handler as a listener for transcriber events.
    # - Start microphone transcription and handle user interaction loop.

    parser = argparse.ArgumentParser(description="Ollama voice example")
    parser.add_argument(
        "--ollama-model", type=str, default="gemma3:4b", help="Ollama model to use"
    )
    parser.add_argument("--language", type=str, default="en", help="Language to use")
    parser.add_argument(
        "--moonshine-model-arch",
        type=int,
        default=None,
        help="Moonshine model architecture to use",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant who is receiving requests from a user talking to them, and responding. Responses should be short and conversational.",
        help="System prompt to use",
    )
    parser.add_argument(
        "--tts-model-name",
        type=str,
        default="tsuki-max-en",
        help="Name of the Moonshine TTS model for spoken responses (default: tsuki-max-en). "
        "Set to 'none' to disable TTS.",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Audio output device index for TTS playback (see `python -m sounddevice`)",
    )
    args = parser.parse_args()

    model_path, model_arch = get_model_for_language(
        args.language, args.moonshine_model_arch
    )

    # Initialize TTS if requested.
    tts = None
    if args.tts_model_name.lower() != "none":
        print(f"Loading TTS model '{args.tts_model_name}'...")
        tts_model_path, tts_model_arch = get_tts_model(args.tts_model_name)
        tts = TextToSpeech(tts_model_path, tts_model_arch)

    mic_transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

    # Attach the Ollama voice response handler as a listener for transcriber events.
    mic_transcriber.add_listener(
        OllamaVoice(
            args.ollama_model,
            args.system_prompt,
            tts=tts,
            output_device=args.output_device,
        )
    )

    mic_transcriber.start()

    print("Listening to the microphone, press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.1)

    mic_transcriber.stop()
    if tts is not None:
        tts.close()

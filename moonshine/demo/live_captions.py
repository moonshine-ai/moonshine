"""Live captions from microphone using Moonshine and SileroVAD ONNX models."""
import os
import sys
import time

from queue import Queue

import numpy as np

from silero_vad import load_silero_vad, VADIterator
from sounddevice import InputStream

# Local import.
from transcriber_moonshine_onnx import TranscriberMoonshine

SAMPLING_RATE = 16000

CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
LOOKBACK_CHUNKS = 5
MARKER_LENGTH = 6
MAX_LINE_LENGTH = 80
SHOW_NEW_CAPTION = False  # Stacks cached captions above scrolling captions.

# These affect live caption updating - adjust for your platform speed and model.
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.2

VERBOSE = False

def create_source_callback(q):
    def source_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy(), status))
    return source_callback


def end_recording(speech, marker=""):
    """Transcribes, caches and prints the caption.  Clears speech buffer."""
    if len(marker) != MARKER_LENGTH:
        raise ValueError("Unexpected marker length.")
    text = transcribe(speech)
    caption_cache.append(text + " " + marker)
    print_captions(text + (" " + marker) if VERBOSE else "", True)
    speech *= 0.0


def print_captions(text, new_cached_caption=False):
    """Prints right justified on same line, prepending cached captions."""
    print('\r' + " " * MAX_LINE_LENGTH, end='', flush=True)
    if SHOW_NEW_CAPTION and new_cached_caption:
        print('\r', end='', flush=True)
        print(caption_cache[-1][:-MARKER_LENGTH])
    if len(text) > MAX_LINE_LENGTH:
        text = text[-MAX_LINE_LENGTH:]
    elif text != "\n":
        for caption in caption_cache[::-1]:
            text = (caption[:-MARKER_LENGTH] if not VERBOSE else
                    caption + " ") + text
            if len(text) > MAX_LINE_LENGTH:
                break
        if len(text) > MAX_LINE_LENGTH:
            text = text[-MAX_LINE_LENGTH:]
    text = " " * (MAX_LINE_LENGTH - len(text)) + text
    print('\r' + text, end='', flush=True)


if __name__ == '__main__':
    model_size = "base" if len(sys.argv) < 2 else sys.argv[1]
    if model_size not in ["base", "tiny"]:
        raise ValueError("Model size is not supported.")

    models_dir = os.path.join(os.path.dirname(__file__), 'models', f"{model_size}")
    print(f"Loading Moonshine model '{models_dir}' ...")
    transcribe = TranscriberMoonshine(models_dir=models_dir, rate=SAMPLING_RATE)

    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=300,
    )

    q = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_source_callback(q),
    )
    stream.start()

    caption_cache = []
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)

    recording = False

    print("Press Ctrl+C to quit live captions.\n")

    with stream:
        print_captions("Ready...")
        try:
            while True:
                data, status = q.get()
                if VERBOSE and status:
                    print(status)

                chunk = np.array(data).flatten()

                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if 'start' in speech_dict and not recording:
                        recording = True
                        start_time = time.time()

                    if 'end' in speech_dict and recording:
                        recording = False
                        end_recording(speech, "<STOP>")

                elif recording:
                    # Possible speech truncation can cause hallucination.

                    if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        recording = False
                        end_recording(speech, "<SNIP>")
                        # Soft reset without affecting VAD model state.
                        vad_iterator.triggered = False
                        vad_iterator.temp_end = 0
                        vad_iterator.current_sample = 0

                    if (time.time() - start_time) > MIN_REFRESH_SECS:
                        print_captions(transcribe(speech))
                        start_time = time.time()

        except KeyboardInterrupt:
            stream.close()
            print(f"""
    number inferences :  {transcribe.number_inferences}
  mean inference time :  {(transcribe.inference_secs / transcribe.number_inferences):.2f}s
model realtime factor :  {(transcribe.speech_secs / transcribe.inference_secs):0.2f}x
    """)
            if caption_cache:
                print("Cached captions.")
                for caption in caption_cache:
                    print(caption[:-MARKER_LENGTH], end="", flush=True)
            print("")

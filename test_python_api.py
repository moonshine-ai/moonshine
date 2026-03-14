#!/usr/bin/env python3
"""Test the Python API word timestamps by calling the C library directly via ctypes."""

import ctypes
import os
import sys
import wave
import numpy as np

# Load the moonshine shared library
lib_path = os.path.join(os.path.dirname(__file__), "core", "build", "libmoonshine.dylib")
if not os.path.exists(lib_path):
    print(f"FAIL: Library not found at {lib_path}")
    sys.exit(1)

lib = ctypes.CDLL(lib_path)

# Define C structs matching moonshine-c-api.h

class TranscriptWordC(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("start", ctypes.c_float),
        ("end", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]

class TranscriptLineC(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("audio_data", ctypes.POINTER(ctypes.c_float)),
        ("audio_data_count", ctypes.c_size_t),
        ("start_time", ctypes.c_float),
        ("duration", ctypes.c_float),
        ("id", ctypes.c_uint64),
        ("is_complete", ctypes.c_int8),
        ("is_updated", ctypes.c_int8),
        ("is_new", ctypes.c_int8),
        ("has_text_changed", ctypes.c_int8),
        ("has_speaker_id", ctypes.c_int8),
        ("speaker_id", ctypes.c_uint64),
        ("speaker_index", ctypes.c_uint32),
        ("last_transcription_latency_ms", ctypes.c_uint32),
        ("words", ctypes.POINTER(TranscriptWordC)),
        ("word_count", ctypes.c_uint64),
    ]

class TranscriptC(ctypes.Structure):
    _fields_ = [
        ("lines", ctypes.POINTER(TranscriptLineC)),
        ("line_count", ctypes.c_uint64),
    ]

class TranscriberOptionC(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
    ]

# Set up function signatures
lib.moonshine_get_version.restype = ctypes.c_int32

lib.moonshine_load_transcriber_from_files.argtypes = [
    ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(TranscriberOptionC), ctypes.c_int32,
    ctypes.c_int32,
]
lib.moonshine_load_transcriber_from_files.restype = ctypes.c_int32

lib.moonshine_transcribe_without_streaming.argtypes = [
    ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint64, ctypes.c_int32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(TranscriptC)),
]
lib.moonshine_transcribe_without_streaming.restype = ctypes.c_int32

lib.moonshine_free_transcriber.argtypes = [ctypes.c_int32]

# Load model
model_path = os.path.join(os.path.dirname(__file__), "test-assets", "tiny-en")
wav_path = os.path.join(os.path.dirname(__file__), "test-assets", "beckett.wav")

options = (TranscriberOptionC * 2)(
    TranscriberOptionC(b"word_timestamps", b"true"),
    TranscriberOptionC(b"identify_speakers", b"false"),
)

version = lib.moonshine_get_version()
print(f"Moonshine version: {version}")
print(f"Loading model from {model_path}...")

handle = lib.moonshine_load_transcriber_from_files(
    model_path.encode(), 0, options, 2, version
)
if handle < 0:
    print(f"FAIL: Load failed with code {handle}")
    sys.exit(1)
print(f"Model loaded (handle={handle})")

# Load audio
with wave.open(wav_path, 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

audio_duration = len(audio) / 16000.0
print(f"\nAudio: {wav_path} ({audio_duration:.2f}s)")

# Transcribe
audio_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
transcript_ptr = ctypes.POINTER(TranscriptC)()

err = lib.moonshine_transcribe_without_streaming(
    handle, audio_ptr, len(audio), 16000, 0, ctypes.byref(transcript_ptr)
)

if err != 0:
    print(f"FAIL: Transcribe failed with code {err}")
    lib.moonshine_free_transcriber(handle)
    sys.exit(1)

transcript = transcript_ptr.contents
print(f"\nTranscript lines: {transcript.line_count}")

total_words = 0
violations = 0

for i in range(transcript.line_count):
    line = transcript.lines[i]
    text = line.text.decode('utf-8') if line.text else ""
    print(f"\nLine {i}: \"{text}\"")
    print(f"  start_time={line.start_time:.3f}, duration={line.duration:.3f}")
    print(f"  word_count={line.word_count}")

    if line.words and line.word_count > 0:
        prev_start = -1.0
        for j in range(line.word_count):
            word = line.words[j]
            word_text = word.text.decode('utf-8') if word.text else ""
            print(f"    [{word.start:7.3f}s - {word.end:7.3f}s] {word_text:15s}  (conf: {word.confidence:.2f})")
            if word.start < prev_start:
                violations += 1
            prev_start = word.start
            total_words += 1

print(f"\n=== Python API Test Results ===")
print(f"Total words: {total_words}")
print(f"Monotonicity violations: {violations}")

lib.moonshine_free_transcriber(handle)

if total_words == 0:
    print("FAIL: No words produced")
    sys.exit(1)
if violations > 0:
    print("FAIL: Monotonicity violations")
    sys.exit(1)

print(f"PASS: {total_words} words with correct ordering via Python ctypes API")

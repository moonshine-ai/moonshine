"""Transcribe Telnyx Media Streaming WebSocket audio with Moonshine Voice."""

import argparse
import array
import asyncio
import base64
import json
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List

from moonshine_voice import (
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
)

try:
    import websockets
except ImportError:
    print(
        "Install the example dependency with: python -m pip install websockets",
        file=sys.stderr,
    )
    raise SystemExit(1)


@dataclass
class MediaFormat:
    encoding: str = "PCMU"
    sample_rate: int = 8000
    channels: int = 1


class TranscriptPrinter(TranscriptEventListener):
    def __init__(self, show_interim: bool):
        self.show_interim = show_interim
        self.last_line_text_length = 0

    def on_line_started(self, event):
        self.last_line_text_length = 0

    def on_line_text_changed(self, event):
        if not self.show_interim:
            return
        self._update_terminal_line(event.line.text)

    def on_line_completed(self, event):
        if self.show_interim:
            self._update_terminal_line(event.line.text)
            print("", flush=True)
        else:
            print(event.line.text, flush=True)

    def _update_terminal_line(self, text: str):
        print(f"\r{text}", end="", flush=True)
        if len(text) < self.last_line_text_length:
            print(" " * (self.last_line_text_length - len(text)), end="", flush=True)
        self.last_line_text_length = len(text)


def decode_l16(payload: bytes) -> List[float]:
    if len(payload) % 2:
        payload = payload[:-1]
    samples = array.array("h")
    samples.frombytes(payload)
    if sys.byteorder == "little":
        samples.byteswap()
    return [sample / 32768.0 for sample in samples]


def decode_pcmu(payload: bytes) -> List[float]:
    return [_linear_to_float(_decode_ulaw_byte(value)) for value in payload]


def decode_pcma(payload: bytes) -> List[float]:
    return [_linear_to_float(_decode_alaw_byte(value)) for value in payload]


def _decode_ulaw_byte(value: int) -> int:
    value = ~value & 0xFF
    sample = ((value & 0x0F) << 3) + 0x84
    sample <<= (value & 0x70) >> 4
    return 0x84 - sample if value & 0x80 else sample - 0x84


def _decode_alaw_byte(value: int) -> int:
    value ^= 0x55
    sign = value & 0x80
    exponent = (value & 0x70) >> 4
    sample = (value & 0x0F) << 4
    if exponent == 0:
        sample += 8
    else:
        sample += 0x108
        sample <<= exponent - 1
    return sample if sign else -sample


def _linear_to_float(sample: int) -> float:
    return max(-1.0, min(1.0, sample / 32768.0))


DECODERS: Dict[str, Callable[[bytes], List[float]]] = {
    "L16": decode_l16,
    "PCMU": decode_pcmu,
    "PCMA": decode_pcma,
}


def track_matches(received: str, selected: str) -> bool:
    if selected == "both":
        return True
    if received.endswith("_track"):
        received = received[: -len("_track")]
    return received == selected


def parse_media_format(start_event: dict) -> MediaFormat:
    media_format = start_event.get("start", {}).get("media_format", {})
    encoding = str(media_format.get("encoding", "PCMU")).upper()
    sample_rate = int(media_format.get("sample_rate", 8000))
    channels = int(media_format.get("channels", 1))
    return MediaFormat(encoding=encoding, sample_rate=sample_rate, channels=channels)


async def handle_telnyx_stream(websocket, args, transcriber: Transcriber):
    media_format = MediaFormat()
    stream = transcriber.create_stream()
    stream.add_listener(TranscriptPrinter(show_interim=not args.final_only))
    stream.start()

    try:
        async for raw_message in websocket:
            message = json.loads(raw_message)
            event_type = message.get("event")

            if event_type == "start":
                media_format = parse_media_format(message)
                print(
                    "Telnyx stream started: "
                    f"encoding={media_format.encoding}, "
                    f"sample_rate={media_format.sample_rate}, "
                    f"channels={media_format.channels}",
                    file=sys.stderr,
                )
                if media_format.channels != 1:
                    print(
                        "Only mono streams are supported by this example",
                        file=sys.stderr,
                    )
                continue

            if event_type == "media":
                media = message.get("media", {})
                if not track_matches(str(media.get("track", "")), args.track):
                    continue
                decoder = DECODERS.get(media_format.encoding)
                if decoder is None:
                    supported = ", ".join(sorted(DECODERS))
                    raise ValueError(
                        f"Unsupported Telnyx stream encoding {media_format.encoding}. "
                        f"Supported encodings: {supported}"
                    )
                payload = base64.b64decode(media["payload"])
                stream.add_audio(decoder(payload), media_format.sample_rate)
                continue

            if event_type == "stop":
                break

            if event_type == "error":
                print(f"Telnyx stream error: {message.get('payload')}", file=sys.stderr)
                break
    finally:
        stream.stop()


async def run_server(args):
    model_path, model_arch = get_model_for_language(args.language, args.model_arch)
    transcriber = Transcriber(model_path=model_path, model_arch=model_arch)

    async def handler(websocket, *unused):
        await handle_telnyx_stream(websocket, args, transcriber)

    async with websockets.serve(handler, args.host, args.port):
        print(f"Listening for Telnyx media streams on ws://{args.host}:{args.port}")
        await asyncio.Future()


def parse_args():
    parser = argparse.ArgumentParser(description="Telnyx media streaming example")
    parser.add_argument("--host", default="localhost", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--language", default="en", help="Moonshine language")
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Moonshine model architecture to use",
    )
    parser.add_argument(
        "--track",
        choices=("inbound", "outbound", "both"),
        default="inbound",
        help="Telnyx media track to transcribe",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Print completed transcript lines only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_server(parse_args()))

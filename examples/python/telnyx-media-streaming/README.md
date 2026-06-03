# Telnyx Media Streaming

This example receives Telnyx Programmable Voice media events over WebSockets
and sends the decoded call audio to a Moonshine `Transcriber`.

## Install

```bash
pip install moonshine-voice websockets
python -m moonshine_voice.download --language en
```

## Run the server

```bash
python telnyx_media_streaming.py --host 0.0.0.0 --port 8765
```

Expose the server through a TLS endpoint, then use that `wss://` URL as the Telnyx `stream_url`.

## Start streaming from a call

When starting the media stream, request linear PCM audio so the example can
pass the samples directly to Moonshine:

```bash
curl -X POST \
  --header "Content-Type: application/json" \
  --header "Accept: application/json" \
  --header "Authorization: Bearer YOUR_API_KEY" \
  --data '{
    "stream_url": "wss://example.com/telnyx",
    "stream_track": "inbound_track",
    "stream_codec": "L16"
  }' \
  https://api.telnyx.com/v2/calls/{call_control_id}/actions/streaming_start
```

The script reads the `media_format` from the Telnyx `start` event, decodes
`media.payload`, and prints completed transcript lines. It also accepts
Telnyx's default `PCMU` payloads and `PCMA` payloads for quick testing.

Use `--track outbound` to transcribe only outbound audio, or `--track both` to
transcribe all received media frames.

![Moonshine Voice Logo](docs/images/logo.png)

# Moonshine Voice

**Voice Interfaces for Everyone**

[Moonshine](https://moonshine.ai) Voice is an open source AI toolkit for developers building real-time voice agents and applications.

- Everything runs on-device — fast, private, and with no account or API keys.
- Optimized for live streaming, with low latency by doing work while the user is still talking.
- Speech to text models trained from scratch, from [higher accuracy than Whisper Large V3](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) down to [tiny 1MB models](micro/README.md).
- One library across Python, JavaScript/WASM, iOS, Android, macOS, Linux, Windows, and Raspberry Pi.

## Quickstart

```bash
pip install moonshine-voice
moonshine-voice mic --language en
```

Or in the browser / Node:

```bash
npm install @moonshine-ai/moonshine-wasm
```

## Documentation

Full guides, models, and API reference: **[moonshine-voice.readthedocs.io](https://moonshine-voice.readthedocs.io/)**

- [Discord](https://discord.gg/27qp9zSRXF) for live support
- [GitHub Issues](https://github.com/moonshine-ai/moonshine/issues) for bugs and feature requests

## License

Licensed under the [Apache License 2.0](LICENSE).

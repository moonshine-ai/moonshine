![Moonshine Voice Logo](docs/images/logo.png)

# Moonshine Voice

**Voice Interfaces for Everyone**

[Moonshine](https://moonshine.ai) Voice is an open source AI toolkit for developers building real-time voice agents and applications.

Full guides, models, and API reference are at **[moonshine-voice.readthedocs.io](https://moonshine-voice.readthedocs.io/)**.

- Everything runs on-device — fast, private, and with no account or API keys.
- Optimized for live streaming, with low latency by doing work while the user is still talking.
- Speech to text models trained from scratch, from [higher accuracy than Whisper Large V3](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) down to [tiny 1MB models](micro/README.md).
- One library across [Python, JavaScript/WASM, iOS, Android, macOS, Linux, Windows, and Raspberry Pi](https://moonshine-voice.readthedocs.io/en/latest/quickstart/).

## Quickstart

<!-- doc-test: parse-only -->
```bash
pip install moonshine-voice
moonshine-voice mic --language en
```

Every other platform is covered in the [Quickstart](https://moonshine-voice.readthedocs.io/en/latest/quickstart/), with runnable samples in [Examples](https://moonshine-voice.readthedocs.io/en/latest/examples/).

## Documentation

- [Quickstart](https://moonshine-voice.readthedocs.io/en/latest/quickstart/) to install and run on your platform.
- [Using the Library](https://moonshine-voice.readthedocs.io/en/latest/using/) for transcription, text to speech, and conversational agents.
- [Models](https://moonshine-voice.readthedocs.io/en/latest/models/) for what's available, accuracy, and domain customization.
- [API Reference](https://moonshine-voice.readthedocs.io/en/latest/api/classes/) for the classes, options, and C API.

## Support

- [Discord](https://discord.gg/27qp9zSRXF) for live support.
- [GitHub Issues](https://github.com/moonshine-ai/moonshine/issues) for bugs and feature requests.

## License

Licensed under the [MIT License](LICENSE).

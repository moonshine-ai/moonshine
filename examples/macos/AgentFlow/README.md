# Dialog Flow Example

A Swift command-line assistant that walks you through setting up wifi by voice,
mirroring the web `agent-flow` demo and the Python `agent_flow.py` script.

`AgentFlow` owns the speech-to-text, embedding, and text-to-speech engines, so
the
interesting part of the program is the flow itself: ordinary `async` code that
asks a question and gets an answer back:

```swift
let ssid = try await d.ask("What's the name of your wifi network?")
guard try await d.confirm("I heard \(ssid). Is that right?") else { ... }
```

## Building

```bash
cd examples/macos/AgentFlow
swift build
```

## Running

```bash
# Speak to it (asks for microphone access the first time)
swift run AgentFlow

# Type the answers instead, for machines without a microphone
swift run AgentFlow --text
```

Say (or type) **"set up wifi"** to start the flow. Two commands work at any
point without being registered, because `AgentFlow` provides them:

- **"cancel"** abandons the flow
- **"start over"** runs it again from the top

## Notes

- The speech, embedding, and voice models are downloaded on first run into
  `~/Library/Caches/MoonshineModels/` and reused afterwards. The first run
  fetches roughly a gigabyte, so give it a minute.
- The microphone is muted while the assistant is speaking, so it does not
  transcribe itself.

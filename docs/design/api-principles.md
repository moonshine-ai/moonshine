# High-level API

## Goal

I want the language bindings to offer a fluent and opinionated API primarily focused on building voice interfaces, with enough flexibility to also build applications for related purposes like meet transcription and note-taking.

## Principles

 - The APIs should lean heavily on opinionated defaults that make the expected voice interface usage code terse and readable, even if it means that other use cases are more verbose. Common paths should involve a single function call or two, with the implementation details hidden away in the background.
 - Each API should adopt its platform's native idioms and patterns as much as possible.
 - The configuration process should lean towards a "builder" pattern, where the API is configured in a series of method calls, rather than a single function call with a lot of options.
 - The APIs can assume that a fast network connection is available by default, and treat that as the common case, with completely offline operation possible, but may involve more verbose code.
 - The language bindings should implement higher-level concepts like AgentFlow and voice cloning with defaults that hide the complexity of the underlying implementation. For example, AgentFlow should load the required STT models, embedding models, TTS models, and use an internal MicTranscriber instance so that the user can just call a single function to initialize a working voice interface, and not have to create multiple objects and pass them into an AgentFlow instance. As another example, voice cloning should hide the fact that it needs a STT model internally to transcribe the text from a clip to be cloned, should default to extracting the first four seconds of detected speech from any input audio, and should have a streaming API so that the user can clone a voice from a live audio stream. The streaming API should flag when it has enough data to clone a voice.
 - The (possibly time-consuming) resource loading should follow the native patterns used for similar high-latency, failure prone operations, with a bias towards patterns that allow the code itself to be written in a procedural way, but wrapped in an asynchronous block in the client code.

## Examples

These are in pseudo-code and may not compile as-is, but are intended to give concrete goals for the kind of results I want to see.

```javascript

let agent = AgentFlow();

function sayHello(df) {
    df.say("Hello!");
}

agent.listenFor("hello", sayHello);

await agent.load();

agent.startListening();

```

```java

TextToSpeech tts = new TextToSpeech(context).cloning();

tts.load();

tts.cloneFrom("some-speech.wav");

tts.say("Hello world!");

```
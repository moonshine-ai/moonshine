// The default app path: the live mic/speaker recognition service.
//
// Streams microphone hops in (USB by default; see usb_audio_io.h), runs the
// VAD -> STT pipeline, and speaks the recognized letter/digit back via the TTS.
// This is the moonshine_micro_echo target's app (see main_live.cc).

#ifndef SPELLING_ECHO_APP_H_
#define SPELLING_ECHO_APP_H_

namespace spelling {

// Run the live recognition service forever (never returns).
[[noreturn]] void RunEchoApp();

}  // namespace spelling

#endif  // SPELLING_ECHO_APP_H_

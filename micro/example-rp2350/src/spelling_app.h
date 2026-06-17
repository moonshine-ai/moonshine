// The default app path: the live mic/speaker recognition service.
//
// Streams microphone hops in (USB by default; see usb_audio_io.h), runs the
// VAD -> STT pipeline, and speaks the recognized letter/digit back via the TTS.
// Built only when SPELLING_TINY_AUDIO is set (which forces VAD + TTS on).

#ifndef SPELLING_SPELLING_APP_H_
#define SPELLING_SPELLING_APP_H_

namespace spelling {

// Run the live recognition service forever (never returns).
[[noreturn]] void RunSpellingApp();

}  // namespace spelling

#endif  // SPELLING_SPELLING_APP_H_

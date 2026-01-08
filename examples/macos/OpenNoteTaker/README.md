# OpenNoteTaker

A modern macOS desktop application for capturing and transcribing conversations using Moonshine Voice.

## Features

- **Dual Audio Capture**: Records from both your microphone and system audio (Zoom, meetings, etc.) using ScreenCaptureKit
- **Real-time Transcription**: Live transcription updates as speech is detected
- **Document-based**: Multiple transcript windows can be open simultaneously
- **Editable Transcripts**: Click on any completed transcript line to edit or delete it
- **Notes**: Insert manual notes between transcript lines
- **Export**: Save transcripts as .txt or .docx files
- **Save/Load**: Native .ont file format (JSON-based) for saving and loading transcripts
- **Time Stamps**: Optional time prefixes (HH:MM:SS) for each entry
- **Keyboard Shortcuts**: 
  - `Cmd+R`: Toggle recording
  - `Cmd+N`: New transcript
  - `Cmd+Shift+N`: New note

## Building

### Prerequisites

- macOS 13.0 or later
- Xcode 15.0 or later
- XcodeGen (optional, for generating Xcode project from project.yml)

### Using XcodeGen

1. Install XcodeGen if you haven't already:
   ```bash
   brew install xcodegen
   ```

2. Generate the Xcode project:
   ```bash
   cd examples/macos/OpenNoteTaker
   xcodegen generate
   ```

3. Open the generated project:
   ```bash
   open OpenNoteTaker.xcodeproj
   ```

### Manual Setup

1. Open Xcode and create a new macOS App project
2. Set the project name to "OpenNoteTaker"
3. Add the Moonshine Swift package as a dependency:
   - File → Add Package Dependencies
   - Add local package: `../../../swift`
4. Add all Swift files from `Sources/OpenNoteTaker/` to the project
5. Configure the Info.plist with the required permissions (see Info.plist in this directory)
6. Add the entitlements file (OpenNoteTaker.entitlements)

## Permissions

The app requires:
- **Microphone Permission**: For capturing your voice
- **Screen Recording Permission**: For capturing system audio from other applications

These permissions will be requested automatically when you first start recording.

## Usage

1. Launch the app
2. A new transcript window will open automatically
3. Click the "Start Recording" button (or press `Cmd+R`) to begin transcription
4. Speak or play audio from other applications
5. Transcript lines will appear in real-time
6. Click on any completed line to edit or delete it
7. Use `Cmd+Shift+N` to insert a manual note
8. Use File → Export to save as .txt or .docx
9. Use File → Save to save in the native .ont format

## Architecture

- **App.swift**: Main app entry point with menu commands
- **TranscriptDocument.swift**: Document model and file I/O
- **TranscriptWindowView.swift**: Main UI with scrollable transcript view
- **TranscriptWindowViewModel.swift**: Recording logic and transcription event handling
- **AudioCapture.swift**: Dual audio capture (microphone + system audio)
- **AppState.swift**: Global app state for coordinating multiple windows
- **ExportManager.swift**: Export functionality for .txt and .docx
- **SettingsView.swift**: User preferences (timestamp visibility)

## Notes

- The app uses the Moonshine Voice Swift package for transcription
- System audio capture requires Screen Recording permission
- Multiple document windows can be open, but only one can record at a time
- When recording starts in one window, any other active recording will pause automatically


export const MoonshineSettings = {
    FRAME_SIZE: 250,
    LOOKBACK_FRAMES: 5,
    // MAX_SPEECH_SECS: 5,
    // MIN_REFRESH_SECS: 0.3,
    LOOKBACK_SIZE: 500 * 5,
    MAX_RECORD_MS: 30000,
    BASE_ASSET_PATH: "/dist/"
}

export enum MoonshineLifecycle {
    idle = "idle",
    loading = "loading",
    transcribing = "transcribing"
}

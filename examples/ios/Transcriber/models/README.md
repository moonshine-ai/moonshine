# Models

This example no longer bundles model files in the app. The **Medium Streaming
English** model is downloaded on first run (via `AssetDownloader`) into the
app's Application Support directory and reused thereafter. See
`TranscriberApp.swift`.

This folder is intentionally kept (the Xcode project references it) but ships
empty, so the app package stays small.

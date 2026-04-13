// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TextToSpeech",
    platforms: [.macOS(.v13)],
    dependencies: [
        // Uncomment this back in when you want to use the locally-built Swift package.
        // .package(path: "../../../swift")
        .package(url: "https://github.com/moonshine-ai/moonshine-swift.git", from: "0.0.54")
    ],
    targets: [
        .executableTarget(
            name: "TextToSpeech",
            dependencies: [
                // Uncomment this back in when you want to use the locally-built Swift package.
                // .product(name: "MoonshineVoice", package: "swift")
                .product(name: "MoonshineVoice", package: "moonshine-swift")
            ]
        )
    ]
)

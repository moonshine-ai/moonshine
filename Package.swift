// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "Moonshine",
    platforms: [
        .iOS(.v14),
        .macOS(.v13),
    ],
    products: [
        .library(name: "Moonshine", targets: ["MoonshineVoice"])
    ],
    targets: [
        .binaryTarget(
            name: "moonshine",
            // path: "swift/Moonshine.xcframework",
            url:
                "https://github.com/moonshine-ai/moonshine-v2/releases/download/v0.0.8/Moonshine.xcframework.zip",
            checksum: "79535bc9e6d9fc554dc35740b856cc7e47240ace31e538cce4dba6eb20fd8ff1"
        ),
        .target(
            name: "MoonshineVoice",
            dependencies: ["moonshine"],
            path: "swift/Sources/MoonshineVoice"
        ),
        .testTarget(
            name: "MoonshineVoiceTests",
            dependencies: ["MoonshineVoice"],
            path: "swift/Tests/MoonshineVoiceTests"
        ),
    ]
)

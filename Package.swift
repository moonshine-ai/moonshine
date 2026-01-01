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
                "https://github.com/moonshine-ai/moonshine-v2/releases/download/v0.0.7/Moonshine.xcframework.zip",
            checksum: "40e0891feb4cdd2993f0f4e6d2001c331e3ef1d551b3b01f57a79fccff1b3632"
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

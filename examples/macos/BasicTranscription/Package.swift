// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "BasicTranscription",
    platforms: [.macOS(.v13)],
    dependencies: [
        // .package(path: "../../../swift")
        .package(url: "https://github.com/moonshine-ai/moonshine-v2.git", from: "0.0.8")
    ],
    targets: [
        .executableTarget(
            name: "BasicTranscription",
            dependencies: [
                .product(name: "Moonshine", package: "moonshine-v2")
            ]
        )
    ]
)

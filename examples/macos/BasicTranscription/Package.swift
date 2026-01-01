// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "BasicTranscription",
    platforms: [.macOS(.v13)],
    dependencies: [
        // .package(path: "../../..")
        .package(url: "https://github.com/moonshine-ai/moonshine-swift.git", from: "0.0.15")
    ],
    targets: [
        .executableTarget(
            name: "BasicTranscription",
            dependencies: [
                .product(name: "Moonshine", package: "moonshine-swift")
            ]
        )
    ]
)

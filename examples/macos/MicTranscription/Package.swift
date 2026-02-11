// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "MicTranscription",
    platforms: [.macOS(.v13)],
    dependencies: [
        // Uncomment this back in when you want to use the locally-built Swift package.
        // .package(path: "../../../swift")
        .package(url: "https://github.com/moonshine-ai/moonshine-swift.git", from: "0.0.48")
    ],
    targets: [
        .executableTarget(
            name: "MicTranscription",
            dependencies: [
                // Uncomment this back in when you want to use the locally-built Swift package.
                // .product(name: "Moonshine", package: "swift")
                .product(name: "Moonshine", package: "moonshine-swift")
            ]
        )
    ]
)

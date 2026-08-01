// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "AgentFlow",
    platforms: [.macOS(.v13)],
    dependencies: [
        // Uncomment this back in when you want to use the locally-built Swift package.
        // .package(path: "../../../swift")
        .package(url: "https://github.com/moonshine-ai/moonshine-swift.git", from: "0.1.1")
    ],
    targets: [
        .executableTarget(
            name: "AgentFlow",
            dependencies: [
                // Uncomment this back in when you want to use the locally-built Swift package.
                // .product(name: "MoonshineVoice", package: "swift")
                .product(name: "MoonshineVoice", package: "moonshine-swift")
            ]
        )
    ]
)

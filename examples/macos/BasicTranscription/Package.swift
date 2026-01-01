// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "BasicTranscription",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../../../ios")
    ],
    targets: [
        .executableTarget(
            name: "BasicTranscription",
            dependencies: [
                .product(name: "Moonshine", package: "ios")
            ]
        )
    ]
)

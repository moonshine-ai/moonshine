// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "OpenNoteTaker",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../../../swift")
    ],
    targets: [
        .executableTarget(
            name: "OpenNoteTaker",
            dependencies: [
                .product(name: "Moonshine", package: "swift")
            ]
        )
    ]
)


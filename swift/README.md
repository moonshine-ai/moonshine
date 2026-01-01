# Moonshine iOS/macOS Package

Swift Package for Moonshine Voice that supports both iOS and macOS.

## Building for macOS

The `Moonshine.xcframework` currently only contains iOS slices. To use this package on macOS, you need to build a macOS framework and add it to the xcframework.

### Option 1: Build macOS Framework and Add to XCFramework

1. Build the macOS framework:
```bash
cd core
mkdir -p build/build-macos
cd build/build-macos
cmake -G Xcode -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 ..
cmake --build . --config Release
```

2. Create a macOS framework from the build:
```bash
# The framework should be at: core/build/build-macos/Release/moonshine.framework
# Or if built as a library, create a framework structure
```

3. Add macOS slice to the xcframework:
```bash
cd ../../ios
xcodebuild -create-xcframework \
    -framework ../core/build/build-macos/Release/moonshine.framework \
    -framework Moonshine.xcframework/ios-arm64/moonshine.framework \
    -framework Moonshine.xcframework/ios-arm64_x86_64-simulator/moonshine.framework \
    -output Moonshine.xcframework
```

### Option 2: Use System Library (Current Workaround)

The package includes a system library target for macOS that links against `libmoonshine.dylib`. To use this:

1. Build the macOS dylib:
```bash
cd core
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
# This creates libmoonshine.dylib in the build directory
```

2. Ensure the dylib is in a location where the linker can find it:
   - Copy to `/usr/local/lib/`, or
   - Set `DYLD_LIBRARY_PATH` to include the build directory, or
   - Update the linker settings in Package.swift to point to the correct path

## Using the Package

### iOS
```swift
import Moonshine

let transcriber = try Transcriber(modelPath: "...", modelArch: .base)
```

### macOS
```swift
import Moonshine

let transcriber = try Transcriber(modelPath: "...", modelArch: .base)
```

The package automatically uses the correct underlying library (framework for iOS, dylib for macOS).


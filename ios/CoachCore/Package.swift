// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CoachCore",
    defaultLocalization: "en",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "CoachCore", targets: ["CoachCore"])
    ],
    targets: [
        .target(name: "CoachCore"),
        .testTarget(name: "CoachCoreTests", dependencies: ["CoachCore"]),
    ]
)

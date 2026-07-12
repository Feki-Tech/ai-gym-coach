// Exercise definitions — thresholds identical to the desktop prototype.

import Foundation

public enum ExerciseMode { case reps, hold }

public struct ExerciseSpec {
    public let name: String
    public let signal: SignalKey       // angle driving the FSM (down, then up)
    public let startBelow: Double      // below this => rep started
    public let bottomBelow: Double     // deep enough for full ROM
    public let lockoutAbove: Double    // back above this => rep complete
    public let concentricPhase: String // "ascent" | "descent" (angle direction of the lift)
    public let minRepS: Double
    public let minConcentricS: Double  // faster => "slow down"
    public let mode: ExerciseMode
    public let cameraHint: String

    public init(name: String, signal: SignalKey,
                startBelow: Double = 0, bottomBelow: Double = 0,
                lockoutAbove: Double = 0, concentricPhase: String = "ascent",
                minRepS: Double = 0.8, minConcentricS: Double = 0.6,
                mode: ExerciseMode = .reps,
                cameraHint: String? = nil) {
        self.name = name
        self.signal = signal
        self.startBelow = startBelow
        self.bottomBelow = bottomBelow
        self.lockoutAbove = lockoutAbove
        self.concentricPhase = concentricPhase
        self.minRepS = minRepS
        self.minConcentricS = minConcentricS
        self.mode = mode
        self.cameraHint = cameraHint ?? loc("hint.side")
    }
}

public let specs: [String: ExerciseSpec] = [
    "squat": ExerciseSpec(name: "squat", signal: .knee, startBelow: 150,
                          bottomBelow: 100, lockoutAbove: 165,
                          cameraHint: loc("hint.side_45front")),
    "pushup": ExerciseSpec(name: "pushup", signal: .elbow, startBelow: 140,
                           bottomBelow: 95, lockoutAbove: 155,
                           minConcentricS: 0.4),
    "bench": ExerciseSpec(name: "bench", signal: .elbow, startBelow: 140,
                          bottomBelow: 90, lockoutAbove: 160,
                          cameraHint: loc("hint.side_head")),
    "deadlift": ExerciseSpec(name: "deadlift", signal: .hip, startBelow: 150,
                             bottomBelow: 100, lockoutAbove: 165),
    "lunge": ExerciseSpec(name: "lunge", signal: .knee, startBelow: 150,
                          bottomBelow: 110, lockoutAbove: 165,
                          cameraHint: loc("hint.side_45knee")),
    "shoulder_press": ExerciseSpec(name: "shoulder_press", signal: .elbow,
                                   startBelow: 150, bottomBelow: 100,
                                   lockoutAbove: 160,
                                   cameraHint: loc("hint.front")),
    "curl": ExerciseSpec(name: "curl", signal: .elbow, startBelow: 140,
                         bottomBelow: 70, lockoutAbove: 155,
                         concentricPhase: "descent", minConcentricS: 0.5),
    "pullup": ExerciseSpec(name: "pullup", signal: .elbow, startBelow: 140,
                           bottomBelow: 80, lockoutAbove: 160,
                           concentricPhase: "descent",
                           cameraHint: loc("hint.front")),
    "plank": ExerciseSpec(name: "plank", signal: .bodyLine, mode: .hold),
]

/// Display order for pickers.
public let exerciseOrder = ["squat", "pushup", "bench", "deadlift", "lunge",
                            "shoulder_press", "curl", "pullup", "plank"]

public func displayName(_ exercise: String) -> String {
    let key = "exercise.\(exercise)"
    let localized = NSLocalizedString(key, bundle: .module, comment: "")
    if localized != key { return localized }
    return exercise.replacingOccurrences(of: "_", with: " ").capitalized
}

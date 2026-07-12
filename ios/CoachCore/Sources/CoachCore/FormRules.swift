// Form rules, fault catalog, scoring, and rate-limited feedback.

import Foundation

/// fault id -> (priority, message, score penalty). Lower priority = said first.
public let faultMessages: [String: (priority: Int, message: String, penalty: Int)] = [
    "back_lean": (0, "Straighten your back.", 30),
    "back_round": (0, "Keep your back flat — chest up.", 30),
    "body_sag": (0, "Keep your body in a straight line.", 25),
    "knees_cave": (0, "Push your knees out — don't let them cave in.", 25),
    "shallow": (1, "Go deeper — full range of motion.", 20),
    "elbow_swing": (1, "Keep your elbows pinned to your sides.", 20),
    "elbow_flare": (1, "Tuck your elbows closer to your body.", 15),
    "torso_lean": (1, "Keep your torso upright.", 15),
    "lean_back": (1, "Don't lean back — brace your core.", 15),
    "uneven": (1, "Even it out — both sides together.", 15),
    "chin": (1, "Pull higher — chin over the bar.", 15),
    "shrug_neck": (1, "Keep your neck neutral.", 10),
    "too_fast": (2, "Slow down — control the movement.", 10),
]

public let fatigueMessage = "You're slowing down — keep form tight or end the set."

@inline(__always)
private func moving(_ s: RepState) -> Bool { s != .idle }

/// Per-frame faults, phase-gated — mirrors the desktop LIVE_RULES table.
public func liveFaults(exercise: String, ang: BodyAngles, state: RepState) -> [String] {
    var f: [String] = []
    switch exercise {
    case "squat":
        if moving(state) && ang.trunkLean > 50 { f.append("back_lean") }
        if (state == .bottom || state == .ascent) && ang.valgusRatio < 0.7 {
            f.append("knees_cave")
        }
    case "pushup":
        if moving(state) && ang.bodyLine < 155 { f.append("body_sag") }
        if state == .bottom && ang.elbowFlare > 100 { f.append("elbow_flare") }
    case "bench":
        if moving(state) && ang.wristYDiff > 0.08 { f.append("uneven") }
    case "deadlift":
        if moving(state) && ang.neck < 150 { f.append("back_round") }
    case "lunge":
        if moving(state) && ang.trunkLean > 30 { f.append("torso_lean") }
    case "shoulder_press":
        if moving(state) && ang.trunkLean > 20 { f.append("lean_back") }
        if moving(state) && ang.wristYDiff > 0.08 { f.append("uneven") }
    case "curl":
        if moving(state) && ang.upperArmSwing > 25 { f.append("elbow_swing") }
        if moving(state) && ang.trunkLean > 20 { f.append("torso_lean") }
    case "pullup":
        if state == .bottom && ang.noseAboveWrists < 0 { f.append("chin") }
        if moving(state) && ang.wristYDiff > 0.10 { f.append("uneven") }
    case "plank":
        if ang.neck < 140 { f.append("shrug_neck") }
    default:
        break
    }
    return f
}

/// Faults judged once per completed rep.
public func repFaults(spec: ExerciseSpec, ev: RepEvent) -> [String] {
    var f: [String] = []
    if !ev.fullDepth { f.append("shallow") }
    if ev.concentricS < spec.minConcentricS { f.append("too_fast") }
    return f
}

public func scoreRep(_ ev: RepEvent) -> Int {
    max(0, 100 - ev.faults.compactMap { faultMessages[$0]?.penalty }.reduce(0, +))
}

/// Rate-limited, priority-ordered coaching cues.
public final class FeedbackEngine {
    let cooldown: Double
    var lastSaid: [String: Double] = [:]
    public var current = ""

    public init(cooldown: Double = 3.0) {
        self.cooldown = cooldown
    }

    /// Returns the message if a new cue fired (for the voice channel).
    @discardableResult
    public func push(_ faults: [String], t: Double) -> String? {
        let ordered = faults.sorted {
            (faultMessages[$0]?.priority ?? 9) < (faultMessages[$1]?.priority ?? 9)
        }
        for fault in ordered {
            if t - (lastSaid[fault] ?? -1e9) >= cooldown {
                lastSaid[fault] = t
                current = faultMessages[fault]?.message ?? fault
                return current
            }
        }
        if faults.isEmpty { current = "" }
        return nil
    }

    public func praise() -> String {
        current = "Great form!"
        return current
    }
}

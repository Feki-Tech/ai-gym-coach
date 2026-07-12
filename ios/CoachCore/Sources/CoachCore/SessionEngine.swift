// SessionEngine — platform-independent per-frame orchestrator.
// Feed it smoothed-or-raw skeletons; it handles smoothing, auto-detection,
// rep counting, plank tracking, live/rep faults, fatigue, feedback and the
// session log. The UI layer only renders `HUDState` and speaks `spokenCues`.

import Foundation

public struct HUDState {
    public init() {}
    public var exercise: String?          // nil while auto-detect is searching
    public var phase: String = "IDLE"
    public var repCount = 0
    public var lastScore: Int?
    public var cue = ""                   // on-screen coaching line
    public var plankHold: Double?
    public var plankBest: Double?
    public var signalValue: Double?
    public var trunkLean: Double?
    public var detecting = false
}

public struct FrameOutput {
    public var hud: HUDState
    public var spokenCues: [String]       // hand these to TTS
    public var repEvent: RepEvent?
}

public final class SessionEngine {
    public private(set) var exercise: String?
    var spec: ExerciseSpec?
    let detector: AutoDetector?
    let smoother = SkeletonSmoother()
    let feedback = FeedbackEngine()
    var counter: RepCounter?
    var plank: PlankTracker?
    let fatigue = FatigueMonitor()
    public let builder = SessionBuilder()
    public private(set) var hud = HUDState()

    /// exercise: name from `specs`, or "auto" to detect from movement.
    public init(exercise: String) {
        if exercise == "auto" {
            self.exercise = nil
            self.spec = nil
            self.detector = AutoDetector()
            hud.detecting = true
        } else {
            self.exercise = exercise
            self.spec = specs[exercise]
            self.detector = nil
            self.counter = specs[exercise].map { RepCounter(spec: $0) }
            if specs[exercise]?.mode == .hold { self.plank = PlankTracker() }
        }
    }

    public func process(_ raw: Skeleton, t: Double) -> FrameOutput {
        var spoken: [String] = []
        var repEvent: RepEvent? = nil
        let pts = smoother.update(raw, t: t)
        let ang = bodyAngles(pts)

        if spec == nil, let det = detector {                 // auto-detect
            hud.detecting = true
            if let found = det.update(frameFeatures(ang, pts), t: t) {
                exercise = found
                spec = specs[found]
                counter = specs[found].map { RepCounter(spec: $0) }
                if specs[found]?.mode == .hold { plank = PlankTracker() }
                hud.exercise = found
                hud.detecting = false
                spoken.append(String(format: loc("coach.detected"),
                                     displayName(found)))
            }
        } else if let spec = spec, let counter = counter {
            hud.exercise = exercise
            hud.detecting = false
            if let plank = plank {                           // timed hold
                var faultsNow = liveFaults(exercise: spec.name, ang: ang,
                                           state: counter.state)
                if plank.update(bodyLine: ang.bodyLine, t: t) {
                    faultsNow.append("body_sag")
                }
                if let msg = feedback.push(faultsNow, t: t) {
                    spoken.append(msg)
                }
                hud.plankHold = plank.total
                hud.plankBest = plank.best
            } else {                                         // rep exercise
                let faultsNow = liveFaults(exercise: spec.name, ang: ang,
                                           state: counter.state)
                for f in faultsNow { counter.noteFault(f) }
                let ev0 = counter.update(angle: ang.value(spec.signal), t: t)
                if let msg = feedback.push(faultsNow, t: t) {
                    spoken.append(msg)
                }
                if var ev = ev0 {
                    ev.faults = Array(Set(ev.faults)
                        .union(repFaults(spec: spec, ev: ev))).sorted()
                    ev.score = scoreRep(ev)
                    hud.lastScore = ev.score
                    // concentric velocity proxy: ROM (deg) / lift time (s)
                    let vel = max(spec.lockoutAbove - ev.minAngle, 1.0)
                        / max(ev.concentricS, 0.05)
                    builder.addRep(ev, velocity: vel)
                    if fatigue.add(vel) {
                        feedback.current = fatigueMessage
                        spoken.append(fatigueMessage)
                    } else if ev.faults.isEmpty {
                        spoken.append("\(ev.count). \(feedback.praise())")
                    } else if let cue = feedback.push(ev.faults, t: t) {
                        spoken.append("\(ev.count). \(cue)")
                    } else {
                        spoken.append("\(ev.count).")
                    }
                    repEvent = ev
                }
                hud.repCount = counter.count
            }
            hud.phase = counter.state.rawValue
            hud.signalValue = ang.value(spec.signal)
            hud.trunkLean = ang.trunkLean
        }

        hud.cue = feedback.current
        return FrameOutput(hud: hud, spokenCues: spoken, repEvent: repEvent)
    }

    /// Build the final session record (call once at the end).
    public func finish(durationS: Double) -> SessionRecord {
        builder.finish(exercise: exercise ?? "auto", durationS: durationS,
                       plank: plank)
    }
}

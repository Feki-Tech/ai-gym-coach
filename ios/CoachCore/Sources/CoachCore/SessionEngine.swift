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
    // --- what the coach can drive / what the richer HUD shows
    public var faultsNow: [String] = []    // live faults this frame (skeleton highlight)
    public var repGoal: Int?
    public var restLeft: Double = 0        // seconds of coach-set rest remaining
    public var restNext: String?           // what comes after the rest
    public var program: ProgramStatus?
    public var loadKg: Double?
    public var tempoTarget: Double?
    public var cuesOn = true
    public var bestReps = 0                // record to beat (0 = unknown)
    public var thresholds: (start: Double, bottom: Double, lockout: Double)?
    public var concentricPhase = "ascent"
}

/// Coach-driven session settings (the ACTION protocol of the desktop app).
public struct SessionConfig {
    public var repGoal: Int?
    public var restUntil: Double = 0       // engine time (s)
    public var tempoEccTarget: Double?
    public var cuesOn = true
    public var loadKg: Double?
    public init() {}
}

public struct FrameOutput {
    public var hud: HUDState
    public var spokenCues: [String]       // hand these to TTS
    public var repEvent: RepEvent?
    public var events: [SessionEvent] = []
}

/// Things the UI/coach should react to (spoken by the app, sent to the LLM).
public enum SessionEvent: Equatable {
    case goalReached(reps: Int)
    case setDone(exercise: String)          // a program set finished
    case programAdvanced(message: String, restS: Int, next: String?)
    case programDone
    case restOver
    case personalRecord(String)
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
    public var config = SessionConfig()
    public private(set) var program: WorkoutProgram?
    public var bestReps = 0                 // from history, for live PR cues
    private var prReported = false
    private var wasResting = false
    private var goalReported = false
    private var setMark = 0
    private var lastT = 0.0

    /// exercise: name from `specs`, or "auto" to detect from movement.
    /// exercise: name from `specs`, or "auto" to detect from movement.
    /// model: optional trained classifier (TinyMLP.load) — auto-detect then
    /// uses the same gated, versioned MLP as the desktop instead of rules.
    public init(exercise: String, model: TinyMLP? = nil) {
        if exercise == "auto" {
            self.exercise = nil
            self.spec = nil
            self.detector = model.map { MLDetector(model: $0) } ?? AutoDetector()
            hud.detecting = true
        } else {
            self.exercise = exercise
            self.spec = specs[exercise]
            self.detector = nil
            self.counter = specs[exercise].map { RepCounter(spec: $0) }
            if specs[exercise]?.mode == .hold { self.plank = PlankTracker() }
        }
    }

    // MARK: - coach controls

    /// Switch exercise (fresh counters); "auto" re-detects.
    public func switchExercise(_ name: String) {
        guard name == "auto" || specs[name] != nil else { return }
        if name == "auto" {
            exercise = nil
            spec = nil
            counter = nil
            plank = nil
            hud.detecting = true
            hud.exercise = nil
        } else {
            exercise = name
            spec = specs[name]
            counter = specs[name].map { RepCounter(spec: $0) }
            plank = specs[name]?.mode == .hold ? PlankTracker() : nil
            hud.detecting = false
            hud.exercise = name
        }
        hud.repCount = 0
        hud.lastScore = nil
        hud.plankHold = nil
        hud.plankBest = nil
        fatigue.reset()
        prReported = false
        goalReported = false
        setMark = builder.reps.count
    }

    /// Start a fresh set of the same exercise (program "same" step, key press).
    public func newSet() {
        if let ex = exercise { switchExercise(ex) }
    }

    public func startRest(seconds: Double) {
        config.restUntil = max(config.restUntil, lastT + seconds)
    }

    public func cancelRest() {
        config.restUntil = 0
    }

    public func start(program p: WorkoutProgram) -> String {
        program = p
        if let b = p.current {
            switchExercise(b.exercise)
            config.repGoal = b.reps
        }
        return String(format: loc("program.start"), p.overview,
                      displayName(p.current?.exercise ?? ""), p.current?.target ?? "")
    }

    public func stopProgram() {
        program = nil
        config.repGoal = nil
    }

    /// Reps logged since the current set started (for the set_done debrief).
    public var currentSetReps: [RepRecord] {
        Array(builder.reps.dropFirst(setMark))
    }

    private func advanceProgram(t: Double) -> [SessionEvent] {
        guard let p = program else { return [] }
        var out: [SessionEvent] = [.setDone(exercise: exercise ?? "")]
        let r = p.onSetDone()
        if r.restS > 0 { config.restUntil = max(config.restUntil, t + Double(r.restS)) }
        switch r.step {
        case .done:
            program = nil
            config.repGoal = nil
            out.append(.programAdvanced(message: r.message, restS: 0, next: nil))
            out.append(.programDone)
        case .next:
            let b = p.current!
            switchExercise(b.exercise)
            config.repGoal = b.reps
            out.append(.programAdvanced(message: r.message, restS: r.restS, next: b.exercise))
        case .same:
            newSet()
            config.repGoal = p.current?.reps
            out.append(.programAdvanced(message: r.message, restS: r.restS, next: exercise))
        }
        return out
    }

    public func process(_ raw: Skeleton, t: Double) -> FrameOutput {
        var spoken: [String] = []
        var events: [SessionEvent] = []
        var repEvent: RepEvent? = nil
        lastT = t
        let pts = smoother.update(raw, t: t)
        let ang = bodyAngles(pts)
        let resting = t < config.restUntil
        if wasResting && !resting { events.append(.restOver); spoken.append(loc("coach.rest_over")) }
        wasResting = resting
        // form cues respect the coach's mute switch and the rest timer
        func cue(_ msg: String?) {
            if let m = msg, config.cuesOn, !resting { spoken.append(m) }
        }

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
                cue(feedback.push(faultsNow, t: t))
                hud.plankHold = plank.total
                hud.plankBest = plank.best
                hud.faultsNow = faultsNow
                if let b = program?.current, let hold = b.holdS, b.exercise == spec.name,
                   plank.streak >= Double(hold) {
                    events += advanceProgram(t: t)
                }
            } else {                                         // rep exercise
                let faultsNow = liveFaults(exercise: spec.name, ang: ang,
                                           state: counter.state)
                hud.faultsNow = faultsNow
                for f in faultsNow { counter.noteFault(f) }
                let ev0 = counter.update(angle: ang.value(spec.signal), t: t)
                cue(feedback.push(faultsNow, t: t))
                if var ev = ev0 {
                    ev.faults = Array(Set(ev.faults)
                        .union(repFaults(spec: spec, ev: ev))).sorted()
                    ev.score = scoreRep(ev)
                    hud.lastScore = ev.score
                    // concentric velocity proxy: ROM (deg) / lift time (s)
                    let vel = max(spec.lockoutAbove - ev.minAngle, 1.0)
                        / max(ev.concentricS, 0.05)
                    builder.addRep(ev, velocity: vel, loadKg: config.loadKg)
                    if fatigue.add(vel) {
                        feedback.current = fatigueMessage
                        spoken.append(fatigueMessage)
                    } else if ev.faults.isEmpty {
                        cue("\(ev.count). \(feedback.praise())")
                    } else if let c = feedback.push(ev.faults, t: t) {
                        cue("\(ev.count). \(c)")
                    } else {
                        cue("\(ev.count).")
                    }
                    if let tgt = config.tempoEccTarget, ev.eccentricS < tgt - 0.4 {
                        cue(String(format: loc("coach.tempo_slower"), tgt))
                    }
                    if bestReps > 0, ev.count > bestReps, !prReported {
                        prReported = true
                        let msg = String(format: loc("coach.pr_reps"), ev.count)
                        feedback.current = msg
                        spoken.append(msg)
                        events.append(.personalRecord(msg))
                    }
                    if let goal = config.repGoal, ev.count == goal, program == nil, !goalReported {
                        goalReported = true
                        spoken.append(String(format: loc("coach.goal_reached"), goal))
                        events.append(.goalReached(reps: goal))
                    }
                    repEvent = ev
                    if let b = program?.current, let want = b.reps, b.exercise == spec.name,
                       ev.count >= want {
                        events += advanceProgram(t: t)
                    }
                }
                hud.repCount = self.counter?.count ?? 0
            }
            hud.phase = (self.counter ?? counter).state.rawValue
            hud.signalValue = ang.value(spec.signal)
            hud.trunkLean = ang.trunkLean
            hud.thresholds = spec.mode == .hold ? nil
                : (spec.startBelow, spec.bottomBelow, spec.lockoutAbove)
            hud.concentricPhase = spec.concentricPhase
        }

        hud.cue = feedback.current
        hud.repGoal = config.repGoal
        hud.restLeft = max(0, config.restUntil - t)
        hud.restNext = program?.status.map {
            String(format: loc("program.next"), displayName($0.exercise), $0.set, $0.sets, $0.target)
        }
        hud.program = program?.status
        hud.loadKg = config.loadKg
        hud.tempoTarget = config.tempoEccTarget
        hud.cuesOn = config.cuesOn
        hud.bestReps = bestReps
        return FrameOutput(hud: hud, spokenCues: spoken, repEvent: repEvent, events: events)
    }

    /// Build the final session record (call once at the end).
    public func finish(durationS: Double) -> SessionRecord {
        builder.finish(exercise: exercise ?? "auto", durationS: durationS,
                       plank: plank)
    }

    /// Set reps counted so far (rep exercises).
    public var repCount: Int { counter?.count ?? 0 }

}

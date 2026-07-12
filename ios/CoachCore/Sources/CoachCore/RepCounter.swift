// Rep-counting finite-state machine + plank hold tracker.
// Port of the desktop prototype's RepCounter/PlankTracker.

import Foundation

public enum RepState: String {
    case idle = "IDLE"
    case descent = "DESCENT"
    case bottom = "BOTTOM"
    case ascent = "ASCENT"
}

public struct RepEvent {
    public var count: Int
    public var duration: Double
    public var eccentricS: Double
    public var concentricS: Double
    public var minAngle: Double
    public var fullDepth: Bool
    public var faults: [String]
    public var score: Int

    public init(count: Int, duration: Double, eccentricS: Double,
                concentricS: Double, minAngle: Double, fullDepth: Bool,
                faults: [String] = [], score: Int = 100) {
        self.count = count
        self.duration = duration
        self.eccentricS = eccentricS
        self.concentricS = concentricS
        self.minAngle = minAngle
        self.fullDepth = fullDepth
        self.faults = faults
        self.score = score
    }
}

/// IDLE -> DESCENT -> BOTTOM -> ASCENT -> (rep++) on the signal angle.
///
/// "Descent/ascent" refer to the *angle*: for curls and pull-ups the angle
/// descends during the lift, so `concentricPhase` maps phases to tempo names.
public final class RepCounter {
    public let spec: ExerciseSpec
    public private(set) var state: RepState = .idle
    public private(set) var count = 0
    var tStart = 0.0
    var tBottom = 0.0
    var minAngle = 180.0
    var repFaults = Set<String>()

    public init(spec: ExerciseSpec) {
        self.spec = spec
    }

    public func noteFault(_ fault: String) {
        if state != .idle { repFaults.insert(fault) }
    }

    public func update(angle: Double, t: Double) -> RepEvent? {
        switch state {
        case .idle:
            if angle < spec.startBelow {
                state = .descent
                tStart = t
                minAngle = angle
                repFaults = []
            }
        case .descent:
            minAngle = min(minAngle, angle)
            if angle < spec.bottomBelow {
                state = .bottom
                tBottom = t
            } else if angle > minAngle + 15 {          // turned around early
                state = .ascent
                tBottom = t
            }
        case .bottom:
            minAngle = min(minAngle, angle)
            if angle > minAngle + 10 {
                state = .ascent
            }
        case .ascent:
            if angle > spec.lockoutAbove {
                let dur = t - tStart
                state = .idle
                if dur < spec.minRepS { return nil }   // noise blip, not a rep
                count += 1
                let downS = tBottom - tStart
                let upS = t - tBottom
                let (ecc, con) = spec.concentricPhase == "ascent"
                    ? (downS, upS) : (upS, downS)
                return RepEvent(count: count, duration: dur,
                                eccentricS: ecc, concentricS: con,
                                minAngle: minAngle,
                                fullDepth: minAngle < spec.bottomBelow,
                                faults: repFaults.sorted())
            }
        }
        return nil
    }
}

/// Timed hold: accumulate time while the body line stays straight.
public final class PlankTracker {
    let goodAbove: Double
    let graceS: Double
    public private(set) var total = 0.0
    public private(set) var streak = 0.0
    public private(set) var best = 0.0
    var badFor = 0.0
    var tPrev: Double?

    public init(goodAbove: Double = 160.0, graceS: Double = 1.0) {
        self.goodAbove = goodAbove
        self.graceS = graceS
    }

    /// Returns true when a "fix your line" cue should fire.
    public func update(bodyLine: Double, t: Double) -> Bool {
        let dt = tPrev.map { max(t - $0, 0.0) } ?? 0.0
        tPrev = t
        if bodyLine >= goodAbove {
            total += dt
            streak += dt
            best = max(best, streak)
            badFor = 0.0
            return false
        }
        let wasOK = badFor <= graceS
        badFor += dt
        if badFor > graceS {
            streak = 0.0
            return wasOK             // fire cue once when grace expires
        }
        return false
    }
}

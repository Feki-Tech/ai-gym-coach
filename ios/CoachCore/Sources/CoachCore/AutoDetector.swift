// Rule-based auto exercise detection + velocity-loss fatigue monitor.

import Foundation

/// Per-frame features consumed by AutoDetector (kept minimal so tests can
/// synthesize them without full skeletons).
public struct FrameFeatures {
    public var trunk: Double
    public var knee: Double
    public var elbow: Double
    public var hip: Double
    public var shoY: Double
    public var wriY: Double
    public var torso: Double
    public var overhead: Bool
    public var kneeSplit: Double

    public init(trunk: Double = 10, knee: Double = 170, elbow: Double = 170,
                hip: Double = 170, shoY: Double = 0.3, wriY: Double = 0.5,
                torso: Double = 0.25, overhead: Bool = false,
                kneeSplit: Double = 0.1) {
        self.trunk = trunk
        self.knee = knee
        self.elbow = elbow
        self.hip = hip
        self.shoY = shoY
        self.wriY = wriY
        self.torso = torso
        self.overhead = overhead
        self.kneeSplit = kneeSplit
    }
}

public func frameFeatures(_ ang: BodyAngles, _ pts: Skeleton) -> FrameFeatures {
    let shoY = (pts[.leftShoulder].y + pts[.rightShoulder].y) / 2
    let hipY = (pts[.leftHip].y + pts[.rightHip].y) / 2
    let wriY = (pts[.leftWrist].y + pts[.rightWrist].y) / 2
    let torso = max(abs(hipY - shoY), 1e-3)
    return FrameFeatures(
        trunk: ang.trunkLean, knee: ang.knee, elbow: ang.elbow, hip: ang.hip,
        shoY: shoY, wriY: wriY, torso: torso,
        overhead: wriY < shoY - 0.03,              // image y grows downward
        kneeSplit: abs(pts[.leftKnee].y - pts[.rightKnee].y) / torso
    )
}

/// Rule-based exercise classifier over a sliding window of skeleton features.
/// Locks after 3 agreeing votes. Bench press is NOT detectable from the
/// skeleton alone (looks like a push-up) — select it manually.
public final class AutoDetector {
    static let windowS = 2.0
    static let voteEveryS = 0.5
    static let needAgree = 3

    var buf: [(Double, FrameFeatures)] = []
    var votes: [String?] = []
    var nextVoteT = windowS

    public init() {}

    public func update(_ feat: FrameFeatures, t: Double) -> String? {
        buf.append((t, feat))
        while let first = buf.first, t - first.0 > Self.windowS {
            buf.removeFirst()
        }
        if t < nextVoteT || buf.count < 20 { return nil }
        nextVoteT = t + Self.voteEveryS
        let vote = classify()
        votes.append(vote)
        if votes.count > Self.needAgree { votes.removeFirst() }
        if votes.count == Self.needAgree, let v = vote,
           votes.allSatisfy({ $0 == v }) {
            return v
        }
        return nil
    }

    func classify() -> String? {
        let f = buf.map { $0.1 }
        func rom(_ key: (FrameFeatures) -> Double) -> Double {
            let vals = f.map(key)
            return (vals.max() ?? 0) - (vals.min() ?? 0)
        }
        func mean(_ key: (FrameFeatures) -> Double) -> Double {
            f.map(key).reduce(0, +) / Double(f.count)
        }
        let torso = mean { $0.torso }
        let trunkMean = mean { $0.trunk }
        let trunkMax = f.map { $0.trunk }.max() ?? 0
        let romKnee = rom { $0.knee }
        let romElbow = rom { $0.elbow }
        let romHip = rom { $0.hip }
        let overhead = Double(f.filter { $0.overhead }.count) / Double(f.count)
        let dispSho = rom { $0.shoY } / torso
        let dispWri = rom { $0.wriY } / torso
        let kneeSplit = f.map { $0.kneeSplit }.max() ?? 0

        if trunkMean > 55 {                        // body horizontal
            return romElbow > 25 ? "pushup" : "plank"
        }
        if overhead > 0.7 && romElbow > 30 {       // hands overhead
            return dispSho > 1.3 * dispWri ? "pullup" : "shoulder_press"
        }
        if romKnee > 35 {                          // legs driving
            if trunkMax > 55 { return "deadlift" }
            if kneeSplit > 0.35 { return "lunge" }
            return "squat"
        }
        if trunkMax > 55 && romHip > 30 {          // hip hinge, stiff knees
            return "deadlift"
        }
        if romElbow > 40 && overhead < 0.3 {       // arms only, below head
            return "curl"
        }
        return nil
    }
}

/// Velocity-based fatigue: warn when concentric speed drops >20% against
/// the best of the first three reps.
public final class FatigueMonitor {
    let threshold: Double
    var vels: [Double] = []
    var warned = false
    public private(set) var loss = 0.0

    public init(threshold: Double = 0.20) {
        self.threshold = threshold
    }

    /// Feed one rep's concentric velocity; true => fire fatigue cue.
    public func add(_ velocity: Double) -> Bool {
        vels.append(velocity)
        if vels.count < 4 { return false }
        let base = vels.prefix(3).max() ?? 1
        let cur = vels.suffix(2).reduce(0, +) / 2
        loss = base > 0 ? max(0.0, 1 - cur / base) : 0.0
        if loss > threshold && !warned {
            warned = true
            return true
        }
        return false
    }
}

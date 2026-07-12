// Skeleton model + per-frame body angles.
// 15 joints (Apple Vision body pose subset used by the coaching rules).

import Foundation

public enum Joint: Int, CaseIterable {
    case nose = 0
    case leftEar, rightEar
    case leftShoulder, rightShoulder
    case leftElbow, rightElbow
    case leftWrist, rightWrist
    case leftHip, rightHip
    case leftKnee, rightKnee
    case leftAnkle, rightAnkle
}

public struct Landmark {
    public var x: Double
    public var y: Double
    public var confidence: Double
    public init(x: Double, y: Double, confidence: Double) {
        self.x = x
        self.y = y
        self.confidence = confidence
    }
    public var p2: P2 { P2(x, y) }
}

/// Fixed-size array indexed by `Joint.rawValue`.
public typealias Skeleton = [Landmark]

public let visMin = 0.5

public extension Array where Element == Landmark {
    subscript(_ j: Joint) -> Landmark { self[j.rawValue] }
}

/// One Euro per coordinate per joint, holding the last good value while a
/// joint drops below the visibility threshold.
public final class SkeletonSmoother {
    private var filters: [[OneEuroFilter]]
    private var last: Skeleton?

    public init() {
        filters = (0..<Joint.allCases.count).map { _ in
            [OneEuroFilter(), OneEuroFilter()]
        }
    }

    public func update(_ pts: Skeleton, t: Double) -> Skeleton {
        var out = pts
        for i in 0..<pts.count {
            if pts[i].confidence < visMin, let l = last, l[i].confidence >= visMin {
                out[i] = l[i]                     // hold last good value
                continue
            }
            out[i].x = filters[i][0].filter(pts[i].x, t: t)
            out[i].y = filters[i][1].filter(pts[i].y, t: t)
        }
        last = out
        return out
    }
}

public enum SignalKey: String, Codable {
    case knee, hip, elbow, bodyLine
}

/// All per-frame features used by the FSM and the form rules.
public struct BodyAngles {
    public var side: String
    public var knee: Double
    public var hip: Double
    public var elbow: Double
    public var trunkLean: Double
    public var upperArmSwing: Double
    public var bodyLine: Double        // 180 = straight
    public var elbowFlare: Double
    public var neck: Double
    public var valgusRatio: Double     // < 1 => knees caving in
    public var wristYDiff: Double
    public var noseAboveWrists: Double

    public func value(_ key: SignalKey) -> Double {
        switch key {
        case .knee: return knee
        case .hip: return hip
        case .elbow: return elbow
        case .bodyLine: return bodyLine
        }
    }
}

public func pickSide(_ pts: Skeleton) -> String {
    let lJoints: [Joint] = [.leftShoulder, .leftElbow, .leftHip, .leftKnee, .leftAnkle]
    let rJoints: [Joint] = [.rightShoulder, .rightElbow, .rightHip, .rightKnee, .rightAnkle]
    let left = lJoints.map { pts[$0].confidence }.reduce(0, +) / 5
    let right = rJoints.map { pts[$0].confidence }.reduce(0, +) / 5
    return left >= right ? "L" : "R"
}

public func bodyAngles(_ pts: Skeleton) -> BodyAngles {
    let s = pickSide(pts)
    let ear: Joint = s == "L" ? .leftEar : .rightEar
    let sho: Joint = s == "L" ? .leftShoulder : .rightShoulder
    let elb: Joint = s == "L" ? .leftElbow : .rightElbow
    let wri: Joint = s == "L" ? .leftWrist : .rightWrist
    let hip: Joint = s == "L" ? .leftHip : .rightHip
    let kne: Joint = s == "L" ? .leftKnee : .rightKnee
    let ank: Joint = s == "L" ? .leftAnkle : .rightAnkle

    var valgus = 1.0
    if min(pts[.leftKnee].confidence, pts[.rightKnee].confidence,
           pts[.leftAnkle].confidence, pts[.rightAnkle].confidence) > visMin {
        let kneeW = abs(pts[.leftKnee].x - pts[.rightKnee].x)
        let ankleW = max(abs(pts[.leftAnkle].x - pts[.rightAnkle].x), 1e-4)
        valgus = kneeW / ankleW
    }
    var wristYDiff = 0.0
    var noseAboveWrists = 1.0
    if min(pts[.leftWrist].confidence, pts[.rightWrist].confidence) > visMin {
        wristYDiff = abs(pts[.leftWrist].y - pts[.rightWrist].y)
        noseAboveWrists = (pts[.leftWrist].y + pts[.rightWrist].y) / 2 - pts[.nose].y
    }

    return BodyAngles(
        side: s,
        knee: jointAngle(pts[hip].p2, pts[kne].p2, pts[ank].p2),
        hip: jointAngle(pts[sho].p2, pts[hip].p2, pts[kne].p2),
        elbow: jointAngle(pts[sho].p2, pts[elb].p2, pts[wri].p2),
        trunkLean: segmentVsVertical(top: pts[sho].p2, bottom: pts[hip].p2),
        upperArmSwing: segmentVsVertical(top: pts[sho].p2, bottom: pts[elb].p2),
        bodyLine: jointAngle(pts[sho].p2, pts[hip].p2, pts[ank].p2),
        elbowFlare: jointAngle(pts[hip].p2, pts[sho].p2, pts[elb].p2),
        neck: pts[ear].confidence > visMin
            ? jointAngle(pts[ear].p2, pts[sho].p2, pts[hip].p2) : 180.0,
        valgusRatio: valgus,
        wristYDiff: wristYDiff,
        noseAboveWrists: noseAboveWrists
    )
}

/// Skeleton edges for overlay drawing.
public let skeletonEdges: [(Joint, Joint)] = [
    (.leftShoulder, .rightShoulder), (.leftHip, .rightHip),
    (.leftShoulder, .leftHip), (.rightShoulder, .rightHip),
    (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
    (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
    (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
    (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
]

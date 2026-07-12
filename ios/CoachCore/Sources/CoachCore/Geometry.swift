// Geometry primitives + One Euro smoothing.
// Coordinate convention throughout CoachCore: normalized image coordinates
// with the origin at the TOP-LEFT and y growing DOWN (same as the Python
// desktop prototype). Convert Vision points (bottom-left origin) with
// y = 1 - point.y at ingestion.

import Foundation

public struct P2: Equatable {
    public var x: Double
    public var y: Double
    public init(_ x: Double, _ y: Double) {
        self.x = x
        self.y = y
    }
}

/// Angle ABC in degrees at vertex `b`.
public func jointAngle(_ a: P2, _ b: P2, _ c: P2) -> Double {
    let bax = a.x - b.x, bay = a.y - b.y
    let bcx = c.x - b.x, bcy = c.y - b.y
    let denom = (bax * bax + bay * bay).squareRoot() * (bcx * bcx + bcy * bcy).squareRoot()
    if denom < 1e-9 { return 180.0 }
    let cosang = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / denom))
    return acos(cosang) * 180.0 / .pi
}

/// Angle of the segment (top -> bottom) vs the vertical axis, in degrees.
/// 0 = perfectly vertical. Uses y-down image coordinates (-y is "up").
public func segmentVsVertical(top: P2, bottom: P2) -> Double {
    let vx = top.x - bottom.x, vy = top.y - bottom.y
    let n = (vx * vx + vy * vy).squareRoot()
    if n < 1e-9 { return 0.0 }
    let cosang = max(-1.0, min(1.0, -vy / n))
    return acos(cosang) * 180.0 / .pi
}

/// Adaptive low-pass filter: smooth when slow, responsive when fast.
public final class OneEuroFilter {
    let minCutoff: Double
    let beta: Double
    let dCutoff: Double
    var xPrev: Double?
    var dxPrev = 0.0
    var tPrev: Double?

    public init(minCutoff: Double = 1.0, beta: Double = 0.02, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    static func alpha(cutoff: Double, dt: Double) -> Double {
        let tau = 1.0 / (2 * .pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    public func filter(_ x: Double, t: Double) -> Double {
        guard let xp = xPrev, let tp = tPrev else {
            xPrev = x
            dxPrev = 0
            tPrev = t
            return x
        }
        let dt = max(t - tp, 1e-6)
        let dx = (x - xp) / dt
        let aD = Self.alpha(cutoff: dCutoff, dt: dt)
        let dxHat = aD * dx + (1 - aD) * dxPrev
        let cutoff = minCutoff + beta * abs(dxHat)
        let a = Self.alpha(cutoff: cutoff, dt: dt)
        let xHat = a * x + (1 - a) * xp
        xPrev = xHat
        dxPrev = dxHat
        tPrev = t
        return xHat
    }
}

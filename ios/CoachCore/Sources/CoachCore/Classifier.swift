// The trained exercise classifier — the SAME gated, versioned TinyMLP the
// desktop trains (pose_coach.py --train-classifier), exported as portable
// JSON (--export-model) and run here without any ML framework: ~1.5k
// parameters, a matrix multiply, a ReLU and a softmax. Parity with the
// Python engine is pinned by data/parity_fixtures.json (window_feature and
// mlp sections).

import Foundation

/// Fixed-size feature vector over a window of frames — port of the
/// desktop's window_features: per-channel mean/std/min/max (population
/// std!) plus torso-normalized shoulder & wrist travel.
public enum WindowFeaturesML {
    public static let ndim = 38

    static func channels(_ f: FrameFeatures) -> [Double] {
        [f.trunk, f.knee, f.elbow, f.hip, f.shoY, f.wriY, f.torso,
         f.overhead ? 1.0 : 0.0, f.kneeSplit]
    }

    private static let torsoIdx = 6
    private static let shoYIdx = 4
    private static let wriYIdx = 5

    public static func of(_ frames: [FrameFeatures]) -> [Double] {
        precondition(!frames.isEmpty, "window features need >= 1 frame")
        let rows = frames.map { channels($0) }
        let n = Double(rows.count)
        let nc = rows[0].count
        var mean = [Double](repeating: 0, count: nc)
        var mn = [Double](repeating: .infinity, count: nc)
        var mx = [Double](repeating: -.infinity, count: nc)
        for r in rows {
            for c in 0..<nc {
                mean[c] += r[c]
                mn[c] = Swift.min(mn[c], r[c])
                mx[c] = Swift.max(mx[c], r[c])
            }
        }
        for c in 0..<nc { mean[c] /= n }
        var std = [Double](repeating: 0, count: nc)
        for r in rows {
            for c in 0..<nc {
                let d = r[c] - mean[c]
                std[c] += d * d
            }
        }
        for c in 0..<nc { std[c] = (std[c] / n).squareRoot() }
        let torso = Swift.max(mean[torsoIdx], 1e-3)
        var out = [Double]()
        out.reserveCapacity(ndim)
        out.append(contentsOf: mean)
        out.append(contentsOf: std)
        out.append(contentsOf: mn)
        out.append(contentsOf: mx)
        out.append((mx[shoYIdx] - mn[shoYIdx]) / torso)
        out.append((mx[wriYIdx] - mn[wriYIdx]) / torso)
        return out
    }
}

/// Two-layer MLP inference: (x-mu)/sd -> ReLU hidden -> softmax classes.
public final class TinyMLP {
    public let classes: [String]
    public let minProba: Double
    public let modelVersion: String
    let w1: [[Double]]      // [ndim][hidden]
    let b1: [Double]
    let w2: [[Double]]      // [hidden][classes]
    let b2: [Double]
    let mu: [Double]
    let sd: [Double]

    public init(classes: [String], minProba: Double,
                w1: [[Double]], b1: [Double],
                w2: [[Double]], b2: [Double],
                mu: [Double], sd: [Double],
                modelVersion: String = "unknown") {
        self.classes = classes
        self.minProba = minProba
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.mu = mu
        self.sd = sd
        self.modelVersion = modelVersion
    }

    /// Load a pose_coach.py --export-model file. Returns nil on any
    /// malformed input — a broken model must never crash a workout.
    public static func load(json data: Data) -> TinyMLP? {
        guard let root = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any],
              let classes = root["classes"] as? [String],
              let w1 = matrix(root["W1"]), let b1 = vector(root["b1"]),
              let w2 = matrix(root["W2"]), let b2 = vector(root["b2"]),
              let mu = vector(root["mu"]), let sd = vector(root["sd"])
        else { return nil }
        let minProba = (root["min_proba"] as? NSNumber)?.doubleValue ?? 0.75
        let version = ((root["manifest"] as? [String: Any])?["model_version"]
                       as? String) ?? "unknown"
        return TinyMLP(classes: classes, minProba: minProba,
                       w1: w1, b1: b1, w2: w2, b2: b2, mu: mu, sd: sd,
                       modelVersion: version)
    }

    public static func load(url: URL) -> TinyMLP? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return load(json: data)
    }

    private static func vector(_ any: Any?) -> [Double]? {
        (any as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
    }

    private static func matrix(_ any: Any?) -> [[Double]]? {
        guard let rows = any as? [Any] else { return nil }
        var out: [[Double]] = []
        for row in rows {
            guard let v = vector(row) else { return nil }
            out.append(v)
        }
        return out
    }

    public func predict(_ x: [Double]) -> [Double] {
        var xn = [Double](repeating: 0, count: x.count)
        for i in 0..<x.count { xn[i] = (x[i] - mu[i]) / sd[i] }
        var h = [Double](repeating: 0, count: b1.count)
        for j in 0..<b1.count {
            var acc = b1[j]
            for i in 0..<xn.count { acc += xn[i] * w1[i][j] }
            h[j] = Swift.max(0.0, acc)
        }
        var z = [Double](repeating: 0, count: b2.count)
        for k in 0..<b2.count {
            var acc = b2[k]
            for j in 0..<h.count { acc += h[j] * w2[j][k] }
            z[k] = acc
        }
        let zmax = z.max() ?? 0
        let e = z.map { Foundation.exp($0 - zmax) }
        let tot = e.reduce(0, +)
        return e.map { $0 / tot }
    }
}

/// AutoDetector with the rule-based vote swapped for the trained MLP —
/// same sliding window, vote cadence, and 3-agreeing-votes lock-in as the
/// desktop.
public final class MLDetector: AutoDetector {
    let model: TinyMLP

    public init(model: TinyMLP) {
        self.model = model
        super.init()
    }

    override func classify() -> String? {
        let p = model.predict(WindowFeaturesML.of(buf.map { $0.1 }))
        guard let maxP = p.max(), let ci = p.firstIndex(of: maxP),
              maxP >= model.minProba else { return nil }
        return model.classes[ci]
    }
}

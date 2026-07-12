// Workout logging — Codable session records persisted as JSON, same shape
// as the desktop prototype's workout_log.json.

import Foundation

public struct RepRecord: Codable, Identifiable {
    public var id: Int { n }
    public let n: Int
    public let score: Int
    public let eccentricS: Double
    public let concentricS: Double
    public let minAngle: Double
    public let velocity: Double?
    public let faults: [String]

    enum CodingKeys: String, CodingKey {
        case n, score, faults, velocity
        case eccentricS = "eccentric_s"
        case concentricS = "concentric_s"
        case minAngle = "min_angle"
    }
}

public struct PlankRecord: Codable {
    public let totalHoldS: Double
    public let bestStreakS: Double

    enum CodingKeys: String, CodingKey {
        case totalHoldS = "total_hold_s"
        case bestStreakS = "best_streak_s"
    }
}

public struct SessionSummary: Codable {
    public let reps: Int
    public let avgScore: Double?
    public let avgConcentricS: Double?
    public let faultCounts: [String: Int]
    public let velocityLossPct: Double?

    enum CodingKeys: String, CodingKey {
        case reps
        case avgScore = "avg_score"
        case avgConcentricS = "avg_concentric_s"
        case faultCounts = "fault_counts"
        case velocityLossPct = "velocity_loss_pct"
    }
}

public struct SessionRecord: Codable, Identifiable {
    public var id: String { started }
    public let started: String
    public let exercise: String
    public let durationS: Double
    public let reps: [RepRecord]
    public let plank: PlankRecord?
    public let summary: SessionSummary

    enum CodingKeys: String, CodingKey {
        case started, exercise, reps, plank, summary
        case durationS = "duration_s"
    }
}

/// Collects one session in memory, then builds the record.
public final class SessionBuilder {
    public private(set) var reps: [RepRecord] = []
    let started: Date

    public init(now: Date = Date()) {
        started = now
    }

    public func addRep(_ ev: RepEvent, velocity: Double?) {
        reps.append(RepRecord(
            n: ev.count, score: ev.score,
            eccentricS: (ev.eccentricS * 100).rounded() / 100,
            concentricS: (ev.concentricS * 100).rounded() / 100,
            minAngle: (ev.minAngle * 10).rounded() / 10,
            velocity: velocity.map { ($0 * 10).rounded() / 10 },
            faults: ev.faults))
    }

    public func finish(exercise: String, durationS: Double,
                       plank: PlankTracker? = nil) -> SessionRecord {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd HH:mm:ss"
        var faultCounts: [String: Int] = [:]
        for r in reps {
            for f in r.faults { faultCounts[f, default: 0] += 1 }
        }
        let scores = reps.map { Double($0.score) }
        let cons = reps.map { $0.concentricS }
        let vels = reps.compactMap { $0.velocity }.filter { $0 > 0 }
        var lossPct: Double? = nil
        if vels.count >= 4 {
            let base = vels.prefix(3).max() ?? 0
            let cur = vels.suffix(2).reduce(0, +) / 2
            if base > 0 {
                lossPct = (max(0.0, 1 - cur / base) * 1000).rounded() / 10
            }
        }
        return SessionRecord(
            started: fmt.string(from: started),
            exercise: exercise,
            durationS: (durationS * 10).rounded() / 10,
            reps: reps,
            plank: plank.map {
                PlankRecord(totalHoldS: ($0.total * 10).rounded() / 10,
                            bestStreakS: ($0.best * 10).rounded() / 10)
            },
            summary: SessionSummary(
                reps: reps.count,
                avgScore: scores.isEmpty ? nil
                    : ((scores.reduce(0, +) / Double(scores.count)) * 10).rounded() / 10,
                avgConcentricS: cons.isEmpty ? nil
                    : ((cons.reduce(0, +) / Double(cons.count)) * 100).rounded() / 100,
                faultCounts: faultCounts,
                velocityLossPct: lossPct))
    }
}

/// JSON-file-backed history store (Documents/workout_log.json on device).
public final class WorkoutStore {
    public let url: URL

    public init(url: URL) {
        self.url = url
    }

    public static func documentsStore() -> WorkoutStore {
        let dir = FileManager.default.urls(for: .documentDirectory,
                                           in: .userDomainMask)[0]
        return WorkoutStore(url: dir.appendingPathComponent("workout_log.json"))
    }

    public func load() -> [SessionRecord] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONDecoder().decode([SessionRecord].self, from: data)) ?? []
    }

    public func append(_ session: SessionRecord) throws {
        var history = load()
        history.append(session)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(history).write(to: url, options: .atomic)
    }
}

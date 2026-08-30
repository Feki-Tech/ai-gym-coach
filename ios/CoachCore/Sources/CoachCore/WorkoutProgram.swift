// Guided workout programs — port of the desktop WorkoutProgram: ordered
// blocks of sets the app runs for you (counts sets, rests, switches
// exercises). Text format, one block per comma:
//   "squat 3x10 rest 90, pushup 2x15 rest 45, plank 2x40s rest 30"
// "40s" (or any number on a hold-type exercise) means seconds held.

import Foundation

public struct ProgramBlock: Equatable {
    public let exercise: String
    public let sets: Int
    public let reps: Int?          // nil for timed holds
    public let holdS: Int?
    public let restS: Int

    public var target: String {
        if let h = holdS { return String(format: loc("program.hold_target"), h) }
        return String(format: loc("program.reps_target"), reps ?? 0)
    }
}

public struct ProgramStatus: Equatable {
    public let exercise: String
    public let set: Int
    public let sets: Int
    public let block: Int
    public let blocks: Int
    public let target: String
    public let reps: Int?
    public let holdS: Int?
}

public enum ProgramStep: Equatable {
    case same          // next set, same exercise
    case next          // new block (exercise changes)
    case done
}

public enum ProgramError: Error, Equatable {
    case unreadable(String)
    case unknownExercise(String)
    case range(String)
    case empty
}

private let exerciseAliases: [String: String] = [
    "push_up": "pushup", "push-up": "pushup", "push up": "pushup", "pushups": "pushup",
    "bench_press": "bench", "bench press": "bench",
    "bicep_curl": "curl", "biceps_curl": "curl", "bicep curl": "curl", "curls": "curl",
    "pull_up": "pullup", "pull-up": "pullup", "pull up": "pullup", "chin_up": "pullup",
    "pullups": "pullup", "chin up": "pullup",
    "overhead_press": "shoulder_press", "ohp": "shoulder_press",
    "military_press": "shoulder_press", "press": "shoulder_press",
    "shoulder press": "shoulder_press", "overhead press": "shoulder_press",
    "squats": "squat", "lunges": "lunge", "deadlifts": "deadlift", "planks": "plank",
]

public func normalizeExercise(_ raw: String) -> String? {
    let key = raw.trimmingCharacters(in: .whitespaces).lowercased()
    if specs[key] != nil { return key }
    if let a = exerciseAliases[key] { return a }
    let snake = key.replacingOccurrences(of: " ", with: "_").replacingOccurrences(of: "-", with: "_")
    if specs[snake] != nil { return snake }
    return exerciseAliases[snake]
}

public final class WorkoutProgram {
    public let blocks: [ProgramBlock]
    public private(set) var blockIndex = 0
    public private(set) var setIndex = 1        // 1-based

    public init(blocks: [ProgramBlock]) {
        self.blocks = blocks
    }

    /// "squat 3x10 rest 90, pushup 2x15 rest 45, plank 2x40s"
    public static func parse(_ text: String) throws -> WorkoutProgram {
        let pattern = #"^\s*([A-Za-z][A-Za-z _-]*?)\s*(\d+)\s*[x×]\s*(\d+)\s*(s|sec|secs|seconds)?\s*(?:rest\s*(\d+)\s*(?:s|sec|secs|seconds)?)?\s*$"#
        let re = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        var blocks: [ProgramBlock] = []
        let pieces = text.replacingOccurrences(of: " then ", with: ",", options: .caseInsensitive)
            .components(separatedBy: CharacterSet(charactersIn: ",;\n"))
        for raw in pieces {
            let piece = raw.trimmingCharacters(in: .whitespaces)
            if piece.isEmpty { continue }
            let ns = piece as NSString
            guard let m = re.firstMatch(in: piece, range: NSRange(location: 0, length: ns.length))
            else { throw ProgramError.unreadable(piece) }
            func group(_ i: Int) -> String? {
                let r = m.range(at: i)
                return r.location == NSNotFound ? nil : ns.substring(with: r)
            }
            guard let name = group(1), let ex = normalizeExercise(name)
            else { throw ProgramError.unknownExercise(group(1) ?? piece) }
            let sets = Int(group(2) ?? "") ?? 0
            let num = Int(group(3) ?? "") ?? 0
            let hold = group(4) != nil || specs[ex]?.mode == .hold
            guard (1...10).contains(sets) else { throw ProgramError.range("sets 1-10") }
            if hold {
                guard (5...600).contains(num) else { throw ProgramError.range("hold 5-600 s") }
            } else {
                guard (1...100).contains(num) else { throw ProgramError.range("reps 1-100") }
            }
            let rest = min(Int(group(5) ?? "") ?? 60, 900)
            blocks.append(ProgramBlock(exercise: ex, sets: sets, reps: hold ? nil : num,
                                       holdS: hold ? num : nil, restS: rest))
        }
        if blocks.isEmpty { throw ProgramError.empty }
        return WorkoutProgram(blocks: blocks)
    }

    public var current: ProgramBlock? {
        blockIndex < blocks.count ? blocks[blockIndex] : nil
    }

    public var overview: String {
        blocks.map { b in
            "\(displayName(b.exercise)) \(b.sets)x" + (b.holdS.map { "\($0)s" } ?? "\(b.reps ?? 0)")
        }.joined(separator: ", ")
    }

    public var status: ProgramStatus? {
        guard let b = current else { return nil }
        return ProgramStatus(exercise: b.exercise, set: setIndex, sets: b.sets,
                             block: blockIndex + 1, blocks: blocks.count,
                             target: b.target, reps: b.reps, holdS: b.holdS)
    }

    /// Advance after a completed set → (announcement, rest seconds, what next).
    public func onSetDone() -> (message: String, restS: Int, step: ProgramStep) {
        let b = blocks[blockIndex]
        if setIndex < b.sets {
            setIndex += 1
            return (String(format: loc("program.set_done"), setIndex - 1, b.sets, b.restS,
                           setIndex, b.target), b.restS, .same)
        }
        blockIndex += 1
        setIndex = 1
        if blockIndex >= blocks.count {
            return (loc("program.complete"), 0, .done)
        }
        let nb = blocks[blockIndex]
        return (String(format: loc("program.block_done"), displayName(b.exercise), b.restS,
                       displayName(nb.exercise), nb.sets, nb.target), b.restS, .next)
    }
}

// The interactive trainer on the phone: a conversation with the desktop's
// LLM coach (coach_server.py) that sees the live session, speaks its
// answers sentence by sentence, reacts to set/session events on its own,
// and can drive the workout (switch exercise, rep goal, rest, tempo, cues,
// load, programs) through the same ACTION protocol as the desktop app.

import Foundation
import Combine
import CoachCore

struct ChatLine: Identifiable, Equatable {
    enum Role: Equatable { case athlete, coach, app }
    let id = UUID()
    let role: Role
    var text: String
}

/// What the athlete needs to know about the mic, same five states as the
/// desktop (coach_voice.py): the pill and the chat header render exactly these.
enum TurnState: Equatable { case off, listening, hearing, thinking, speaking }

@MainActor
final class CoachSession: ObservableObject {
    @Published private(set) var transcript: [ChatLine] = []
    @Published private(set) var busy = false
    @Published private(set) var lastReply = ""
    @Published private(set) var handsFreeOn = false
    @Published private(set) var coachSpeaking = false
    @Published var draft = ""

    /// Sentences read aloud per answer on the speaker (the rest waits for
    /// "more"); unlimited in full duplex, where you can just interrupt.
    static let spokenLimit = 2
    static let moreHint = NSLocalizedString("Say more for the rest.", comment: "")
    private static let moreWords: Set<String> = [
        "more", "say more", "tell me more", "go on", "continue", "the rest",
        "details", "encore", "la suite", "weiter", "mehr", "más", "mas", "sigue",
        "continúa", "المزيد", "كمل", "أكمل", "继续", "更多"]

    private var recentSpoken: [String] = []
    private var spokenThisAnswer = 0
    private var more: [String] = []
    private var turnPending = false        // a question was asked: signal when the mic is back

    /// Keep the mic open while the coach talks (echo cancellation) so the
    /// athlete can interrupt by voice. Off = the mic closes while the coach
    /// talks, answers are capped at `spokenLimit` sentences + "more".
    var fullDuplex: Bool {
        UserDefaults.standard.object(forKey: "coach.fullDuplex") as? Bool ?? true
    }

    var turnState: TurnState {
        if coachSpeaking { return .speaking }
        if busy { return .thinking }
        switch handsFreeInput?.state {
        case .hearing: return .hearing
        case .listening: return .listening
        case .paused: return .thinking            // mic closed, answer on its way
        default: return .off
        }
    }

    var turnHint: String {
        switch turnState {
        case .listening: return NSLocalizedString("just talk", comment: "")
        case .hearing: return NSLocalizedString("finish your sentence", comment: "")
        case .thinking: return NSLocalizedString("hang on", comment: "")
        case .speaking: return fullDuplex
            ? NSLocalizedString("just talk to interrupt", comment: "")
            : NSLocalizedString("tap ■ to interrupt · say \"more\" for the rest", comment: "")
        case .off: return NSLocalizedString("hold the mic or type", comment: "")
        }
    }

    let client = CoachClient.shared
    private let speech: SpeechCoach
    private var engineProvider: () -> SessionEngine?
    private var stateProvider: () -> [String: Any]
    private var streamTask: Task<Void, Never>?

    /// Applied actions the workout view wants to know about (e.g. to show a toast).
    var onAction: ((CoachAction, String) -> Void)?

    init(speech: SpeechCoach, engine: @escaping () -> SessionEngine?,
         state: @escaping () -> [String: Any]) {
        self.speech = speech
        self.engineProvider = engine
        self.stateProvider = state
    }

    var isAvailable: Bool { client.isConfigured }
    private var handsFreeInput: SpeechInput?
    private var gateTimer: Timer?

    // MARK: - hands-free listening (like the desktop: just speak)

    /// Continuous mic, gated while the coach thinks or talks so it never
    /// hears its own voice; each finished utterance becomes a question.
    func startHandsFree(_ input: SpeechInput) {
        guard handsFreeInput == nil else { return }
        handsFreeInput = input
        handsFreeOn = true
        input.onUtterance = { [weak self] text in self?.ask(text) }
        input.startHandsFree()
        gateTimer = Timer.scheduledTimer(withTimeInterval: 0.35, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.gate() }
        }
    }

    func stopHandsFree() {
        gateTimer?.invalidate()
        gateTimer = nil
        handsFreeInput?.stopHandsFree()
        handsFreeInput = nil
        handsFreeOn = false
    }

    private func gate() {
        guard let input = handsFreeInput else { return }
        let speaking = speech.isSpeaking
        if speaking != coachSpeaking { coachSpeaking = speaking }
        if fullDuplex {
            input.resume()                       // echo cancellation keeps it open
            if !busy && !speaking { turnPending = false }
        } else if busy || speaking {
            input.pause()
        } else if input.state == .paused {
            input.resume(signal: turnPending)    // "your turn" tone after an answer
            turnPending = false
        }
    }

    /// The coach's own sentence coming back through the mic (full duplex).
    /// Mirrors coach_voice.looks_like_echo: mostly words of one recent
    /// sentence (or two consecutive ones) → echo, never a question.
    static func looksLikeEcho(_ text: String, spoken: [String], minOverlap: Double = 0.6) -> Bool {
        func tokens(_ s: String) -> [String] {
            s.lowercased().split { !($0.isLetter || $0.isNumber || $0 == "'") }.map(String.init)
        }
        let toks = tokens(text)
        let sents = spoken.map(tokens).filter { !$0.isEmpty }
        guard !toks.isEmpty, !sents.isEmpty else { return false }
        var cands = sents
        for i in 1..<max(1, sents.count) { cands.append(sents[i - 1] + sents[i]) }
        if toks.count < 3 {
            let piece = " " + toks.joined(separator: " ") + " "
            return cands.contains { (" " + $0.joined(separator: " ") + " ").contains(piece) }
        }
        let best = cands.map { c -> Double in
            let set = Set(c)
            return Double(toks.filter { set.contains($0) }.count) / Double(toks.count)
        }.max() ?? 0
        return best >= minOverlap
    }

    static func isMore(_ text: String) -> Bool {
        let t = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: .punctuationCharacters)
        return moreWords.contains(t)
    }

    private func speakReply(_ s: String) {
        recentSpoken.append(s)
        if recentSpoken.count > 8 { recentSpoken.removeFirst() }
        let limit = fullDuplex ? 0 : Self.spokenLimit
        if limit > 0 && spokenThisAnswer >= limit {
            more.append(s)                        // waits for "more"
            return
        }
        spokenThisAnswer += 1
        speech.sayChat(s)
    }

    /// "more": read the sentences held back by the spoken cap, no server call.
    private func speakMore(_ q: String) {
        let n = Self.spokenLimit
        let chunk = Array(more.prefix(n))
        more.removeFirst(min(n, more.count))
        transcript.append(ChatLine(role: .athlete, text: q))
        transcript.append(ChatLine(role: .coach, text: chunk.joined(separator: " ")))
        for s in chunk { recentSpoken.append(s); speech.sayChat(s) }
        if !more.isEmpty { speech.sayChat(Self.moreHint) }
        turnPending = true
    }

    // MARK: - talking

    func ask(_ text: String) {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, client.isConfigured else { return }
        if fullDuplex, Self.looksLikeEcho(q, spoken: recentSpoken) { return }
        if Self.isMore(q), !more.isEmpty {
            interrupt()
            speakMore(q)
            return
        }
        interrupt()
        more = []
        turnPending = true
        transcript.append(ChatLine(role: .athlete, text: q))
        run(client.chat(q, state: stateProvider()))
    }

    /// Proactive coaching: set_done / session_start / session_done.
    func notify(_ event: String, payload: [String: Any]) {
        guard client.isConfigured, !busy else { return }        // events are disposable
        run(client.event(event, payload: payload, state: stateProvider()))
    }

    /// Barge-in: stop speaking and drop the answer in progress.
    func interrupt() {
        streamTask?.cancel()
        streamTask = nil
        speech.stopSpeaking()
        busy = false
    }

    private func run(_ stream: AsyncStream<CoachEvent>) {
        busy = true
        spokenThisAnswer = 0
        transcript.append(ChatLine(role: .coach, text: ""))
        let idx = transcript.count - 1
        var buffer = ""
        streamTask = Task { [weak self] in
            guard let self else { return }
            for await ev in stream {
                if Task.isCancelled { break }
                switch ev {
                case .delta(let d):
                    transcript[idx].text += d
                    buffer += d
                    for s in Self.completeSentences(&buffer) { speakReply(s) }
                case .action(let a, let ack):
                    apply(a, ack: ack)
                case .done(let reply):
                    if !buffer.trimmingCharacters(in: .whitespaces).isEmpty {
                        speakReply(buffer.trimmingCharacters(in: .whitespaces))
                        buffer = ""
                    }
                    if !more.isEmpty { speech.sayChat(Self.moreHint) }
                    lastReply = reply.isEmpty ? transcript[idx].text : reply
                case .error(let e):
                    transcript[idx].text += (transcript[idx].text.isEmpty ? "" : " ") + "(\(e))"
                }
            }
            if transcript.indices.contains(idx), transcript[idx].text.isEmpty {
                transcript.remove(at: idx)
            }
            busy = false
        }
    }

    static func completeSentences(_ buf: inout String) -> [String] {
        var out: [String] = []
        while let r = buf.rangeOfCharacter(from: CharacterSet(charactersIn: ".!?。！？")) {
            let end = buf.index(after: r.lowerBound)
            let s = String(buf[..<end]).trimmingCharacters(in: .whitespaces)
            buf = String(buf[end...])
            if s.count > 1 { out.append(s) }
        }
        return out
    }

    // MARK: - the coach drives the app

    private func apply(_ a: CoachAction, ack: String) {
        guard let engine = engineProvider() else { return }
        var note = ack
        switch a.do {
        case "set_exercise":
            if let ex = a.exercise?.lowercased().replacingOccurrences(of: " ", with: "_") {
                engine.switchExercise(ex == "auto" ? "auto" : (normalizeExercise(ex) ?? ex))
            }
        case "set_rep_goal":
            if let n = a.reps, (1...100).contains(n) { engine.config.repGoal = n }
        case "rest_timer":
            if let s = a.seconds, (5...900).contains(s) { engine.startRest(seconds: s) }
        case "set_tempo":
            if let e = a.eccentricS, (0.5...10).contains(e) { engine.config.tempoEccTarget = e }
        case "cues":
            engine.config.cuesOn = a.enabled ?? true
        case "set_load":
            if let kg = a.kg, (0...500).contains(kg) { engine.config.loadKg = kg > 0 ? kg : nil }
        case "start_program":
            if let plan = a.plan, let p = try? WorkoutProgram.parse(plan) {
                note = engine.start(program: p)
                speech.sayChat(note)
            }
        case "stop_program":
            engine.stopProgram()
        default:
            return
        }
        transcript.append(ChatLine(role: .app, text: note))
        onAction?(a, note)
    }
}

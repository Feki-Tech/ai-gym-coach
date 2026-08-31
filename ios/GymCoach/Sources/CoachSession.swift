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

@MainActor
final class CoachSession: ObservableObject {
    @Published private(set) var transcript: [ChatLine] = []
    @Published private(set) var busy = false
    @Published private(set) var lastReply = ""
    @Published private(set) var handsFreeOn = false
    @Published var draft = ""

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
        if busy || speech.isSpeaking {
            input.pause()
        } else {
            input.resume()
        }
    }

    // MARK: - talking

    func ask(_ text: String) {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, client.isConfigured else { return }
        interrupt()
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
                    for s in Self.completeSentences(&buffer) { speech.sayChat(s) }
                case .action(let a, let ack):
                    apply(a, ack: ack)
                case .done(let reply):
                    if !buffer.trimmingCharacters(in: .whitespaces).isEmpty {
                        speech.sayChat(buffer.trimmingCharacters(in: .whitespaces))
                        buffer = ""
                    }
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

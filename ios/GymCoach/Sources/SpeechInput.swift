// Speech input for the coach — hands-free like the desktop app, with
// hold-to-talk as fallback. On-device Speech framework; only the text
// leaves the phone (to your own coach server).
//
// Hands-free: the mic listens continuously; an utterance ends after a
// short pause and is handed to `onUtterance`. The owner gates the mic
// (pause/resume) while the coach is talking so it never hears itself.
// The microphone is selectable (built-in, AirPods/Bluetooth, wired/USB).

import Foundation
import AVFoundation
import Combine
import Speech

struct MicInfo: Identifiable, Equatable {
    let id: String        // AVAudioSessionPortDescription.uid
    let name: String
    let type: String      // port type raw value
}

@MainActor
final class SpeechInput: ObservableObject {
    enum State: Equatable { case off, listening, hearing, paused, holdRecording }

    @Published private(set) var state: State = .off
    @Published private(set) var partial = ""
    @Published private(set) var level: Float = 0
    @Published private(set) var availableMics: [MicInfo] = []
    @Published var lastError: String?
    @Published var preferredMicUID: String {
        didSet {
            UserDefaults.standard.set(preferredMicUID, forKey: "mic.uid")
            applyPreferredMic()
        }
    }

    /// Hands-free: a finished utterance ("why do my knees cave?").
    var onUtterance: ((String) -> Void)?

    private let engine = AVAudioEngine()
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var utteranceTimer: Timer?
    private var lastPartialAt = Date.distantPast
    private var handsFree = false
    private var engineRunning = false

    init() {
        preferredMicUID = UserDefaults.standard.string(forKey: "mic.uid") ?? ""
        let lang = Bundle.main.preferredLocalizations.first ?? "en"
        let locale: Locale
        switch true {
        case lang.hasPrefix("zh"): locale = Locale(identifier: "zh-CN")
        case lang.hasPrefix("hi"): locale = Locale(identifier: "hi-IN")
        case lang.hasPrefix("es"): locale = Locale(identifier: "es-ES")
        case lang.hasPrefix("fr"): locale = Locale(identifier: "fr-FR")
        case lang.hasPrefix("ar"): locale = Locale(identifier: "ar-SA")
        default: locale = Locale(identifier: "en-US")
        }
        recognizer = SFSpeechRecognizer(locale: locale) ?? SFSpeechRecognizer()
        recognizer?.defaultTaskHint = .dictation
    }

    var isAvailable: Bool { recognizer != nil }

    // MARK: - permissions

    func requestPermissions() async -> Bool {
        let speechOK: Bool = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
        }
        let micOK: Bool = await withCheckedContinuation { cont in
            AVAudioSession.sharedInstance().requestRecordPermission { cont.resume(returning: $0) }
        }
        if !speechOK || !micOK {
            lastError = NSLocalizedString("Microphone or speech recognition not allowed — enable both in Settings.", comment: "")
        } else if recognizer?.isAvailable != true {
            lastError = NSLocalizedString("Speech recognition is unavailable right now (language not downloaded, or no network for server-based recognition).", comment: "")
        }
        return speechOK && micOK
    }

    // MARK: - microphone selection

    func refreshMics() {
        prepareAudioSession()
        let ports = AVAudioSession.sharedInstance().availableInputs ?? []
        availableMics = ports.map { MicInfo(id: $0.uid, name: $0.portName,
                                            type: $0.portType.rawValue) }
    }

    private func applyPreferredMic() {
        let session = AVAudioSession.sharedInstance()
        guard !preferredMicUID.isEmpty,
              let port = session.availableInputs?.first(where: { $0.uid == preferredMicUID })
        else {
            try? session.setPreferredInput(nil)          // system default
            return
        }
        do { try session.setPreferredInput(port) }
        catch { lastError = error.localizedDescription }
    }

    private func prepareAudioSession() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playAndRecord, mode: .default,
                                 options: [.duckOthers, .defaultToSpeaker,
                                           .allowBluetooth, .allowBluetoothA2DP])
        try? session.setActive(true, options: .notifyOthersOnDeactivation)
        applyPreferredMic()
    }

    // MARK: - hands-free

    /// Start continuous listening; utterances arrive via `onUtterance`.
    func startHandsFree() {
        guard !handsFree else { return }
        handsFree = true
        lastError = nil
        startEngine()
        beginRecognition()
        utteranceTimer?.invalidate()
        utteranceTimer = Timer.scheduledTimer(withTimeInterval: 0.35, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    func stopHandsFree() {
        guard handsFree else { return }
        handsFree = false
        utteranceTimer?.invalidate()
        utteranceTimer = nil
        endRecognition()
        stopEngine()
        state = .off
    }

    /// Coach is talking / thinking: drop what's buffered and go deaf.
    func pause() {
        guard handsFree, state == .listening || state == .hearing else { return }
        endRecognition()
        partial = ""
        state = .paused
    }

    /// Coach finished: open the mic again.
    func resume() {
        guard handsFree, state == .paused else { return }
        beginRecognition()
    }

    private func tick() {
        guard handsFree, state == .hearing else { return }
        // a short silence after speech = the utterance is complete
        if Date().timeIntervalSince(lastPartialAt) > 1.3 {
            finishUtterance()
        }
    }

    private func finishUtterance() {
        let text = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        endRecognition()
        partial = ""
        if !text.isEmpty, text.count > 1 {
            onUtterance?(text)
            state = .paused          // the owner resumes once the reply is done
        } else if handsFree {
            beginRecognition()
        }
    }

    private func beginRecognition() {
        guard let recognizer, recognizer.isAvailable else {
            state = .off
            lastError = NSLocalizedString("Speech recognition is unavailable right now (language not downloaded, or no network for server-based recognition).", comment: "")
            return
        }
        endRecognition()
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        req.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        request = req
        partial = ""
        state = .listening
        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let r = result {
                    let text = r.bestTranscription.formattedString
                    if !text.isEmpty {
                        self.partial = text
                        self.lastPartialAt = Date()
                        if self.state == .listening { self.state = .hearing }
                    }
                    if r.isFinal, self.handsFree, self.state == .hearing {
                        self.finishUtterance()
                        return
                    }
                }
                if error != nil {
                    // on-device tasks end after ~1 min; just start a new one
                    if self.handsFree, self.state == .listening || self.state == .hearing {
                        self.beginRecognition()
                    } else if self.state == .holdRecording {
                        self.state = .off
                    }
                }
            }
        }
    }

    private func endRecognition() {
        task?.cancel()
        task = nil
        request?.endAudio()
        request = nil
    }

    // MARK: - hold-to-talk (fallback when hands-free is off)

    func startHold() {
        guard state == .off || state == .paused else { return }
        lastError = nil
        startEngine()
        beginRecognition()
        state = .holdRecording
    }

    func stopHold() async -> String {
        guard state == .holdRecording else { return "" }
        request?.endAudio()
        try? await Task.sleep(nanoseconds: 350_000_000)
        let text = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        endRecognition()
        partial = ""
        if handsFree {
            state = .paused
        } else {
            stopEngine()
            state = .off
        }
        return text
    }

    // MARK: - audio engine (one tap, always feeding the current request)

    private func startEngine() {
        guard !engineRunning else { return }
        prepareAudioSession()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            lastError = NSLocalizedString("The microphone produced no audio — check the selected mic.", comment: "")
            return
        }
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            Task { @MainActor in
                self.request?.append(buffer)
                guard let ch = buffer.floatChannelData?[0] else { return }
                let n = Int(buffer.frameLength)
                var sum: Float = 0
                for i in stride(from: 0, to: n, by: 4) { sum += ch[i] * ch[i] }
                let rms = n > 0 ? (sum / Float(max(n / 4, 1))).squareRoot() : 0
                self.level = min(1, rms * 12)
            }
        }
        engine.prepare()
        do {
            try engine.start()
            engineRunning = true
        } catch {
            lastError = error.localizedDescription
        }
    }

    private func stopEngine() {
        guard engineRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engineRunning = false
        level = 0
    }
}

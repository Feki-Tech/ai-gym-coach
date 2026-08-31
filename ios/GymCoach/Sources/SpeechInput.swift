// Push-to-talk speech input for the coach (on-device Speech framework).
// Hold the mic button, speak, release: the transcript goes to the coach.

import Foundation
import AVFoundation
import Combine
import Speech

@MainActor
final class SpeechInput: ObservableObject {
    @Published private(set) var listening = false
    @Published private(set) var partial = ""
    @Published private(set) var level: Float = 0
    @Published var lastError: String?

    private let engine = AVAudioEngine()
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    init() {
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

    var isAvailable: Bool { recognizer?.isAvailable ?? false }

    func requestPermissions() async -> Bool {
        let speechOK: Bool = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
        }
        let micOK: Bool = await withCheckedContinuation { cont in
            AVAudioSession.sharedInstance().requestRecordPermission { cont.resume(returning: $0) }
        }
        if !speechOK || !micOK {
            lastError = NSLocalizedString("Microphone or speech recognition not allowed — enable both in Settings.", comment: "")
        }
        return speechOK && micOK
    }

    func start() {
        guard !listening, let recognizer, recognizer.isAvailable else { return }
        lastError = nil
        partial = ""
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .measurement,
                                    options: [.duckOthers, .defaultToSpeaker, .allowBluetooth])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            let req = SFSpeechAudioBufferRecognitionRequest()
            req.shouldReportPartialResults = true
            if #available(iOS 13, *) { req.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition }
            request = req
            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            input.removeTap(onBus: 0)
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
                req.append(buffer)
                guard let ch = buffer.floatChannelData?[0] else { return }
                let n = Int(buffer.frameLength)
                var sum: Float = 0
                for i in 0..<n { sum += ch[i] * ch[i] }
                let rms = n > 0 ? (sum / Float(n)).squareRoot() : 0
                Task { @MainActor in self?.level = min(1, rms * 12) }
            }
            engine.prepare()
            try engine.start()
            listening = true
            task = recognizer.recognitionTask(with: req) { [weak self] result, error in
                Task { @MainActor in
                    guard let self else { return }
                    if let r = result { self.partial = r.bestTranscription.formattedString }
                    if error != nil, self.listening { self.stopEngine() }
                }
            }
        } catch {
            lastError = error.localizedDescription
            stopEngine()
        }
    }

    /// Release: stop recording; the final transcript is what `partial` holds
    /// after the recognizer's last callback (a short grace period).
    func stop() async -> String {
        guard listening else { return "" }
        request?.endAudio()
        stopEngine()
        try? await Task.sleep(nanoseconds: 350_000_000)
        task?.cancel()
        task = nil
        let text = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        partial = ""
        try? AVAudioSession.sharedInstance().setCategory(.playback, options: [.mixWithOthers, .duckOthers])
        return text
    }

    private func stopEngine() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        listening = false
        level = 0
    }
}

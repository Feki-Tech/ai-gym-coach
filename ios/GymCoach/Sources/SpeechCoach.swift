// Voice coaching via AVSpeechSynthesizer, mixing politely with gym music.

import AVFoundation

final class SpeechCoach: NSObject, AVSpeechSynthesizerDelegate {
    private let synth = AVSpeechSynthesizer()
    private var pending = 0

    /// Voice matching the language the app is running in.
    private static let voice: AVSpeechSynthesisVoice? = {
        let lang = Bundle.main.preferredLocalizations.first ?? "en"
        let bcp47: String
        switch true {
        case lang.hasPrefix("zh"): bcp47 = "zh-CN"
        case lang.hasPrefix("hi"): bcp47 = "hi-IN"
        case lang.hasPrefix("es"): bcp47 = "es-ES"
        case lang.hasPrefix("fr"): bcp47 = "fr-FR"
        case lang.hasPrefix("ar"): bcp47 = "ar-SA"
        default: bcp47 = "en-US"
        }
        return AVSpeechSynthesisVoice(language: bcp47)
    }()

    override init() {
        super.init()
        synth.delegate = self
        try? AVAudioSession.sharedInstance().setCategory(
            .playback, options: [.mixWithOthers, .duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
    }

    func say(_ text: String) {
        guard pending < 2 else { return }        // drop cues if backlogged
        pending += 1
        let u = AVSpeechUtterance(string: text)
        u.rate = 0.52
        u.voice = Self.voice
        synth.speak(u)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didFinish utterance: AVSpeechUtterance) {
        pending = max(0, pending - 1)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didCancel utterance: AVSpeechUtterance) {
        pending = max(0, pending - 1)
    }
}

// Voice coaching via AVSpeechSynthesizer, mixing politely with gym music.

import AVFoundation

final class SpeechCoach: NSObject, AVSpeechSynthesizerDelegate {
    private let synth = AVSpeechSynthesizer()
    private var pending = 0

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

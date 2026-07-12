import SwiftUI
import UIKit
import AVFoundation
import CoachCore

@MainActor
final class WorkoutViewModel: ObservableObject {
    @Published var hud = HUDState()
    @Published var skeleton: Skeleton?
    @Published var bufferSize = CGSize(width: 720, height: 1280)
    @Published var summary: SessionRecord?

    let camera = CameraService()
    private let engine: SessionEngine
    private let speech = SpeechCoach()
    private let voiceOn: Bool
    private var t0 = Date()
    private var started = false

    init(exercise: String, voiceOn: Bool) {
        self.engine = SessionEngine(exercise: exercise)
        self.voiceOn = voiceOn
        self.hud = engine.hud
    }

    func start() {
        guard !started else { return }
        started = true
        t0 = Date()
        camera.onFrame = { [weak self] skel, size in
            DispatchQueue.main.async {
                guard let self else { return }
                self.bufferSize = size
                self.skeleton = skel
                guard let skel else { return }
                let t = Date().timeIntervalSince(self.t0)
                let out = self.engine.process(skel, t: t)
                self.hud = out.hud
                if self.voiceOn {
                    for cue in out.spokenCues { self.speech.say(cue) }
                }
            }
        }
        camera.start()
        if voiceOn { speech.say(NSLocalizedString("Ready. Let's go!", comment: "")) }
    }

    func endSet() {
        camera.stop()
        let rec = engine.finish(durationS: Date().timeIntervalSince(t0))
        try? WorkoutStore.documentsStore().append(rec)
        summary = rec
    }
}

struct WorkoutView: View {
    @StateObject private var vm: WorkoutViewModel
    @Environment(\.dismiss) private var dismiss

    init(exercise: String, voiceOn: Bool) {
        _vm = StateObject(wrappedValue: WorkoutViewModel(exercise: exercise,
                                                         voiceOn: voiceOn))
    }

    var body: some View {
        ZStack {
            CameraPreview(session: vm.camera.session)
                .ignoresSafeArea()
            SkeletonOverlay(skeleton: vm.skeleton, bufferSize: vm.bufferSize)
                .ignoresSafeArea()
            VStack {
                hudHeader
                Spacer()
                if !vm.hud.cue.isEmpty { cueBanner }
                endButton
            }
        }
        .navigationBarBackButtonHidden(true)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            vm.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            vm.camera.stop()
        }
        .sheet(item: $vm.summary) { rec in
            SummaryView(record: rec) {
                vm.summary = nil
                dismiss()
            }
            .interactiveDismissDisabled(true)
        }
    }

    private var hudHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            if vm.hud.detecting {
                Label("Detecting exercise…", systemImage: "wand.and.stars")
                    .font(.headline)
            } else if let ex = vm.hud.exercise {
                HStack {
                    Text(displayName(ex)).font(.headline)
                    Spacer()
                    Text(LocalizedStringKey("phase.\(vm.hud.phase)"))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                if let hold = vm.hud.plankHold {
                    Text(String(format: NSLocalizedString(
                        "Hold %.1f s   ·   best %.1f s", comment: ""),
                        hold, vm.hud.plankBest ?? 0))
                        .font(.title3.monospacedDigit())
                } else {
                    HStack(spacing: 16) {
                        Text("Reps \(vm.hud.repCount)")
                            .font(.title3.monospacedDigit()).bold()
                        if let s = vm.hud.lastScore {
                            Text("Score \(s)")
                                .font(.title3.monospacedDigit())
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.black.opacity(0.55),
                    in: RoundedRectangle(cornerRadius: 16))
        .foregroundColor(.white)
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var cueBanner: some View {
        Text(vm.hud.cue)
            .font(.headline)
            .foregroundColor(.black)
            .padding(.vertical, 10)
            .padding(.horizontal, 18)
            .background(.yellow.opacity(0.92), in: Capsule())
            .padding(.bottom, 10)
    }

    private var endButton: some View {
        Button {
            vm.endSet()
        } label: {
            Text("End set")
                .font(.headline)
                .foregroundColor(.white)
                .padding(.vertical, 14)
                .frame(maxWidth: .infinity)
                .background(.red, in: Capsule())
        }
        .padding(.horizontal, 40)
        .padding(.bottom, 24)
    }
}

/// Live camera preview backed by AVCaptureVideoPreviewLayer.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}

/// Skeleton drawn over the aspect-fill camera preview.
struct SkeletonOverlay: View {
    let skeleton: Skeleton?
    let bufferSize: CGSize

    var body: some View {
        Canvas { context, size in
            guard let skel = skeleton,
                  bufferSize.width > 0, bufferSize.height > 0 else { return }
            let scale = max(size.width / bufferSize.width,
                            size.height / bufferSize.height)
            let dw = bufferSize.width * scale
            let dh = bufferSize.height * scale
            let ox = (size.width - dw) / 2
            let oy = (size.height - dh) / 2
            func point(_ l: Landmark) -> CGPoint {
                CGPoint(x: ox + CGFloat(l.x) * dw, y: oy + CGFloat(l.y) * dh)
            }
            var path = Path()
            for (a, b) in skeletonEdges {
                let la = skel[a], lb = skel[b]
                if la.confidence > visMin && lb.confidence > visMin {
                    path.move(to: point(la))
                    path.addLine(to: point(lb))
                }
            }
            context.stroke(path, with: .color(.green.opacity(0.9)), lineWidth: 4)
            for l in skel where l.confidence > visMin {
                let p = point(l)
                context.fill(
                    Path(ellipseIn: CGRect(x: p.x - 5, y: p.y - 5,
                                           width: 10, height: 10)),
                    with: .color(.white))
            }
        }
        .allowsHitTesting(false)
    }
}

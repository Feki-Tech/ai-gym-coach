import SwiftUI
import UIKit
import AVFoundation
import Combine
import CoachCore

@MainActor
final class WorkoutViewModel: ObservableObject {
    @Published var hud = HUDState()
    @Published var skeleton: Skeleton?
    @Published var bufferSize = CGSize(width: 720, height: 1280)
    @Published var summary: SessionRecord?
    @Published var toast: String?
    @Published var showChat = false

    let camera = CameraService()
    let health = HealthService.shared
    let speech = SpeechCoach()
    let speechIn = SpeechInput()
    private(set) var engine: SessionEngine
    private(set) lazy var coach = CoachSession(
        speech: speech,
        engine: { [weak self] in self?.engine },
        state: { [weak self] in self?.liveState() ?? [:] })
    private let voiceOn: Bool
    private var t0 = Date()
    private var started = false
    private var lastSetReps: [RepRecord] = []
    private let history = WorkoutStore.documentsStore().load()
    private var bag = Set<AnyCancellable>()

    init(exercise: String, voiceOn: Bool) {
        self.engine = SessionEngine(exercise: exercise)
        self.voiceOn = voiceOn
        self.hud = engine.hud
        if exercise != "auto" {
            engine.bestReps = PersonalBests(history: history, exercise: exercise).reps
        }
        coach.onAction = { [weak self] _, note in self?.flash(note) }
        // the coach session and the speech input publish on their own; the
        // workout screen observes the view model, so forward their changes
        coach.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)
        speechIn.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)
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
                for ev in out.events { self.handle(ev) }
            }
        }
        camera.start()
        health.startLiveHeartRate()
        if voiceOn { speech.say(NSLocalizedString("Ready. Let's go!", comment: "")) }
        if coach.isAvailable {
            Task {                                   // greeting from last time
                if let brief = await coach.client.brief(exercise: engine.exercise) {
                    coach.notify("session_start", payload: brief)
                }
            }
        }
    }

    private func handle(_ ev: SessionEvent) {
        switch ev {
        case .goalReached:
            debriefSet()
        case .setDone:
            debriefSet()
        case .programAdvanced(let message, _, _):
            flash(message)
            speech.sayChat(message)
            if let ex = engine.exercise, ex != hud.exercise {
                engine.bestReps = PersonalBests(history: history, exercise: ex).reps
            }
        case .programDone:
            break
        case .restOver, .personalRecord:
            break
        }
    }

    /// Hand the finished set to the coach for a one-line spoken debrief.
    private func debriefSet() {
        let reps = engine.currentSetReps
        guard coach.isAvailable, !reps.isEmpty else { return }
        let scores = reps.map { $0.score }
        var faults: [String: Int] = [:]
        for r in reps { for f in r.faults { faults[f, default: 0] += 1 } }
        coach.notify("set_done", payload: [
            "exercise": engine.exercise ?? "",
            "reps": reps.count,
            "avg_score": Double(scores.reduce(0, +)) / Double(max(scores.count, 1)),
            "first_half_avg": Double(scores.prefix(max(1, scores.count / 2)).reduce(0, +)) / Double(max(1, scores.count / 2)),
            "second_half_avg": Double(scores.suffix(max(1, scores.count / 2)).reduce(0, +)) / Double(max(1, scores.count / 2)),
            "fault_counts": faults,
            "avg_concentric_s": reps.map { $0.concentricS }.reduce(0, +) / Double(reps.count),
        ])
    }

    func liveState() -> [String: Any] {
        var s: [String: Any] = [
            "exercise": engine.exercise as Any,
            "phase": hud.phase, "reps": hud.repCount,
            "fault_counts": Dictionary(grouping: engine.builder.reps.flatMap { $0.faults }, by: { $0 })
                .mapValues { $0.count },
            "coach_config": ["rep_goal": engine.config.repGoal as Any,
                             "rest_left_s": Int(hud.restLeft),
                             "tempo_ecc_target_s": engine.config.tempoEccTarget as Any,
                             "cues_on": engine.config.cuesOn,
                             "load_kg": engine.config.loadKg as Any,
                             "program": hud.program.map { "\($0.exercise) set \($0.set)/\($0.sets) (\($0.target))" } as Any],
            "platform": "ios",
        ]
        if let last = engine.builder.reps.last {
            s["last_rep"] = ["score": last.score, "ecc_s": last.eccentricS, "con_s": last.concentricS,
                             "faults": last.faults, "vel_deg_s": last.velocity as Any]
        }
        if let v = hud.signalValue, let ex = engine.exercise, let spec = specs[ex] {
            s["joint_angles_deg"] = [spec.signal.rawValue: (v * 10).rounded() / 10,
                                     "trunk_lean": ((hud.trunkLean ?? 0) * 10).rounded() / 10]
        }
        if let hr = health.heartRate {
            s["sensors"] = ["heart_rate": Int(hr), "hr_zone": health.heartRateZone as Any,
                            "hr_max_estimated": true]
        }
        if let hold = hud.plankHold { s["plank_hold_s"] = hold }
        return s
    }

    func flash(_ text: String) {
        toast = text
        Task { try? await Task.sleep(nanoseconds: 3_500_000_000); if toast == text { toast = nil } }
    }

    func toggleRest() {
        if hud.restLeft > 0 { engine.cancelRest() } else { engine.startRest(seconds: 60) }
    }

    func setLoad(_ kg: Double?) {
        engine.config.loadKg = kg
        flash(kg.map { String(format: NSLocalizedString("Logging %.1f kg per rep", comment: ""), $0) }
              ?? NSLocalizedString("Load cleared — bodyweight", comment: ""))
    }

    func endSet() {
        camera.stop()
        coach.interrupt()
        let end = Date()
        let hr = health.stopLiveHeartRate()
        var rec = engine.finish(durationS: end.timeIntervalSince(t0))
        if hr.avg != nil { rec = rec.withHeartRate(avg: hr.avg, peak: hr.peak) }
        let prs = PersonalBests(history: history, exercise: rec.exercise).records(in: rec)
        if !prs.isEmpty {
            rec = rec.withRecords(prs)
            speech.sayChat(NSLocalizedString("Personal record!", comment: "") + " " + prs.joined(separator: "; "))
        }
        try? WorkoutStore.documentsStore().append(rec)
        summary = rec
        if rec.summary.reps > 0 || rec.plank != nil {
            let start = t0
            Task { await health.saveWorkout(rec, start: start, end: end) }
            if coach.isAvailable {
                Task {
                    if let data = try? JSONEncoder().encode(rec) { _ = await coach.client.upload(recordJSON: data) }
                }
                var payload: [String: Any] = ["exercise": rec.exercise, "duration_s": rec.durationS,
                                              "reps": rec.summary.reps]
                if let s = rec.summary.avgScore { payload["avg_score"] = s }
                payload["fault_counts"] = rec.summary.faultCounts
                if let v = rec.summary.velocityLossPct { payload["velocity_loss_pct"] = v }
                if !prs.isEmpty { payload["prs"] = prs }
                coach.notify("session_done", payload: payload)
            }
        }
    }
}

struct WorkoutView: View {
    @StateObject private var vm: WorkoutViewModel
    @ObservedObject private var health = HealthService.shared
    @Environment(\.dismiss) private var dismiss
    @State private var askLoad = false
    @State private var loadText = ""

    init(exercise: String, voiceOn: Bool) {
        _vm = StateObject(wrappedValue: WorkoutViewModel(exercise: exercise, voiceOn: voiceOn))
    }

    var body: some View {
        ZStack {
            CameraPreview(session: vm.camera.session)
                .ignoresSafeArea()
            SkeletonOverlay(skeleton: vm.skeleton, bufferSize: vm.bufferSize,
                            faults: vm.hud.faultsNow, signal: currentSignal)
                .ignoresSafeArea()
            VStack(spacing: 8) {
                hudHeader
                if let p = vm.hud.program { programStrip(p) }
                HStack(alignment: .top) {
                    if vm.hud.thresholds != nil || vm.hud.plankHold != nil { gauge }
                    Spacer()
                }
                Spacer()
                if let t = vm.toast { toastView(t) }
                if !vm.hud.cue.isEmpty { cueBanner }
                coachBar
                bottomBar
            }
            if vm.hud.restLeft > 0 { restOverlay }
        }
        .navigationBarBackButtonHidden(true)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            vm.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            vm.camera.stop()
            vm.health.stopLiveHeartRate()
            vm.coach.interrupt()
        }
        .sheet(item: $vm.summary) { rec in
            SummaryView(record: rec) {
                vm.summary = nil
                dismiss()
            }
            .interactiveDismissDisabled(true)
        }
        .sheet(isPresented: $vm.showChat) {
            CoachChatView(session: vm.coach, speechIn: vm.speechIn)
                .presentationDetents([.medium, .large])
        }
        .alert("Load per rep (kg)", isPresented: $askLoad) {
            TextField("60", text: $loadText).keyboardType(.decimalPad)
            Button("Set") { vm.setLoad(Double(loadText.replacingOccurrences(of: ",", with: "."))) }
            Button("Bodyweight") { vm.setLoad(nil) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Bar + plates or dumbbells — for volume, estimated 1RM and records.")
        }
    }

    private var currentSignal: SignalKey? {
        vm.hud.exercise.flatMap { specs[$0]?.signal }
    }

    // MARK: header

    private var hudHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            if vm.hud.detecting {
                Label("Detecting exercise…", systemImage: "wand.and.stars").font(.headline)
                Text("Start moving — I'll recognise it").font(.caption).foregroundStyle(.secondary)
            } else if let ex = vm.hud.exercise {
                HStack {
                    Text(displayName(ex)).font(.headline)
                    Spacer()
                    if let hr = health.heartRate { heartRatePill(hr, zone: health.heartRateZone) }
                    phasePill
                }
                if let hold = vm.hud.plankHold {
                    Text(String(format: NSLocalizedString("Hold %.1f s   ·   best %.1f s", comment: ""),
                                hold, vm.hud.plankBest ?? 0))
                        .font(.title3.monospacedDigit())
                } else {
                    HStack(alignment: .firstTextBaseline, spacing: 14) {
                        if let goal = vm.hud.repGoal {
                            goalRing(vm.hud.repCount, goal)
                        }
                        Text("\(vm.hud.repCount)").font(.system(size: 34, weight: .bold, design: .rounded).monospacedDigit())
                        if let goal = vm.hud.repGoal { Text("/ \(goal)").font(.title3).foregroundStyle(.secondary) }
                        if let s = vm.hud.lastScore {
                            Text("Score \(s)").font(.title3.monospacedDigit()).foregroundStyle(scoreColor(s))
                        }
                        if let kg = vm.hud.loadKg { Text(String(format: "%.0f kg", kg)).font(.subheadline).foregroundStyle(.secondary) }
                        if vm.hud.bestReps > 0, vm.hud.repCount < vm.hud.bestReps {
                            Text(String(format: NSLocalizedString("record %lld", comment: ""), vm.hud.bestReps))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 16))
        .foregroundColor(.white)
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var phasePill: some View {
        let lift = vm.hud.concentricPhase == "ascent"
        let (label, color): (String, Color) = {
            switch vm.hud.phase {
            case "DESCENT": return (lift ? "LOWERING" : "LIFTING", lift ? .blue : .green)
            case "BOTTOM": return (lift ? "BOTTOM" : "TOP", .yellow)
            case "ASCENT": return (lift ? "LIFTING" : "LOWERING", lift ? .green : .blue)
            default: return ("READY", .gray)
            }
        }()
        return Text(LocalizedStringKey(label)).font(.caption2.bold())
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(color.opacity(0.85), in: Capsule()).foregroundColor(.black)
    }

    private func goalRing(_ reps: Int, _ goal: Int) -> some View {
        ZStack {
            Circle().stroke(.white.opacity(0.25), lineWidth: 4)
            Circle().trim(from: 0, to: min(1, CGFloat(reps) / CGFloat(max(goal, 1))))
                .stroke(reps >= goal ? Color.green : Color.blue, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 28, height: 28)
    }

    private func heartRatePill(_ hr: Double, zone: Int?) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "heart.fill")
            Text("\(Int(hr))").monospacedDigit()
            if let z = zone { Text(String(format: NSLocalizedString("Z%lld", comment: ""), z)).font(.caption2).bold() }
        }
        .font(.subheadline).padding(.horizontal, 8).padding(.vertical, 3)
        .background(zoneColor(zone).opacity(0.85), in: Capsule()).foregroundColor(.black)
    }

    private func zoneColor(_ zone: Int?) -> Color {
        switch zone { case 1: return .blue; case 2: return .green; case 3: return .yellow
                      case 4: return .orange; case 5: return .red; default: return .gray }
    }

    private func scoreColor(_ s: Int) -> Color { s >= 85 ? .green : s >= 65 ? .yellow : .red }

    private func programStrip(_ p: ProgramStatus) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "list.bullet.rectangle")
            Text(String(format: NSLocalizedString("Program · block %lld/%lld · %@ · set %lld/%lld · %@", comment: ""),
                        p.block, p.blocks, displayName(p.exercise), p.set, p.sets, p.target))
                .font(.caption.bold()).lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
        .background(.orange.opacity(0.85), in: Capsule()).foregroundColor(.black)
        .padding(.horizontal)
    }

    // MARK: range-of-motion gauge

    private var gauge: some View {
        let value = vm.hud.signalValue ?? 180
        let hold = vm.hud.plankHold != nil
        let top = 180.0
        let bottom = hold ? 120.0 : max(0, (vm.hud.thresholds?.bottom ?? 90) - 30)
        let frac = max(0, min(1, (value - bottom) / (top - bottom)))
        let color: Color = {
            if hold { return value >= 160 ? .green : .red }
            guard let th = vm.hud.thresholds else { return .gray }
            return value < th.bottom ? .green : value < th.start ? .yellow : .blue
        }()
        return VStack(alignment: .leading, spacing: 4) {
            Text((currentSignal?.rawValue ?? "body line").replacingOccurrences(of: "_", with: " ").uppercased())
                .font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "%.0f°", value)).font(.headline.monospacedDigit()).foregroundStyle(color)
            GeometryReader { g in
                ZStack(alignment: .top) {
                    Capsule().fill(.white.opacity(0.2))
                    Capsule().fill(color).frame(height: max(12, g.size.height * (hold ? frac : (1 - frac))))
                        .frame(maxHeight: .infinity, alignment: hold ? .bottom : .top)
                    ForEach(marks, id: \.0) { mark in
                        let y = g.size.height * (1 - (mark.1 - bottom) / (top - bottom))
                        HStack(spacing: 4) {
                            Rectangle().fill(.white).frame(width: 18, height: 2)
                            Text(mark.0).font(.system(size: 9)).foregroundStyle(.white.opacity(0.85)).fixedSize()
                        }
                        .offset(x: -3, y: y - 1)
                    }
                }
            }
            .frame(width: 12, height: 150)
        }
        .padding(10)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
        .foregroundColor(.white)
        .padding(.leading)
    }

    private var marks: [(String, Double)] {
        if vm.hud.plankHold != nil { return [(NSLocalizedString("straight", comment: ""), 160)] }
        guard let th = vm.hud.thresholds else { return [] }
        return [(NSLocalizedString("lockout", comment: ""), th.lockout),
                (NSLocalizedString("rep starts", comment: ""), th.start),
                (NSLocalizedString("full depth", comment: ""), th.bottom)]
    }

    // MARK: bottom

    private var cueBanner: some View {
        Text(vm.hud.cue)
            .font(.headline).foregroundColor(.black)
            .padding(.vertical, 10).padding(.horizontal, 18)
            .background(.yellow.opacity(0.92), in: Capsule())
    }

    private func toastView(_ t: String) -> some View {
        Text(t).font(.footnote).foregroundColor(.white)
            .padding(.vertical, 8).padding(.horizontal, 14)
            .background(.black.opacity(0.7), in: Capsule())
    }

    private var coachBar: some View {
        Group {
            if vm.coach.isAvailable, !vm.coach.lastReply.isEmpty || vm.coach.busy {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "bubble.left.fill").foregroundStyle(.purple)
                    Text(vm.coach.busy && vm.coach.lastReply.isEmpty ? NSLocalizedString("Coach is thinking…", comment: "") : vm.coach.lastReply)
                        .font(.footnote).lineLimit(3)
                    Spacer()
                }
                .padding(10)
                .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
                .foregroundColor(.white)
                .padding(.horizontal)
                .onTapGesture { vm.showChat = true }
            }
        }
    }

    private var bottomBar: some View {
        HStack(spacing: 10) {
            Button { vm.showChat = true } label: {
                Label(vm.coach.isAvailable ? "Talk" : "Coach", systemImage: "waveform.and.mic")
                    .font(.subheadline.bold()).padding(.vertical, 12).frame(maxWidth: .infinity)
                    .background(.purple.opacity(vm.coach.isAvailable ? 0.9 : 0.5), in: Capsule()).foregroundColor(.white)
            }
            Button { vm.toggleRest() } label: {
                Image(systemName: vm.hud.restLeft > 0 ? "timer.circle.fill" : "timer")
                    .font(.title2).padding(10).background(.black.opacity(0.55), in: Circle()).foregroundColor(.white)
            }
            .accessibilityLabel("Rest 60 s")
            Button { loadText = vm.hud.loadKg.map { String(format: "%g", $0) } ?? ""; askLoad = true } label: {
                Image(systemName: "scalemass").font(.title2).padding(10)
                    .background(.black.opacity(0.55), in: Circle()).foregroundColor(.white)
            }
            .accessibilityLabel("Load")
            Button { vm.endSet() } label: {
                Text("End set").font(.subheadline.bold()).foregroundColor(.white)
                    .padding(.vertical, 12).frame(maxWidth: .infinity)
                    .background(.red, in: Capsule())
            }
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 20)
    }

    private var restOverlay: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()
            VStack(spacing: 8) {
                Text("REST").font(.caption.bold()).foregroundStyle(.secondary)
                Text("\(Int(vm.hud.restLeft) + 1)").font(.system(size: 72, weight: .bold, design: .rounded).monospacedDigit())
                Text("seconds").font(.footnote).foregroundStyle(.secondary)
                if let next = vm.hud.restNext { Text(next).font(.headline).padding(.top, 6) }
                Button("Skip rest") { vm.engine.cancelRest() }.buttonStyle(.bordered).padding(.top, 8)
            }
            .foregroundColor(.white)
            .padding(28)
            .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 20))
        }
        .allowsHitTesting(true)
    }
}

/// Live camera preview backed by AVCaptureVideoPreviewLayer.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}

/// Body regions a form fault lights up (mirrors the desktop HUD).
private let faultRegions: [String: [String]] = [
    "back_lean": ["trunk"], "back_round": ["trunk", "neck"], "torso_lean": ["trunk"],
    "lean_back": ["trunk"], "body_sag": ["trunk", "legs"], "knees_cave": ["legs"],
    "shallow": ["driver"], "too_fast": ["driver"], "elbow_swing": ["arms"],
    "elbow_flare": ["arms"], "uneven": ["arms"], "chin": ["arms", "neck"], "shrug_neck": ["neck"],
]

private func regionEdges(_ region: String) -> [(Joint, Joint)] {
    switch region {
    case "trunk": return [(.leftShoulder, .leftHip), (.rightShoulder, .rightHip),
                          (.leftShoulder, .rightShoulder), (.leftHip, .rightHip)]
    case "legs": return [(.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
                         (.rightHip, .rightKnee), (.rightKnee, .rightAnkle)]
    case "arms": return [(.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
                         (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist)]
    case "neck": return [(.leftShoulder, .rightShoulder)]
    default: return []
    }
}

/// Skeleton drawn over the aspect-fill camera preview; faulty regions in red.
struct SkeletonOverlay: View {
    let skeleton: Skeleton?
    let bufferSize: CGSize
    var faults: [String] = []
    var signal: SignalKey? = nil

    var body: some View {
        Canvas { context, size in
            guard let skel = skeleton, bufferSize.width > 0, bufferSize.height > 0 else { return }
            let scale = max(size.width / bufferSize.width, size.height / bufferSize.height)
            let dw = bufferSize.width * scale, dh = bufferSize.height * scale
            let ox = (size.width - dw) / 2, oy = (size.height - dh) / 2
            func point(_ l: Landmark) -> CGPoint { CGPoint(x: ox + CGFloat(l.x) * dw, y: oy + CGFloat(l.y) * dh) }
            var hot = Set<String>()
            for f in faults {
                for region in faultRegions[f] ?? [] {
                    let r = region == "driver" ? (signal == .elbow ? "arms" : signal == .bodyLine ? "trunk" : "legs") : region
                    for (a, b) in regionEdges(r) { hot.insert("\(a.rawValue)-\(b.rawValue)") }
                }
            }
            var good = Path(), bad = Path()
            for (a, b) in skeletonEdges {
                let la = skel[a], lb = skel[b]
                guard la.confidence > visMin && lb.confidence > visMin else { continue }
                let key = "\(a.rawValue)-\(b.rawValue)", key2 = "\(b.rawValue)-\(a.rawValue)"
                if hot.contains(key) || hot.contains(key2) {
                    bad.move(to: point(la)); bad.addLine(to: point(lb))
                } else {
                    good.move(to: point(la)); good.addLine(to: point(lb))
                }
            }
            context.stroke(good, with: .color(.green.opacity(0.9)), lineWidth: 4)
            context.stroke(bad, with: .color(.red.opacity(0.5)), lineWidth: 10)
            context.stroke(bad, with: .color(.red), lineWidth: 4)
            for l in skel where l.confidence > visMin {
                let p = point(l)
                context.fill(Path(ellipseIn: CGRect(x: p.x - 5, y: p.y - 5, width: 10, height: 10)),
                             with: .color(.white))
            }
        }
        .allowsHitTesting(false)
    }
}

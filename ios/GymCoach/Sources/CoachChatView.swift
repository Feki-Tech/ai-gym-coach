import SwiftUI
import CoachCore

/// Talk to the coach: transcript, typed questions, push-to-talk mic.
struct CoachChatView: View {
    @ObservedObject var session: CoachSession
    @ObservedObject var speechIn: SpeechInput
    @ObservedObject private var client = CoachClient.shared
    @FocusState private var typing: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if !client.isConfigured {
                    notPaired
                } else {
                    transcript
                    composer
                }
            }
            .navigationTitle("Coach")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var notPaired: some View {
        VStack(spacing: 12) {
            Image(systemName: "bubble.left.and.bubble.right").font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("Pair with your coach server to talk").font(.headline)
            Text("On the PC: python coach_server.py — then enter its URL and pairing code under Coach settings.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            NavigationLink { CoachSettingsView() } label: { Label("Coach settings", systemImage: "gear") }
                .buttonStyle(.bordered)
        }
        .padding(28)
        .frame(maxHeight: .infinity)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if session.transcript.isEmpty {
                        Text("Ask anything — “why do my knees cave?”, “plan me a leg workout and start it”, “give me 90 seconds”, “I'm on 60 kilos”.")
                            .font(.footnote).foregroundStyle(.secondary).padding(.top, 8)
                    }
                    ForEach(session.transcript) { line in
                        bubble(line).id(line.id)
                    }
                    if session.busy {
                        HStack(spacing: 6) { ProgressView(); Text("Coach is thinking…").font(.caption).foregroundStyle(.secondary) }
                    }
                }
                .padding()
            }
            .onChange(of: session.transcript.count) { _ in
                if let last = session.transcript.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }

    private func bubble(_ line: ChatLine) -> some View {
        HStack {
            if line.role == .athlete { Spacer(minLength: 40) }
            Text(line.text)
                .padding(.vertical, 8).padding(.horizontal, 12)
                .background(line.role == .athlete ? Color.accentColor.opacity(0.25)
                            : line.role == .app ? Color.orange.opacity(0.18)
                            : Color.gray.opacity(0.18),
                            in: RoundedRectangle(cornerRadius: 14))
                .font(line.role == .app ? .footnote : .body)
            if line.role != .athlete { Spacer(minLength: 40) }
        }
    }

    private var composer: some View {
        VStack(spacing: 6) {
            if session.handsFreeOn {
                HStack(spacing: 8) {
                    Image(systemName: session.turnState == .speaking ? "speaker.wave.2.fill" : "mic.fill")
                        .foregroundStyle(turnColor)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(turnLabel).font(.caption.bold()).foregroundStyle(turnColor)
                        Text(speechIn.state == .hearing && !speechIn.partial.isEmpty
                             ? speechIn.partial : session.turnHint)
                            .font(.caption2).foregroundStyle(.secondary).lineLimit(2)
                    }
                    Spacer()
                    if speechIn.state == .listening || speechIn.state == .hearing {
                        Capsule().fill(turnColor).frame(width: CGFloat(6 + 40 * speechIn.level), height: 4)
                    }
                    micMenu
                }
                .padding(.horizontal)
            } else if speechIn.state == .holdRecording {
                HStack {
                    Image(systemName: "waveform").foregroundStyle(.red)
                    Text(speechIn.partial.isEmpty ? NSLocalizedString("Listening… release to send", comment: "") : speechIn.partial)
                        .font(.footnote).lineLimit(2)
                    Spacer()
                    Capsule().fill(.red).frame(width: CGFloat(20 + 80 * speechIn.level), height: 6)
                }
                .padding(.horizontal)
            }
            if let err = speechIn.lastError ?? client.lastError {
                Text(err).font(.caption2).foregroundStyle(.red).padding(.horizontal)
            }
            HStack(spacing: 10) {
                TextField("Ask your coach", text: $session.draft)
                    .textFieldStyle(.roundedBorder)
                    .focused($typing)
                    .submitLabel(.send)
                    .onSubmit(send)
                Button(action: send) { Image(systemName: "paperplane.fill") }
                    .disabled(session.draft.trimmingCharacters(in: .whitespaces).isEmpty)
                Image(systemName: speechIn.state == .holdRecording ? "mic.fill" : "mic")
                    .font(.title2)
                    .foregroundStyle(speechIn.state == .holdRecording ? .red : .accentColor)
                    .padding(8)
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { _ in if speechIn.state != .holdRecording { startTalking() } }
                            .onEnded { _ in Task { await stopTalking() } })
                    .accessibilityLabel("Hold to talk")
                if session.busy {
                    Button { session.interrupt() } label: { Image(systemName: "stop.circle") }
                }
            }
            .padding([.horizontal, .bottom])
        }
        .background(.thinMaterial)
    }

    private var turnLabel: String {
        switch session.turnState {
        case .listening: return NSLocalizedString("Listening", comment: "")
        case .hearing: return NSLocalizedString("Hearing you…", comment: "")
        case .thinking: return NSLocalizedString("Thinking…", comment: "")
        case .speaking: return NSLocalizedString("Coach speaking", comment: "")
        case .off: return NSLocalizedString("Mic off", comment: "")
        }
    }

    private var turnColor: Color {
        switch session.turnState {
        case .listening: return .green
        case .hearing: return .cyan
        case .thinking: return .orange
        case .speaking: return .purple
        case .off: return .secondary
        }
    }

    private func send() {
        let t = session.draft
        session.draft = ""
        typing = false
        session.ask(t)
    }

    private func startTalking() {
        session.interrupt()
        Task {
            if await speechIn.requestPermissions() { speechIn.startHold() }
        }
    }

    private func stopTalking() async {
        let text = await speechIn.stopHold()
        if !text.isEmpty { session.ask(text) }
    }

    private var micMenu: some View {
        Menu {
            Button {
                speechIn.preferredMicUID = ""
            } label: {
                speechIn.preferredMicUID.isEmpty
                    ? AnyView(Label("System default", systemImage: "checkmark"))
                    : AnyView(Text("System default"))
            }
            ForEach(speechIn.availableMics) { mic in
                Button {
                    speechIn.preferredMicUID = mic.id
                } label: {
                    mic.id == speechIn.preferredMicUID
                        ? AnyView(Label(mic.name, systemImage: "checkmark"))
                        : AnyView(Text(mic.name))
                }
            }
        } label: {
            Image(systemName: "mic.badge.plus").font(.subheadline)
        }
        .onTapGesture { speechIn.refreshMics() }
        .accessibilityLabel("Microphone")
    }
}

/// Where the desktop coach lives: URL + pairing code, connection test.
struct CoachSettingsView: View {
    @ObservedObject private var client = CoachClient.shared
    @StateObject private var mic = SpeechInput()
    @AppStorage("coach.fullDuplex") private var fullDuplex = true
    @AppStorage("coach.earcons") private var earcons = true
    @State private var checking = false
    @State private var url = ""
    @State private var code = ""

    var body: some View {
        Form {
            Section {
                TextField("http://192.168.1.20:7799", text: $url)
                    .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                TextField("Pairing code", text: $code)
                    .textInputAutocapitalization(.characters).autocorrectionDisabled()
                Button {
                    client.baseURL = url
                    client.token = code
                    checking = true
                    Task { await client.checkHealth(); checking = false }
                } label: {
                    HStack { Label("Save and test", systemImage: "antenna.radiowaves.left.and.right"); if checking { Spacer(); ProgressView() } }
                }
            } header: {
                Text("Coach server")
            } footer: {
                Text("On your PC, in the ai-gym-coach folder: `python coach_server.py` — it prints the URL and a pairing code. Phone and PC must be on the same Wi‑Fi. The LLM runs on the PC (Ollama by default); nothing goes to the cloud unless you configured a remote model there.")
            }
            Section("Status") {
                if let h = client.health {
                    Label(h.llm.reachable ? "Connected · \(h.llm.model)" : "Server reached, LLM not reachable: \(h.llm.error)",
                          systemImage: h.llm.reachable ? "checkmark.circle.fill" : "exclamationmark.triangle")
                        .foregroundStyle(h.llm.reachable ? .green : .orange)
                    Text("Knowledge: \(h.knowledge.chunks) notes, \(h.knowledge.exercises) exercises · prompt \(h.prompt_version)")
                        .font(.footnote).foregroundStyle(.secondary)
                } else if let err = client.lastError {
                    Label(err, systemImage: "xmark.octagon").foregroundStyle(.red)
                } else {
                    Text("Not paired yet.").foregroundStyle(.secondary)
                }
            }
            Section {
                Picker("Microphone", selection: $mic.preferredMicUID) {
                    Text("System default").tag("")
                    ForEach(mic.availableMics) { m in
                        Text(m.name).tag(m.id)
                    }
                }
                Button {
                    mic.refreshMics()
                } label: {
                    Label("Refresh microphones", systemImage: "arrow.clockwise")
                }
            } header: {
                Text("Microphone")
            } footer: {
                Text("Built-in, AirPods / Bluetooth headsets and wired or USB-C microphones. The camera is picked in the workout screen (🔄 button) — external USB-C cameras appear there on supported devices.")
            }
            Section {
                Toggle("Interrupt by voice", isOn: $fullDuplex)
                Toggle("Mic sounds", isOn: $earcons)
            } header: {
                Text("Talking over the coach")
            } footer: {
                Text("Interrupt by voice keeps the mic open while the coach talks (the phone cancels its own voice) — just talk over it. Turn it off if the coach keeps answering itself on a loud speaker: the mic then closes while it talks, it reads two sentences per answer and “more” fetches the rest. Mic sounds: a rising tone when the mic is yours, a tick when your sentence was captured.")
            }
            Section {
                Text("What the coach sees: your live set (exercise, phase, reps, last rep's score, tempo and faults), your history and profile on the PC, and its knowledge base (form faults, programming, recovery, nutrition, 870+ exercises). It can switch exercise, set a rep goal, start a rest, set tempo, mute cues, log the load and run a guided program — say it and it happens.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Coach settings")
        .onAppear { url = client.baseURL; code = client.token }
    }
}

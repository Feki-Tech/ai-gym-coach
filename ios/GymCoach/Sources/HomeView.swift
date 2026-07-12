import SwiftUI
import CoachCore

struct HomeView: View {
    @State private var voiceOn = true
    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 12)]

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 12) {
                    card("auto", title: "Auto-detect", icon: "wand.and.stars",
                         subtitle: "I'll recognize the movement")
                    ForEach(exerciseOrder, id: \.self) { ex in
                        card(ex, title: displayName(ex), icon: icon(for: ex),
                             subtitle: specs[ex]?.cameraHint ?? "")
                    }
                }
                .padding()

                Toggle(isOn: $voiceOn) {
                    Label("Voice coaching", systemImage: "speaker.wave.2.fill")
                }
                .padding(.horizontal)
                .padding(.bottom, 24)
            }
            .navigationTitle("AI Gym Coach")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    NavigationLink {
                        HistoryView()
                    } label: {
                        Image(systemName: "chart.xyaxis.line")
                    }
                }
            }
        }
    }

    private func card(_ exercise: String, title: String, icon: String,
                      subtitle: String) -> some View {
        NavigationLink {
            WorkoutView(exercise: exercise, voiceOn: voiceOn)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundStyle(.tint)
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2, reservesSpace: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(Color.gray.opacity(0.15),
                        in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }

    private func icon(for exercise: String) -> String {
        switch exercise {
        case "squat": return "figure.cross.training"
        case "pushup": return "figure.wrestling"
        case "bench": return "figure.strengthtraining.traditional"
        case "deadlift": return "figure.strengthtraining.functional"
        case "lunge": return "figure.walk"
        case "shoulder_press": return "figure.arms.open"
        case "curl": return "dumbbell.fill"
        case "pullup": return "figure.play"
        case "plank": return "figure.core.training"
        default: return "figure.mixed.cardio"
        }
    }
}

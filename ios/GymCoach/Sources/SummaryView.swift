import SwiftUI
import CoachCore

struct SummaryView: View {
    let record: SessionRecord
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            List {
                Section("Set") {
                    row("Exercise", displayName(record.exercise))
                    row("Duration", "\(Int(record.durationS)) s")
                    if let plank = record.plank {
                        row("Total hold", String(format: "%.1f s", plank.totalHoldS))
                        row("Best streak", String(format: "%.1f s", plank.bestStreakS))
                    } else {
                        row("Reps", "\(record.summary.reps)")
                        if let s = record.summary.avgScore {
                            row("Average score", "\(Int(s)) / 100")
                        }
                        if let v = record.summary.velocityLossPct {
                            row("Velocity loss", String(format: "%.0f %%", v))
                        }
                    }
                }
                if !record.summary.faultCounts.isEmpty {
                    Section("Focus points") {
                        ForEach(record.summary.faultCounts.sorted(by: { $0.value > $1.value }),
                                id: \.key) { item in
                            HStack {
                                Text(faultMessages[item.key]?.message ?? item.key)
                                Spacer()
                                Text("×\(item.value)")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } else if record.plank == nil, record.summary.reps > 0 {
                    Section {
                        Label("Clean set — no recurring faults!",
                              systemImage: "checkmark.seal.fill")
                            .foregroundStyle(.green)
                    }
                }
            }
            .navigationTitle("Set complete")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done", action: onClose)
                }
            }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
    }
}

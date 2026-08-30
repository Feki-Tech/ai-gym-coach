import SwiftUI
import CoachCore

struct SummaryView: View {
    let record: SessionRecord
    let onClose: () -> Void
    @ObservedObject private var health = HealthService.shared

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
                    if let load = record.summary.loadKg {
                        row("Load per rep", String(format: "%g kg", load))
                        row("Volume", String(format: "%g kg", record.summary.volumeKg ?? 0))
                        row("Estimated 1RM", String(format: "%.1f kg", record.summary.e1rmKg ?? 0))
                    }
                    if let hr = record.summary.avgHr {
                        row("Heart rate", String(format: NSLocalizedString(
                            "avg %lld · peak %lld bpm", comment: ""),
                            hr, record.summary.peakHr ?? hr))
                    }
                }
                if let prs = record.summary.prs, !prs.isEmpty {
                    Section("Personal records") {
                        ForEach(prs, id: \.self) { Label($0, systemImage: "trophy.fill").foregroundStyle(.orange) }
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
                if health.isAvailable {
                    Section("Apple Health") {
                        if health.enabled {
                            if let saved = health.lastSavedWorkout,
                               saved.timeIntervalSinceNow > -300 {
                                Label("Saved as a Strength Training workout",
                                      systemImage: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                            } else if let err = health.lastError {
                                Label(err, systemImage: "exclamationmark.triangle")
                                    .foregroundStyle(.orange)
                            } else if record.summary.reps > 0 || record.plank != nil {
                                Label("Saving to Apple Health…", systemImage: "heart.fill")
                                    .foregroundStyle(.secondary)
                            }
                            Button {
                                AppleAppLinks.open(AppleAppLinks.fitness)
                            } label: {
                                Label("Open the Fitness app", systemImage: "figure.run.circle")
                            }
                            Button {
                                AppleAppLinks.open(AppleAppLinks.health)
                            } label: {
                                Label("Open the Health app", systemImage: "heart.text.square")
                            }
                        } else {
                            NavigationLink {
                                HealthView()
                            } label: {
                                Label("Connect Apple Health to save workouts and heart rate",
                                      systemImage: "heart")
                            }
                        }
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

    private func row(_ label: LocalizedStringKey, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
    }
}

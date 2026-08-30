import SwiftUI
import CoachCore

/// Apple Health settings: what the coach reads, what it writes, and links
/// into the Health and Fitness apps. Nothing here leaves the phone.
struct HealthView: View {
    @ObservedObject private var health = HealthService.shared
    @State private var refreshing = false

    var body: some View {
        List {
            if !health.isAvailable {
                Section {
                    Label("Apple Health is not available on this device.",
                          systemImage: "heart.slash")
                        .foregroundStyle(.secondary)
                }
            } else {
                Section {
                    Toggle(isOn: $health.enabled) {
                        Label("Connect Apple Health", systemImage: "heart.fill")
                    }
                    .onChange(of: health.enabled) { on in
                        guard on else { return }
                        Task {
                            _ = await health.requestAuthorization()
                            await health.refreshSnapshot()
                        }
                    }
                    if let err = health.lastError {
                        Text(err).font(.footnote).foregroundStyle(.red)
                    }
                } footer: {
                    Text("Every finished set is saved to Health as a Strength Training workout (it counts toward your Fitness rings), and the coach reads the fitness data below to personalise cues, zones and rest. Data is read on this iPhone only — nothing is uploaded.")
                }

                if health.enabled {
                    Section {
                        if let s = health.snapshot {
                            row("Resting heart rate", s.restingHeartRate, "%.0f bpm")
                            row("Heart rate variability", s.heartRateVariabilityMs, "%.0f ms")
                            row("VO₂ max", s.vo2Max, "%.1f ml/kg·min")
                            row("Weight", s.bodyMassKg, "%.1f kg")
                            row("Height", s.heightCm, "%.0f cm")
                            if let age = s.age {
                                textRow("Age", "\(age)")
                                textRow("Estimated max heart rate",
                                        String(format: "%.0f bpm", s.estimatedMaxHeartRate ?? 0))
                            }
                            row("Sleep last night", s.sleepHoursLastNight, "%.1f h")
                            row("Steps today", s.stepsToday, "%.0f")
                            row("Active energy today", s.activeEnergyKcalToday, "%.0f kcal")
                            row("Exercise minutes today", s.exerciseMinutesToday, "%.0f min")
                            textRow("Workouts, last 7 days",
                                    String(format: NSLocalizedString("%lld · %.0f min", comment: ""),
                                           s.workoutsLast7Days, s.workoutMinutesLast7Days))
                        } else {
                            Text("No data yet — tap Refresh, or grant access in the Health app.")
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                        Button {
                            refreshing = true
                            Task {
                                await health.refreshSnapshot()
                                refreshing = false
                            }
                        } label: {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        .disabled(refreshing)
                    } header: {
                        Text("What the coach reads")
                    } footer: {
                        Text("Resting heart rate, HRV and sleep say how recovered you are; weight sets your protein target; age gives the heart-rate zones; steps, energy and this week's workouts show how much you have already done. Live heart rate during a set needs an Apple Watch workout or a strap that writes to Health.")
                    }

                    Section {
                        Label("One Strength Training workout per set, with reps, score and focus points in its details.",
                              systemImage: "figure.strengthtraining.traditional")
                            .font(.footnote)
                        if let saved = health.lastSavedWorkout {
                            textRow("Last saved", saved.formatted(date: .abbreviated, time: .shortened))
                        }
                    } header: {
                        Text("What the coach writes")
                    }
                }

                Section {
                    Button {
                        AppleAppLinks.open(AppleAppLinks.health)
                    } label: {
                        Label("Open the Health app", systemImage: "heart.text.square")
                    }
                    Button {
                        AppleAppLinks.open(AppleAppLinks.fitness)
                    } label: {
                        Label("Open the Fitness app", systemImage: "figure.run.circle")
                    }
                } footer: {
                    Text("Manage permissions in Health → Sharing → Apps → AI Gym Coach. Workouts appear in Fitness → Summary and Health → Activity → Workouts.")
                }
            }
        }
        .navigationTitle("Apple Health")
        .task {
            if health.enabled, health.snapshot == nil {
                await health.refreshSnapshot()
            }
        }
    }

    private func row(_ label: LocalizedStringKey, _ value: Double?, _ fmt: String) -> some View {
        textRow(label, value.map { String(format: fmt, $0) } ?? "—")
    }

    private func textRow(_ label: LocalizedStringKey, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
    }
}

import SwiftUI
import Charts
import CoachCore

struct HistoryView: View {
    @State private var sessions: [SessionRecord] = []
    @State private var selected = "squat"

    private var exercisesInLog: [String] {
        Array(Set(sessions.map(\.exercise))).sorted()
    }
    private var filtered: [SessionRecord] {
        sessions.filter { $0.exercise == selected }
    }
    private var scores: [Double] {
        filtered.compactMap { $0.summary.avgScore }
    }

    var body: some View {
        List {
            if sessions.isEmpty {
                ContentUnavailableCompat()
            } else {
                Section("Score trend") {
                    Picker("Exercise", selection: $selected) {
                        ForEach(exercisesInLog, id: \.self) {
                            Text(displayName($0)).tag($0)
                        }
                    }
                    .pickerStyle(.menu)

                    if scores.count >= 2 {
                        Chart(Array(scores.enumerated()), id: \.offset) { item in
                            LineMark(x: .value("Session", item.offset + 1),
                                     y: .value("Score", item.element))
                            PointMark(x: .value("Session", item.offset + 1),
                                      y: .value("Score", item.element))
                        }
                        .chartYScale(domain: 0...100)
                        .frame(height: 180)
                    } else {
                        Text("Complete two scored sessions of \(displayName(selected)) to see a trend.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Sessions") {
                    ForEach(sessions.reversed()) { rec in
                        row(rec)
                    }
                }
            }
        }
        .navigationTitle("Progress")
        .onAppear {
            sessions = WorkoutStore.documentsStore().load()
            if let last = sessions.last?.exercise { selected = last }
        }
    }

    private func row(_ rec: SessionRecord) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(displayName(rec.exercise)).font(.headline)
                Spacer()
                Text(rec.started)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 12) {
                if let plank = rec.plank {
                    Text(String(format: NSLocalizedString("hold %.0f s",
                                                          comment: ""),
                                plank.totalHoldS))
                    Text(String(format: NSLocalizedString("best %.0f s",
                                                          comment: ""),
                                plank.bestStreakS))
                } else {
                    Text("\(rec.summary.reps) reps")
                    if let s = rec.summary.avgScore {
                        Text("score \(Int(s))")
                    }
                    if let v = rec.summary.velocityLossPct {
                        Text(String(format: NSLocalizedString("vel. loss %.0f%%",
                                                              comment: ""), v))
                    }
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

/// Empty-state hint (kept iOS 16 compatible — ContentUnavailableView is 17+).
private struct ContentUnavailableCompat: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "figure.run.circle")
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            Text("No workouts yet")
                .font(.headline)
            Text("Finish a set and it will show up here.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
    }
}

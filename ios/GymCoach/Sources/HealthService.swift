// Apple Health integration — reads the athlete's fitness data for the coach,
// streams live heart rate during a set, and writes every finished set to
// Health as a strength-training workout (so it shows up in the Fitness app
// rings and history). Everything stays on the device: HealthKit data never
// leaves the phone, and the app has no server.

import Foundation
import Combine
import HealthKit
import UIKit
import CoachCore

/// What the coach can learn about the athlete from Apple Health.
struct HealthSnapshot: Equatable {
    var age: Int?
    var biologicalSex: String?
    var bodyMassKg: Double?
    var heightCm: Double?
    var restingHeartRate: Double?
    var heartRateVariabilityMs: Double?
    var vo2Max: Double?
    var sleepHoursLastNight: Double?
    var stepsToday: Double?
    var activeEnergyKcalToday: Double?
    var exerciseMinutesToday: Double?
    var workoutsLast7Days: Int = 0
    var workoutMinutesLast7Days: Double = 0
    var lastWorkout: Date?
    var fetched: Date?

    /// Max heart rate: measured never available from HealthKit directly, so
    /// the age formula (220 − age) is used — the coach says so when it matters.
    var estimatedMaxHeartRate: Double? {
        age.map { 220 - Double($0) }
    }
}

/// Deep links into Apple's own apps (documented URL schemes).
enum AppleAppLinks {
    static let health = URL(string: "x-apple-health://")!
    static let fitness = URL(string: "fitnessapp://")!

    static func open(_ url: URL) {
        UIApplication.shared.open(url, options: [:], completionHandler: nil)
    }
}

@MainActor
final class HealthService: ObservableObject {
    static let shared = HealthService()

    @Published private(set) var snapshot: HealthSnapshot?
    @Published private(set) var heartRate: Double?
    @Published private(set) var heartRateZone: Int?
    @Published private(set) var lastSavedWorkout: Date?
    @Published private(set) var lastError: String?
    @Published var enabled: Bool {
        didSet { UserDefaults.standard.set(enabled, forKey: Self.enabledKey) }
    }

    private static let enabledKey = "health.enabled"
    private let store = HKHealthStore()
    private var liveQuery: HKAnchoredObjectQuery?
    private var liveSamples: [Double] = []
    private var liveStart = Date()

    var isAvailable: Bool { HKHealthStore.isHealthDataAvailable() }

    private init() {
        enabled = UserDefaults.standard.bool(forKey: Self.enabledKey)
    }

    // MARK: - Types we ask for

    private static func quantity(_ id: HKQuantityTypeIdentifier) -> HKQuantityType {
        HKObjectType.quantityType(forIdentifier: id)!
    }

    /// Read set — every item has a coaching use (see docs/IOS.md §6).
    private var readTypes: Set<HKObjectType> {
        var types: Set<HKObjectType> = [
            Self.quantity(.heartRate),                 // live zones, avg/peak per set
            Self.quantity(.restingHeartRate),          // recovery / readiness
            Self.quantity(.heartRateVariabilitySDNN),  // readiness trend
            Self.quantity(.vo2Max),                    // conditioning level
            Self.quantity(.bodyMass),                  // protein target, relative strength
            Self.quantity(.height),
            Self.quantity(.stepCount),                 // daily activity context
            Self.quantity(.activeEnergyBurned),        // daily load
            Self.quantity(.appleExerciseTime),         // minutes already trained today
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,   // recovery
            HKObjectType.workoutType(),                // other training this week
        ]
        if let dob = HKObjectType.characteristicType(forIdentifier: .dateOfBirth) {
            types.insert(dob)                          // max-HR estimate, zones
        }
        if let sex = HKObjectType.characteristicType(forIdentifier: .biologicalSex) {
            types.insert(sex)
        }
        return types
    }

    /// Write set — the finished set as a workout (rings, Fitness history).
    private var shareTypes: Set<HKSampleType> {
        [HKObjectType.workoutType()]
    }

    // MARK: - Authorization

    /// Asks once; HealthKit never reveals read denials, so `true` only means
    /// the prompt was shown without error.
    func requestAuthorization() async -> Bool {
        guard isAvailable else { return false }
        do {
            try await store.requestAuthorization(toShare: shareTypes, read: readTypes)
            lastError = nil
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    // MARK: - Snapshot for the coach

    func refreshSnapshot() async {
        guard isAvailable, enabled else { return }
        var s = HealthSnapshot()
        let now = Date()
        let dayStart = Calendar.current.startOfDay(for: now)
        if let dob = try? store.dateOfBirthComponents().date {
            s.age = Calendar.current.dateComponents([.year], from: dob, to: now).year
        }
        if let sex = try? store.biologicalSex().biologicalSex {
            switch sex {
            case .female: s.biologicalSex = "female"
            case .male: s.biologicalSex = "male"
            case .other: s.biologicalSex = "other"
            default: break
            }
        }
        s.bodyMassKg = await latest(.bodyMass, unit: .gramUnit(with: .kilo))
        s.heightCm = await latest(.height, unit: .meterUnit(with: .centi))
        s.restingHeartRate = await latest(.restingHeartRate, unit: Self.bpm)
        s.heartRateVariabilityMs = await latest(.heartRateVariabilitySDNN,
                                               unit: .secondUnit(with: .milli))
        s.vo2Max = await latest(.vo2Max, unit: HKUnit(from: "ml/kg*min"))
        s.stepsToday = await sum(.stepCount, unit: .count(), from: dayStart)
        s.activeEnergyKcalToday = await sum(.activeEnergyBurned, unit: .kilocalorie(),
                                            from: dayStart)
        s.exerciseMinutesToday = await sum(.appleExerciseTime, unit: .minute(),
                                           from: dayStart)
        s.sleepHoursLastNight = await sleepHours(since: now.addingTimeInterval(-24 * 3600))
        let workouts = await workouts(since: now.addingTimeInterval(-7 * 24 * 3600))
        s.workoutsLast7Days = workouts.count
        s.workoutMinutesLast7Days = workouts.reduce(0) { $0 + $1.duration / 60 }
        s.lastWorkout = workouts.map(\.endDate).max()
        s.fetched = now
        snapshot = s
    }

    private static let bpm = HKUnit.count().unitDivided(by: .minute())

    private func latest(_ id: HKQuantityTypeIdentifier, unit: HKUnit) async -> Double? {
        await withCheckedContinuation { cont in
            let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
            let q = HKSampleQuery(sampleType: Self.quantity(id), predicate: nil, limit: 1,
                                  sortDescriptors: [sort]) { _, samples, _ in
                let v = (samples?.first as? HKQuantitySample)?.quantity.doubleValue(for: unit)
                cont.resume(returning: v)
            }
            store.execute(q)
        }
    }

    private func sum(_ id: HKQuantityTypeIdentifier, unit: HKUnit, from: Date) async -> Double? {
        await withCheckedContinuation { cont in
            let pred = HKQuery.predicateForSamples(withStart: from, end: nil, options: .strictStartDate)
            let q = HKStatisticsQuery(quantityType: Self.quantity(id), quantitySamplePredicate: pred,
                                      options: .cumulativeSum) { _, stats, _ in
                cont.resume(returning: stats?.sumQuantity()?.doubleValue(for: unit))
            }
            store.execute(q)
        }
    }

    private func sleepHours(since: Date) async -> Double? {
        await withCheckedContinuation { cont in
            let type = HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!
            let asleep = HKCategoryValueSleepAnalysis.predicateForSamples(
                equalTo: HKCategoryValueSleepAnalysis.allAsleepValues)
            let window = HKQuery.predicateForSamples(withStart: since, end: nil)
            let pred = NSCompoundPredicate(andPredicateWithSubpredicates: [asleep, window])
            let q = HKSampleQuery(sampleType: type, predicate: pred, limit: HKObjectQueryNoLimit,
                                  sortDescriptors: nil) { _, samples, _ in
                guard let samples, !samples.isEmpty else { return cont.resume(returning: nil) }
                let secs = samples.reduce(0.0) { $0 + $1.endDate.timeIntervalSince($1.startDate) }
                cont.resume(returning: secs / 3600)
            }
            store.execute(q)
        }
    }

    private func workouts(since: Date) async -> [HKWorkout] {
        await withCheckedContinuation { cont in
            let pred = HKQuery.predicateForSamples(withStart: since, end: nil)
            let q = HKSampleQuery(sampleType: HKObjectType.workoutType(), predicate: pred,
                                  limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, _ in
                cont.resume(returning: (samples as? [HKWorkout]) ?? [])
            }
            store.execute(q)
        }
    }

    // MARK: - Live heart rate during a set

    /// Heart-rate samples arrive from Apple Watch (continuously during a
    /// Watch workout, every few minutes otherwise) or from any strap that
    /// writes to Health. Zones use the age-based max from the snapshot.
    func startLiveHeartRate() {
        guard isAvailable, enabled, liveQuery == nil else { return }
        liveStart = Date()
        liveSamples = []
        heartRate = nil
        heartRateZone = nil
        let type = Self.quantity(.heartRate)
        let pred = HKQuery.predicateForSamples(withStart: liveStart.addingTimeInterval(-60),
                                               end: nil)
        let handler: (HKAnchoredObjectQuery, [HKSample]?, [HKDeletedObject]?,
                      HKQueryAnchor?, Error?) -> Void = { [weak self] _, samples, _, _, _ in
            guard let self, let samples = samples as? [HKQuantitySample], !samples.isEmpty
            else { return }
            let values = samples.sorted { $0.endDate < $1.endDate }
                .map { $0.quantity.doubleValue(for: Self.bpm) }
            Task { @MainActor in self.ingest(values) }
        }
        let q = HKAnchoredObjectQuery(type: type, predicate: pred, anchor: nil,
                                      limit: HKObjectQueryNoLimit, resultsHandler: handler)
        q.updateHandler = handler
        store.execute(q)
        liveQuery = q
    }

    private func ingest(_ values: [Double]) {
        liveSamples.append(contentsOf: values)
        guard let hr = values.last else { return }
        heartRate = hr
        let maxHR = snapshot?.estimatedMaxHeartRate ?? 190
        heartRateZone = Self.zone(hr, maxHR: maxHR)
    }

    /// Same bands as the desktop EffortModel: % of max → zone 1–5.
    static func zone(_ hr: Double, maxHR: Double) -> Int {
        let f = hr / max(maxHR, 1)
        if f < 0.6 { return 1 }
        if f < 0.7 { return 2 }
        if f < 0.8 { return 3 }
        if f < 0.9 { return 4 }
        return 5
    }

    /// Stops streaming; returns the set's average and peak (nil without data).
    @discardableResult
    func stopLiveHeartRate() -> (avg: Int?, peak: Int?) {
        if let q = liveQuery { store.stop(q) }
        liveQuery = nil
        let samples = liveSamples
        heartRate = nil
        heartRateZone = nil
        guard !samples.isEmpty else { return (nil, nil) }
        let avg = Int((samples.reduce(0, +) / Double(samples.count)).rounded())
        return (avg, Int(samples.max()!.rounded()))
    }

    // MARK: - Save a finished set as a workout

    /// One HKWorkout (traditional strength training, indoor) per set, with
    /// the coach's numbers in its metadata so they survive in Health.
    func saveWorkout(_ rec: SessionRecord, start: Date, end: Date) async {
        guard isAvailable, enabled else { return }
        let config = HKWorkoutConfiguration()
        config.activityType = .traditionalStrengthTraining
        config.locationType = .indoor
        let builder = HKWorkoutBuilder(healthStore: store, configuration: config,
                                       device: .local())
        var meta: [String: Any] = [
            HKMetadataKeyIndoorWorkout: true,
            "tech.fekitech.gymcoach.exercise": rec.exercise,
            "tech.fekitech.gymcoach.reps": rec.summary.reps,
        ]
        if let s = rec.summary.avgScore { meta["tech.fekitech.gymcoach.avg_score"] = s }
        if let p = rec.plank { meta["tech.fekitech.gymcoach.hold_s"] = p.totalHoldS }
        if !rec.summary.faultCounts.isEmpty {
            meta["tech.fekitech.gymcoach.faults"] = rec.summary.faultCounts
                .map { "\($0.key)x\($0.value)" }.sorted().joined(separator: ",")
        }
        do {
            try await builder.beginCollection(at: start)
            try await builder.addMetadata(meta)
            try await builder.endCollection(at: max(end, start.addingTimeInterval(1)))
            _ = try await builder.finishWorkout()
            lastSavedWorkout = Date()
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }
}

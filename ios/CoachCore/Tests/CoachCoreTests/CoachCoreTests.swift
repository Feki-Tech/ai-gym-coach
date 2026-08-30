// CoachCore unit tests — mirrors the desktop prototype's selftest suite.

import XCTest
@testable import CoachCore

final class CoachCoreTests: XCTestCase {

    // 1) joint angle sanity
    func testJointAngle() {
        XCTAssertEqual(jointAngle(P2(0, 1), P2(0, 0), P2(1, 0)), 90, accuracy: 1e-6)
        XCTAssertEqual(jointAngle(P2(0, 1), P2(0, 0), P2(0, 2)), 0, accuracy: 1e-6)
        XCTAssertEqual(jointAngle(P2(-1, 0), P2(0, 0), P2(1, 0)), 180, accuracy: 1e-6)
        // vertical segment => 0° lean (y-down coords: top has smaller y)
        XCTAssertEqual(segmentVsVertical(top: P2(0, 0), bottom: P2(0, 1)), 0, accuracy: 1e-6)
        XCTAssertEqual(segmentVsVertical(top: P2(1, 1), bottom: P2(0, 1)), 90, accuracy: 1e-6)
    }

    // 1b) heart-rate fields share the desktop log's keys and stay optional
    func testSessionSummaryHeartRateRoundTrip() throws {
        let builder = SessionBuilder()
        let rec = builder.finish(exercise: "squat", durationS: 42)
        XCTAssertNil(rec.summary.avgHr)
        let withHr = rec.withHeartRate(avg: 131, peak: 158)
        let data = try JSONEncoder().encode(withHr)
        let json = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(json.contains("\"avg_hr\":131") && json.contains("\"peak_hr\":158"))
        let back = try JSONDecoder().decode(SessionRecord.self, from: data)
        XCTAssertEqual(back.summary.avgHr, 131)
        XCTAssertEqual(back.summary.peakHr, 158)
        // records written before the field existed still decode
        let legacy = try JSONDecoder().decode(SessionRecord.self, from: try JSONEncoder().encode(rec))
        XCTAssertNil(legacy.summary.peakHr)
    }

    // 1c) programs parse like the desktop and advance set by set
    func testWorkoutProgram() throws {
        let p = try WorkoutProgram.parse("squat 2x3 rest 5, push-up 1x2 rest 4, plank 1x5s")
        XCTAssertEqual(p.blocks.count, 3)
        XCTAssertEqual(p.blocks[0], ProgramBlock(exercise: "squat", sets: 2, reps: 3, holdS: nil, restS: 5))
        XCTAssertEqual(p.blocks[1].exercise, "pushup")
        XCTAssertEqual(p.blocks[2].holdS, 5)
        XCTAssertEqual(p.status?.block, 1)
        var r = p.onSetDone()
        XCTAssertEqual(r.step, .same); XCTAssertEqual(r.restS, 5); XCTAssertEqual(p.status?.set, 2)
        r = p.onSetDone()
        XCTAssertEqual(r.step, .next); XCTAssertEqual(p.status?.exercise, "pushup")
        r = p.onSetDone()
        XCTAssertEqual(r.step, .next); XCTAssertEqual(p.status?.exercise, "plank")
        r = p.onSetDone()
        XCTAssertEqual(r.step, .done); XCTAssertNil(p.current)
        XCTAssertThrowsError(try WorkoutProgram.parse("yoga 3x10"))
        XCTAssertThrowsError(try WorkoutProgram.parse("squat 30x10"))
        XCTAssertThrowsError(try WorkoutProgram.parse(""))
        XCTAssertEqual(normalizeExercise("Overhead Press"), "shoulder_press")
    }

    // 1d) load → volume / e1RM, personal records
    func testLoadAndRecords() throws {
        let b = SessionBuilder()
        for i in 1...5 {
            b.addRep(RepEvent(count: i, duration: 2, eccentricS: 1, concentricS: 1,
                              minAngle: 80, fullDepth: true), velocity: 30, loadKg: 60)
        }
        let rec = b.finish(exercise: "squat", durationS: 30)
        XCTAssertEqual(rec.summary.volumeKg, 300)
        XCTAssertEqual(rec.summary.e1rmKg!, 70, accuracy: 0.01)      // 60 × (1 + 5/30)
        XCTAssertEqual(rec.reps[0].loadKg, 60)
        let json = String(decoding: try JSONEncoder().encode(rec), as: UTF8.self)
        XCTAssertTrue(json.contains("\"e1rm_kg\"") && json.contains("\"load_kg\":60"))
        var bests = PersonalBests(history: [], exercise: "squat")
        XCTAssertEqual(bests.records(in: rec), [])                      // first session: no PRs
        bests = PersonalBests(history: [rec], exercise: "squat")
        XCTAssertEqual(bests.reps, 5); XCTAssertEqual(bests.e1rmKg, 70)
        let b2 = SessionBuilder()
        for i in 1...6 {
            b2.addRep(RepEvent(count: i, duration: 2, eccentricS: 1, concentricS: 1,
                               minAngle: 80, fullDepth: true), velocity: 30, loadKg: 60)
        }
        let rec2 = b2.finish(exercise: "squat", durationS: 30)
        let prs = bests.records(in: rec2)
        XCTAssertEqual(prs.count, 2)                                    // reps 6 > 5, e1RM 72 > 70
        XCTAssertEqual(rec2.withRecords(prs).summary.prs?.count, 2)
    }

    // 2) One Euro reduces jitter on a noisy static hold
    func testOneEuroReducesJitter() {
        var rng = SystemRandomNumberGenerator()
        let f = OneEuroFilter()
        var rawPrev = 0.0, filtPrev = 0.0
        var rawJitter = 0.0, filtJitter = 0.0, filtErr = 0.0
        for i in 0..<200 {
            let t = Double(i) / 30.0
            let noise = Double.random(in: -0.05...0.05, using: &rng)
            let raw = 1.0 + noise
            let filt = f.filter(raw, t: t)
            if i > 10 {
                rawJitter += abs(raw - rawPrev)
                filtJitter += abs(filt - filtPrev)
                filtErr = max(filtErr, abs(filt - 1.0))
            }
            rawPrev = raw
            filtPrev = filt
        }
        XCTAssertLessThan(filtJitter, rawJitter * 0.6, "filter should cut jitter")
        XCTAssertLessThan(filtErr, 0.06, "filtered static value should stay near truth")
    }

    // 3) FSM counts 5 synthetic squat reps
    func testFSMCountsSquats() {
        let counter = RepCounter(spec: specs["squat"]!)
        var reps = 0
        for i in 0..<(5 * 90) {
            let t = Double(i) / 30.0
            let angle = 130 + 45 * cos(2 * .pi * t / 3)   // 85..175, 3 s period
            if counter.update(angle: angle, t: t) != nil { reps += 1 }
        }
        XCTAssertEqual(reps, 5)
    }

    // 4) shallow rep flagged
    func testShallowRepFlagged() {
        let spec = specs["squat"]!
        let counter = RepCounter(spec: spec)
        var ev: RepEvent?
        for i in 0..<180 {
            let t = Double(i) / 30.0
            let angle = 140 + 32 * cos(2 * .pi * t / 3)   // 108..172: never below 100
            if let e = counter.update(angle: angle, t: t) { ev = e }
        }
        let e = try! XCTUnwrap(ev)
        XCTAssertFalse(e.fullDepth)
        XCTAssertTrue(repFaults(spec: spec, ev: e).contains("shallow"))
    }

    // 5) concentric-first FSM (curl) maps tempo correctly
    func testCurlTempoMapping() {
        let spec = specs["curl"]!
        let counter = RepCounter(spec: spec)
        var ev: RepEvent?
        var t = 0.0
        func seg(from: Double, to: Double, seconds: Double) {
            let steps = Int(seconds * 30)
            for k in 0..<steps {
                t += 1.0 / 30
                let a = from + (to - from) * Double(k) / Double(steps - 1)
                if let e = counter.update(angle: a, t: t) { ev = e }
            }
        }
        seg(from: 170, to: 60, seconds: 0.6)   // fast lift (angle down)
        seg(from: 60, to: 170, seconds: 1.5)   // slow lower (angle up)
        let e = try! XCTUnwrap(ev)
        XCTAssertLessThan(e.concentricS, e.eccentricS,
                          "curl lift is the descending-angle phase")
        XCTAssertLessThan(abs(e.concentricS - 0.6), 0.25)
    }

    // 6) plank tracker accumulates hold and fires cue once
    func testPlankTracker() {
        let p = PlankTracker()
        var cues = 0
        for i in 0..<(12 * 30) {
            let t = Double(i) / 30.0
            let line = (t >= 5 && t < 8) ? 150.0 : 172.0   // 3 s sag
            if p.update(bodyLine: line, t: t) { cues += 1 }
        }
        XCTAssertEqual(cues, 1)
        XCTAssertGreaterThan(p.total, 8)
        XCTAssertGreaterThanOrEqual(p.best, 4.5)
    }

    // 7) all rep specs have ordered thresholds
    func testSpecThresholdsOrdered() {
        for sp in specs.values where sp.mode == .reps {
            XCTAssertLessThan(sp.bottomBelow, sp.startBelow, sp.name)
            XCTAssertLessThan(sp.startBelow, sp.lockoutAbove + 1, sp.name)
            XCTAssertTrue(["ascent", "descent"].contains(sp.concentricPhase))
        }
        XCTAssertEqual(specs.count, 9)
        XCTAssertEqual(Set(exerciseOrder), Set(specs.keys))
    }

    // 8) session builder summary + velocity loss + JSON roundtrip
    func testSessionBuilderAndStore() throws {
        let b = SessionBuilder()
        for (i, v) in [100.0, 100, 100, 70, 60].enumerated() {
            b.addRep(RepEvent(count: i + 1, duration: 3, eccentricS: 2,
                              concentricS: 1, minAngle: 85, fullDepth: true,
                              faults: i == 0 ? ["too_fast"] : [], score: 90),
                     velocity: v)
        }
        let rec = b.finish(exercise: "squat", durationS: 60)
        XCTAssertEqual(rec.summary.reps, 5)
        XCTAssertEqual(rec.summary.velocityLossPct ?? -1, 35.0, accuracy: 0.01)
        XCTAssertEqual(rec.summary.faultCounts["too_fast"], 1)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir,
                                                withIntermediateDirectories: true)
        let store = WorkoutStore(url: dir.appendingPathComponent("log.json"))
        try store.append(rec)
        try store.append(rec)
        let history = store.load()
        XCTAssertEqual(history.count, 2)
        XCTAssertEqual(history[1].exercise, "squat")
        XCTAssertEqual(history[1].reps.count, 5)
    }

    // 9) fatigue monitor warns once at >20% velocity loss
    func testFatigueMonitor() {
        let fm = FatigueMonitor()
        let fired = [10.0, 10, 10, 9, 8, 7, 5].map { fm.add($0) }
        XCTAssertEqual(fired, [false, false, false, false, false, true, false])
        XCTAssertGreaterThan(fm.loss, 0.2)
    }

    // 10) auto-detector classifies 8 synthetic movements
    func testAutoDetector() {
        func synth(_ expected: String,
                   _ make: @escaping (Double) -> FrameFeatures) {
            let ad = AutoDetector()
            for i in 0..<150 {
                let t = Double(i) / 30.0
                if let det = ad.update(make(t), t: t) {
                    XCTAssertEqual(det, expected, "\(expected) misread as \(det)")
                    return
                }
            }
            XCTFail("\(expected) never detected")
        }
        func osc(_ lo: Double, _ hi: Double, _ t: Double) -> Double {
            (lo + hi) / 2 + (hi - lo) / 2 * cos(2 * .pi * t / 2.0)
        }
        synth("squat") { t in
            FrameFeatures(trunk: osc(5, 35, t), knee: osc(80, 170, t),
                          hip: osc(90, 170, t))
        }
        synth("pushup") { t in
            FrameFeatures(trunk: 75, elbow: osc(90, 160, t))
        }
        synth("plank") { _ in FrameFeatures(trunk: 75) }
        synth("pullup") { t in
            FrameFeatures(elbow: osc(60, 160, t), shoY: osc(0.3, 0.6, t),
                          wriY: 0.1, overhead: true)
        }
        synth("shoulder_press") { t in
            FrameFeatures(elbow: osc(90, 170, t), wriY: osc(0.05, 0.25, t),
                          overhead: true)
        }
        synth("deadlift") { t in
            FrameFeatures(trunk: osc(10, 70, t), knee: osc(120, 170, t),
                          hip: osc(90, 170, t))
        }
        synth("lunge") { t in
            FrameFeatures(knee: osc(90, 170, t), kneeSplit: osc(0.1, 0.5, t))
        }
        synth("curl") { t in
            FrameFeatures(elbow: osc(60, 160, t))
        }
    }

    // 11) live rules fire on bad squat form
    func testLiveRules() {
        var ang = BodyAngles(side: "L", knee: 90, hip: 90, elbow: 170,
                             trunkLean: 60, upperArmSwing: 0, bodyLine: 170,
                             elbowFlare: 40, neck: 170, valgusRatio: 0.5,
                             wristYDiff: 0, noseAboveWrists: 1)
        let f = liveFaults(exercise: "squat", ang: ang, state: .bottom)
        XCTAssertTrue(f.contains("back_lean"))
        XCTAssertTrue(f.contains("knees_cave"))
        XCTAssertTrue(liveFaults(exercise: "squat", ang: ang, state: .idle).isEmpty,
                      "rules are phase-gated")
        ang.trunkLean = 10
        ang.valgusRatio = 1.0
        XCTAssertTrue(liveFaults(exercise: "squat", ang: ang, state: .bottom).isEmpty)
    }

    // 12) feedback engine rate-limits and prioritizes
    func testFeedbackEngine() {
        let fe = FeedbackEngine(cooldown: 3)
        XCTAssertEqual(fe.push(["too_fast", "back_lean"], t: 0),
                       faultMessages["back_lean"]?.message,
                       "higher priority first")
        XCTAssertEqual(fe.push(["back_lean"], t: 1), nil, "cooldown suppresses")
        XCTAssertNotNil(fe.push(["back_lean"], t: 4))
        XCTAssertEqual(scoreRep(RepEvent(count: 1, duration: 3, eccentricS: 2,
                                         concentricS: 1, minAngle: 80,
                                         fullDepth: true,
                                         faults: ["back_lean", "too_fast"])),
                       60)
    }

    // 13) end-to-end: SessionEngine counts synthetic squats from skeletons
    func testSessionEngineEndToEnd() {
        let engine = SessionEngine(exercise: "squat")
        var lastOut: FrameOutput?
        for i in 0..<(3 * 90) {
            let t = Double(i) / 30.0
            let kneeAngle = (130 + 45 * cos(2 * .pi * t / 3)) * .pi / 180
            lastOut = engine.process(syntheticSquatSkeleton(kneeRad: kneeAngle), t: t)
        }
        let out = try! XCTUnwrap(lastOut)
        XCTAssertGreaterThanOrEqual(out.hud.repCount, 2,
                                    "should count squat reps from skeletons")
        let rec = engine.finish(durationS: 9)
        XCTAssertEqual(rec.exercise, "squat")
        XCTAssertEqual(rec.summary.reps, out.hud.repCount)
    }

    /// Side-view squat skeleton with the requested knee angle (hip-knee-ankle).
    private func syntheticSquatSkeleton(kneeRad: Double) -> Skeleton {
        var pts = [Landmark](repeating: Landmark(x: 0, y: 0, confidence: 0.9),
                             count: Joint.allCases.count)
        let ankle = P2(0.5, 0.9)
        let shin = 0.18, thigh = 0.18, torso = 0.25
        let knee = P2(ankle.x, ankle.y - shin)              // shin vertical
        // thigh direction: rotate "up" by (180 - kneeDeg) around the knee
        let a = .pi - kneeRad
        let hip = P2(knee.x + thigh * sin(a), knee.y - thigh * cos(a))
        let sho = P2(hip.x, hip.y - torso)                  // upright trunk
        func set(_ j: Joint, _ p: P2) {
            pts[j.rawValue] = Landmark(x: p.x, y: p.y, confidence: 0.9)
        }
        set(.leftAnkle, ankle); set(.rightAnkle, P2(ankle.x + 0.12, ankle.y))
        set(.leftKnee, knee); set(.rightKnee, P2(knee.x + 0.12, knee.y))
        set(.leftHip, hip); set(.rightHip, P2(hip.x + 0.10, hip.y))
        set(.leftShoulder, sho); set(.rightShoulder, P2(sho.x + 0.10, sho.y))
        set(.leftElbow, P2(sho.x, sho.y + 0.12)); set(.rightElbow, P2(sho.x + 0.1, sho.y + 0.12))
        set(.leftWrist, P2(sho.x, sho.y + 0.24)); set(.rightWrist, P2(sho.x + 0.1, sho.y + 0.24))
        set(.nose, P2(sho.x, sho.y - 0.10))
        set(.leftEar, P2(sho.x - 0.02, sho.y - 0.08)); set(.rightEar, P2(sho.x + 0.02, sho.y - 0.08))
        return pts
    }
}

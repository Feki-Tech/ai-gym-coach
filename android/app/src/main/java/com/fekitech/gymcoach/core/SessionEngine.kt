// SessionEngine — platform-independent per-frame orchestrator.
// Feed it raw skeletons; it handles smoothing, auto-detection, rep counting,
// plank tracking, live/rep faults, fatigue, feedback and the session log.
// The UI layer only renders `HUDState` and speaks `spokenCues`.
package com.fekitech.gymcoach.core

import kotlin.math.max

data class HUDState(
    var exercise: String? = null,     // null while auto-detect is searching
    var phase: String = "IDLE",
    var repCount: Int = 0,
    var lastScore: Int? = null,
    var cue: String = "",             // on-screen coaching line
    var plankHold: Double? = null,
    var plankBest: Double? = null,
    var signalValue: Double? = null,
    var trunkLean: Double? = null,
    var detecting: Boolean = false,
)

data class FrameOutput(
    val hud: HUDState,
    val spokenCues: List<String>,     // hand these to TTS
    val repEvent: RepEvent?,
)

class SessionEngine(exercise: String) {
    var exercise: String? = null
        private set
    private var spec: ExerciseSpec? = null
    private val detector: AutoDetector?
    private val smoother = SkeletonSmoother()
    private val feedback = FeedbackEngine()
    private var counter: RepCounter? = null
    private var plank: PlankTracker? = null
    private val fatigue = FatigueMonitor()
    val builder = SessionBuilder()
    val hud = HUDState()

    init {
        if (exercise == "auto") {
            detector = AutoDetector()
            hud.detecting = true
        } else {
            detector = null
            this.exercise = exercise
            spec = specs[exercise]
            counter = specs[exercise]?.let { RepCounter(it) }
            if (specs[exercise]?.mode == ExerciseMode.HOLD) plank = PlankTracker()
        }
    }

    fun process(raw: Skeleton, t: Double): FrameOutput {
        val spoken = mutableListOf<String>()
        var repEvent: RepEvent? = null
        val pts = smoother.update(raw, t)
        val ang = bodyAngles(pts)

        val det = detector
        if (spec == null && det != null) {                   // auto-detect
            hud.detecting = true
            val found = det.update(frameFeatures(ang, pts), t)
            if (found != null) {
                exercise = found
                spec = specs[found]
                counter = specs[found]?.let { RepCounter(it) }
                if (specs[found]?.mode == ExerciseMode.HOLD) plank = PlankTracker()
                hud.exercise = found
                hud.detecting = false
                spoken.add("Detected ${displayName(found)}")
            }
        } else {
            val sp = spec
            val ctr = counter
            if (sp != null && ctr != null) {
                hud.exercise = exercise
                hud.detecting = false
                val pl = plank
                if (pl != null) {                            // timed hold
                    val faultsNow = liveFaults(sp.name, ang, ctr.state)
                        .toMutableList()
                    if (pl.update(bodyLine = ang.bodyLine, t = t)) {
                        faultsNow.add("body_sag")
                    }
                    feedback.push(faultsNow, t)?.let { spoken.add(it) }
                    hud.plankHold = pl.total
                    hud.plankBest = pl.best
                } else {                                     // rep exercise
                    val faultsNow = liveFaults(sp.name, ang, ctr.state)
                    for (f in faultsNow) ctr.noteFault(f)
                    val ev0 = ctr.update(angle = ang.value(sp.signal), t = t)
                    feedback.push(faultsNow, t)?.let { spoken.add(it) }
                    if (ev0 != null) {
                        val ev = ev0.copy(
                            faults = (ev0.faults.toSet() +
                                repFaults(sp, ev0)).sorted()
                        )
                        ev.score = scoreRep(ev)
                        hud.lastScore = ev.score
                        // concentric velocity proxy: ROM (deg) / lift time (s)
                        val vel = max(sp.lockoutAbove - ev.minAngle, 1.0) /
                            max(ev.concentricS, 0.05)
                        builder.addRep(ev, velocity = vel)
                        if (fatigue.add(vel)) {
                            feedback.current = FATIGUE_MESSAGE
                            spoken.add(FATIGUE_MESSAGE)
                        } else if (ev.faults.isEmpty()) {
                            spoken.add("${ev.count}. ${feedback.praise()}")
                        } else {
                            val cue = feedback.push(ev.faults, t)
                            spoken.add(if (cue != null) "${ev.count}. $cue"
                                       else "${ev.count}.")
                        }
                        repEvent = ev
                    }
                    hud.repCount = ctr.count
                }
                hud.phase = ctr.state.label
                hud.signalValue = ang.value(sp.signal)
                hud.trunkLean = ang.trunkLean
            }
        }

        hud.cue = feedback.current
        return FrameOutput(hud = hud, spokenCues = spoken, repEvent = repEvent)
    }

    /** Build the final session record (call once at the end). */
    fun finish(durationS: Double, started: String): SessionRecord =
        builder.finish(exercise = exercise ?: "auto", durationS = durationS,
                       started = started, plank = plank)
}

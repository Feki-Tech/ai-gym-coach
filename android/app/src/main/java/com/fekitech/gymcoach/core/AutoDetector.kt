// Rule-based auto exercise detection + velocity-loss fatigue monitor.
package com.fekitech.gymcoach.core

import kotlin.math.abs
import kotlin.math.max

/**
 * Per-frame features consumed by AutoDetector (kept minimal so tests can
 * synthesize them without full skeletons).
 */
data class FrameFeatures(
    val trunk: Double = 10.0,
    val knee: Double = 170.0,
    val elbow: Double = 170.0,
    val hip: Double = 170.0,
    val shoY: Double = 0.3,
    val wriY: Double = 0.5,
    val torso: Double = 0.25,
    val overhead: Boolean = false,
    val kneeSplit: Double = 0.1,
)

fun frameFeatures(ang: BodyAngles, pts: Skeleton): FrameFeatures {
    val shoY = (pts[Joint.LEFT_SHOULDER].y + pts[Joint.RIGHT_SHOULDER].y) / 2
    val hipY = (pts[Joint.LEFT_HIP].y + pts[Joint.RIGHT_HIP].y) / 2
    val wriY = (pts[Joint.LEFT_WRIST].y + pts[Joint.RIGHT_WRIST].y) / 2
    val torso = max(abs(hipY - shoY), 1e-3)
    return FrameFeatures(
        trunk = ang.trunkLean, knee = ang.knee, elbow = ang.elbow,
        hip = ang.hip, shoY = shoY, wriY = wriY, torso = torso,
        overhead = wriY < shoY - 0.03,             // image y grows downward
        kneeSplit = abs(pts[Joint.LEFT_KNEE].y - pts[Joint.RIGHT_KNEE].y) / torso,
    )
}

/**
 * Rule-based exercise classifier over a sliding window of skeleton features.
 * Locks after 3 agreeing votes. Bench press is NOT detectable from the
 * skeleton alone (looks like a push-up) — select it manually.
 */
class AutoDetector {
    companion object {
        const val WINDOW_S = 2.0
        const val VOTE_EVERY_S = 0.5
        const val NEED_AGREE = 3
    }

    private val buf = ArrayDeque<Pair<Double, FrameFeatures>>()
    private val votes = ArrayDeque<String?>()
    private var nextVoteT = WINDOW_S

    fun update(feat: FrameFeatures, t: Double): String? {
        buf.addLast(t to feat)
        while (buf.isNotEmpty() && t - buf.first().first > WINDOW_S) {
            buf.removeFirst()
        }
        if (t < nextVoteT || buf.size < 20) return null
        nextVoteT = t + VOTE_EVERY_S
        val vote = classify()
        votes.addLast(vote)
        if (votes.size > NEED_AGREE) votes.removeFirst()
        if (votes.size == NEED_AGREE && vote != null &&
            votes.all { it == vote }
        ) {
            return vote
        }
        return null
    }

    internal fun classify(): String? {
        val f = buf.map { it.second }
        fun rom(key: (FrameFeatures) -> Double): Double {
            val vals = f.map(key)
            return (vals.max()) - (vals.min())
        }
        fun mean(key: (FrameFeatures) -> Double): Double =
            f.sumOf(key) / f.size

        val torso = mean { it.torso }
        val trunkMean = mean { it.trunk }
        val trunkMax = f.maxOf { it.trunk }
        val romKnee = rom { it.knee }
        val romElbow = rom { it.elbow }
        val romHip = rom { it.hip }
        val overhead = f.count { it.overhead }.toDouble() / f.size
        val dispSho = rom { it.shoY } / torso
        val dispWri = rom { it.wriY } / torso
        val kneeSplit = f.maxOf { it.kneeSplit }

        if (trunkMean > 55) {                      // body horizontal
            return if (romElbow > 25) "pushup" else "plank"
        }
        if (overhead > 0.7 && romElbow > 30) {     // hands overhead
            return if (dispSho > 1.3 * dispWri) "pullup" else "shoulder_press"
        }
        if (romKnee > 35) {                        // legs driving
            if (trunkMax > 55) return "deadlift"
            if (kneeSplit > 0.35) return "lunge"
            return "squat"
        }
        if (trunkMax > 55 && romHip > 30) {        // hip hinge, stiff knees
            return "deadlift"
        }
        if (romElbow > 40 && overhead < 0.3) {     // arms only, below head
            return "curl"
        }
        return null
    }
}

/**
 * Velocity-based fatigue: warn when concentric speed drops >20% against
 * the best of the first three reps.
 */
class FatigueMonitor(private val threshold: Double = 0.20) {
    private val vels = mutableListOf<Double>()
    private var warned = false
    var loss = 0.0
        private set

    /** Feed one rep's concentric velocity; true => fire fatigue cue. */
    fun add(velocity: Double): Boolean {
        vels.add(velocity)
        if (vels.size < 4) return false
        val base = vels.take(3).max()
        val cur = vels.takeLast(2).sum() / 2
        loss = if (base > 0) max(0.0, 1 - cur / base) else 0.0
        if (loss > threshold && !warned) {
            warned = true
            return true
        }
        return false
    }
}

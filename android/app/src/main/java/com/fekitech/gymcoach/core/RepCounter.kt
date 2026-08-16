// Rep-counting finite-state machine + plank hold tracker.
// Port of the desktop prototype's RepCounter/PlankTracker (via iOS CoachCore).
package com.fekitech.gymcoach.core

import kotlin.math.max
import kotlin.math.min

enum class RepState(val label: String) {
    IDLE("IDLE"), DESCENT("DESCENT"), BOTTOM("BOTTOM"), ASCENT("ASCENT")
}

data class RepEvent(
    val count: Int,
    val duration: Double,
    val eccentricS: Double,
    val concentricS: Double,
    val minAngle: Double,
    val fullDepth: Boolean,
    var faults: List<String> = emptyList(),
    var score: Int = 100,
)

/**
 * IDLE -> DESCENT -> BOTTOM -> ASCENT -> (rep++) on the signal angle.
 *
 * "Descent/ascent" refer to the *angle*: for curls and pull-ups the angle
 * descends during the lift, so `concentricPhase` maps phases to tempo names.
 */
class RepCounter(val spec: ExerciseSpec) {
    var state: RepState = RepState.IDLE
        private set
    var count = 0
        private set
    private var tStart = 0.0
    private var tBottom = 0.0
    private var minAngle = 180.0
    private var repFaults = mutableSetOf<String>()

    fun noteFault(fault: String) {
        if (state != RepState.IDLE) repFaults.add(fault)
    }

    fun update(angle: Double, t: Double): RepEvent? {
        when (state) {
            RepState.IDLE -> if (angle < spec.startBelow) {
                state = RepState.DESCENT
                tStart = t
                minAngle = angle
                repFaults = mutableSetOf()
            }
            RepState.DESCENT -> {
                minAngle = min(minAngle, angle)
                if (angle < spec.bottomBelow) {
                    state = RepState.BOTTOM
                    tBottom = t
                } else if (angle > minAngle + 15) {    // turned around early
                    state = RepState.ASCENT
                    tBottom = t
                }
            }
            RepState.BOTTOM -> {
                minAngle = min(minAngle, angle)
                if (angle > minAngle + 10) state = RepState.ASCENT
            }
            RepState.ASCENT -> if (angle > spec.lockoutAbove) {
                val dur = t - tStart
                state = RepState.IDLE
                if (dur < spec.minRepS) return null    // noise blip, not a rep
                count += 1
                val downS = tBottom - tStart
                val upS = t - tBottom
                val (ecc, con) = if (spec.concentricPhase == "ascent")
                    downS to upS else upS to downS
                return RepEvent(
                    count = count, duration = dur,
                    eccentricS = ecc, concentricS = con,
                    minAngle = minAngle,
                    fullDepth = minAngle < spec.bottomBelow,
                    faults = repFaults.sorted(),
                )
            }
        }
        return null
    }
}

/** Timed hold: accumulate time while the body line stays straight. */
class PlankTracker(
    private val goodAbove: Double = 160.0,
    private val graceS: Double = 1.0,
) {
    var total = 0.0
        private set
    var streak = 0.0
        private set
    var best = 0.0
        private set
    private var badFor = 0.0
    private var tPrev: Double? = null

    /** Returns true when a "fix your line" cue should fire. */
    fun update(bodyLine: Double, t: Double): Boolean {
        val dt = tPrev?.let { max(t - it, 0.0) } ?: 0.0
        tPrev = t
        if (bodyLine >= goodAbove) {
            total += dt
            streak += dt
            best = max(best, streak)
            badFor = 0.0
            return false
        }
        val wasOK = badFor <= graceS
        badFor += dt
        if (badFor > graceS) {
            streak = 0.0
            return wasOK                 // fire cue once when grace expires
        }
        return false
    }
}

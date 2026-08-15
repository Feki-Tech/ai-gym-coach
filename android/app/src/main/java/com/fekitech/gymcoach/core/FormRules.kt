// Form rules, fault catalog, scoring, and rate-limited feedback.
// English-only for now (the iOS package localizes; Android localization is a
// follow-up — messages live here in one place to make that swap easy).
package com.fekitech.gymcoach.core

import kotlin.math.max

data class FaultInfo(val priority: Int, val message: String, val penalty: Int)

/** fault id -> (priority, message, score penalty). Lower priority = said first. */
val faultMessages: Map<String, FaultInfo> = mapOf(
    "back_lean" to FaultInfo(0, "Keep your chest up", 30),
    "back_round" to FaultInfo(0, "Straighten your back", 30),
    "body_sag" to FaultInfo(0, "Keep your body straight", 25),
    "knees_cave" to FaultInfo(0, "Push your knees out", 25),
    "shallow" to FaultInfo(1, "Go deeper", 20),
    "elbow_swing" to FaultInfo(1, "Keep your elbows still", 20),
    "elbow_flare" to FaultInfo(1, "Tuck your elbows in", 15),
    "torso_lean" to FaultInfo(1, "Keep your torso upright", 15),
    "lean_back" to FaultInfo(1, "Don't lean back", 15),
    "uneven" to FaultInfo(1, "Even out both sides", 15),
    "chin" to FaultInfo(1, "Chin over the bar", 15),
    "shrug_neck" to FaultInfo(1, "Relax your neck", 10),
    "too_fast" to FaultInfo(2, "Slow down", 10),
)

const val FATIGUE_MESSAGE = "You're slowing down — consider ending the set"
const val PRAISE_MESSAGE = "Great form!"

private fun moving(s: RepState) = s != RepState.IDLE

/** Per-frame faults, phase-gated — mirrors the desktop LIVE_RULES table. */
fun liveFaults(exercise: String, ang: BodyAngles, state: RepState): List<String> {
    val f = mutableListOf<String>()
    when (exercise) {
        "squat" -> {
            if (moving(state) && ang.trunkLean > 50) f.add("back_lean")
            if ((state == RepState.BOTTOM || state == RepState.ASCENT) &&
                ang.valgusRatio < 0.7
            ) f.add("knees_cave")
        }
        "pushup" -> {
            if (moving(state) && ang.bodyLine < 155) f.add("body_sag")
            if (state == RepState.BOTTOM && ang.elbowFlare > 100) f.add("elbow_flare")
        }
        "bench" -> if (moving(state) && ang.wristYDiff > 0.08) f.add("uneven")
        "deadlift" -> if (moving(state) && ang.neck < 150) f.add("back_round")
        "lunge" -> if (moving(state) && ang.trunkLean > 30) f.add("torso_lean")
        "shoulder_press" -> {
            if (moving(state) && ang.trunkLean > 20) f.add("lean_back")
            if (moving(state) && ang.wristYDiff > 0.08) f.add("uneven")
        }
        "curl" -> {
            if (moving(state) && ang.upperArmSwing > 25) f.add("elbow_swing")
            if (moving(state) && ang.trunkLean > 20) f.add("torso_lean")
        }
        "pullup" -> {
            if (state == RepState.BOTTOM && ang.noseAboveWrists < 0) f.add("chin")
            if (moving(state) && ang.wristYDiff > 0.10) f.add("uneven")
        }
        "plank" -> if (ang.neck < 140) f.add("shrug_neck")
    }
    return f
}

/** Faults judged once per completed rep. */
fun repFaults(spec: ExerciseSpec, ev: RepEvent): List<String> {
    val f = mutableListOf<String>()
    if (!ev.fullDepth) f.add("shallow")
    if (ev.concentricS < spec.minConcentricS) f.add("too_fast")
    return f
}

fun scoreRep(ev: RepEvent): Int =
    max(0, 100 - ev.faults.sumOf { faultMessages[it]?.penalty ?: 0 })

/** Rate-limited, priority-ordered coaching cues. */
class FeedbackEngine(private val cooldown: Double = 3.0) {
    private val lastSaid = mutableMapOf<String, Double>()
    var current = ""

    /** Returns the message if a new cue fired (for the voice channel). */
    fun push(faults: List<String>, t: Double): String? {
        val ordered = faults.sortedBy { faultMessages[it]?.priority ?: 9 }
        for (fault in ordered) {
            if (t - (lastSaid[fault] ?: -1e9) >= cooldown) {
                lastSaid[fault] = t
                current = faultMessages[fault]?.message ?: fault
                return current
            }
        }
        if (faults.isEmpty()) current = ""
        return null
    }

    fun praise(): String {
        current = PRAISE_MESSAGE
        return current
    }
}

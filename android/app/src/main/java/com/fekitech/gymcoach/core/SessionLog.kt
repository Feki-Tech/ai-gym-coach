// Session records — same shape as the desktop prototype's workout_log.json
// (serialization to JSON happens in the app layer; core stays JVM-pure).
package com.fekitech.gymcoach.core

import kotlin.math.max

data class RepRecord(
    val n: Int,
    val score: Int,
    val eccentricS: Double,
    val concentricS: Double,
    val minAngle: Double,
    val velocity: Double?,
    val faults: List<String>,
)

data class PlankRecord(val totalHoldS: Double, val bestStreakS: Double)

data class SessionSummary(
    val reps: Int,
    val avgScore: Double?,
    val avgConcentricS: Double?,
    val faultCounts: Map<String, Int>,
    val velocityLossPct: Double?,
)

data class SessionRecord(
    val started: String,
    val exercise: String,
    val durationS: Double,
    val reps: List<RepRecord>,
    val plank: PlankRecord?,
    val summary: SessionSummary,
)

/** Accumulates rep events during a session and builds the final record. */
class SessionBuilder {
    private val reps = mutableListOf<RepRecord>()
    private val vels = mutableListOf<Double>()

    fun addRep(ev: RepEvent, velocity: Double) {
        reps.add(
            RepRecord(
                n = ev.count, score = ev.score,
                eccentricS = ev.eccentricS, concentricS = ev.concentricS,
                minAngle = ev.minAngle, velocity = velocity,
                faults = ev.faults,
            )
        )
        vels.add(velocity)
    }

    fun finish(exercise: String, durationS: Double, started: String,
               plank: PlankTracker? = null): SessionRecord {
        val faultCounts = mutableMapOf<String, Int>()
        for (r in reps) for (f in r.faults) {
            faultCounts[f] = (faultCounts[f] ?: 0) + 1
        }
        var velocityLoss: Double? = null
        if (vels.size >= 4) {
            val base = vels.take(3).max()
            val cur = vels.takeLast(2).sum() / 2
            if (base > 0) velocityLoss = max(0.0, (1 - cur / base) * 100)
        }
        return SessionRecord(
            started = started,
            exercise = exercise,
            durationS = durationS,
            reps = reps.toList(),
            plank = plank?.let { PlankRecord(it.total, it.best) },
            summary = SessionSummary(
                reps = reps.size,
                avgScore = if (reps.isEmpty()) null
                    else reps.sumOf { it.score.toDouble() } / reps.size,
                avgConcentricS = if (reps.isEmpty()) null
                    else reps.sumOf { it.concentricS } / reps.size,
                faultCounts = faultCounts,
                velocityLossPct = velocityLoss,
            ),
        )
    }
}

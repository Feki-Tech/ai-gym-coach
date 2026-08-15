// JVM unit tests for the core port — mirrors the invariants CoachCoreTests
// covers on iOS and the desktop selftests cover in Python.
package com.fekitech.gymcoach.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos

class GeometryTest {
    @Test fun rightAngle() {
        val a = jointAngle(P2(0.0, 0.0), P2(0.0, 1.0), P2(1.0, 1.0))
        assertEquals(90.0, a, 1e-6)
    }

    @Test fun straightLineIs180() {
        val a = jointAngle(P2(0.0, 0.0), P2(0.0, 0.5), P2(0.0, 1.0))
        assertEquals(180.0, a, 1e-6)
    }

    @Test fun verticalSegmentIsZero() {
        assertEquals(0.0, segmentVsVertical(P2(0.5, 0.2), P2(0.5, 0.8)), 1e-6)
    }

    @Test fun oneEuroConverges() {
        val f = OneEuroFilter()
        var out = 0.0
        for (i in 0..100) out = f.filter(10.0, i / 30.0)
        assertEquals(10.0, out, 0.1)
    }
}

class RepCounterTest {
    private val squat = specs.getValue("squat")

    private fun angleAt(t: Double, periodS: Double = 3.0): Double {
        // 170 -> 90 -> 170 knee wave
        return 130 + 40 * cos(2 * PI * t / periodS)
    }

    @Test fun countsCleanReps() {
        val c = RepCounter(squat)
        var reps = 0
        var lastEv: RepEvent? = null
        var t = 0.0
        while (t < 9.0) {
            c.update(angleAt(t), t)?.let { reps += 1; lastEv = it }
            t += 1.0 / 30
        }
        assertEquals(3, reps)
        assertTrue(lastEv!!.fullDepth)
        assertTrue(lastEv!!.eccentricS > 0 && lastEv!!.concentricS > 0)
    }

    @Test fun rejectsNoiseBlip() {
        val c = RepCounter(squat)
        var reps = 0
        // full-depth dip and lockout inside 0.4 s — below minRepS
        val angles = listOf(170.0, 140.0, 95.0, 120.0, 168.0, 170.0)
        angles.forEachIndexed { i, a ->
            if (c.update(a, i * 0.08) != null) reps += 1
        }
        assertEquals(0, reps)
    }

    @Test fun shallowRepNotFullDepth() {
        val c = RepCounter(squat)
        var ev: RepEvent? = null
        var t = 0.0
        // dips to 120 (past startBelow 150, above bottomBelow 100)
        while (t < 3.0 && ev == null) {
            ev = c.update(145 + 25 * cos(2 * PI * t / 2.5), t)
            t += 1.0 / 30
        }
        assertNotNull(ev)
        assertFalse(ev!!.fullDepth)
    }

    @Test fun curlConcentricIsDescentPhase() {
        val curl = specs.getValue("curl")
        val c = RepCounter(curl)
        var ev: RepEvent? = null
        var t = 0.0
        while (t < 4.0 && ev == null) {
            // elbow: 165 -> 60 -> 165; for curls the angle-descent is the lift
            ev = c.update(112.5 + 52.5 * cos(2 * PI * t / 3.0), t)
            t += 1.0 / 30
        }
        assertNotNull(ev)
        // concentric = the descent phase of the angle wave
        assertEquals(ev!!.duration, ev!!.eccentricS + ev!!.concentricS, 0.2)
    }
}

class PlankTrackerTest {
    @Test fun accumulatesAndFiresOnce() {
        val p = PlankTracker()
        var cues = 0
        var t = 0.0
        while (t < 5.0) {                       // straight for 5 s
            if (p.update(175.0, t)) cues += 1
            t += 0.1
        }
        assertEquals(0, cues)
        assertEquals(5.0, p.total, 0.2)
        while (t < 7.0) {                       // sagging past the grace period
            if (p.update(140.0, t)) cues += 1
            t += 0.1
        }
        assertEquals(1, cues)
        assertEquals(0.0, p.streak, 1e-9)
        assertEquals(5.0, p.best, 0.2)
    }
}

class AutoDetectorTest {
    private fun run(frames: (Double) -> FrameFeatures, seconds: Double): String? {
        val d = AutoDetector()
        var found: String? = null
        var t = 0.0
        while (t < seconds) {
            d.update(frames(t), t)?.let { found = found ?: it }
            t += 1.0 / 30
        }
        return found
    }

    @Test fun detectsSquat() {
        val got = run({ t ->
            FrameFeatures(knee = 130 + 40 * cos(2 * PI * t / 3.0),
                          hip = 130 + 40 * cos(2 * PI * t / 3.0),
                          trunk = 20.0)
        }, 5.0)
        assertEquals("squat", got)
    }

    @Test fun detectsPushup() {
        val got = run({ t ->
            FrameFeatures(trunk = 75.0,
                          elbow = 125 + 35 * cos(2 * PI * t / 2.5))
        }, 5.0)
        assertEquals("pushup", got)
    }

    @Test fun detectsPlank() {
        val got = run({ FrameFeatures(trunk = 75.0, elbow = 170.0) }, 5.0)
        assertEquals("plank", got)
    }

    @Test fun detectsCurl() {
        val got = run({ t ->
            FrameFeatures(elbow = 105 + 55 * cos(2 * PI * t / 2.5),
                          overhead = false)
        }, 5.0)
        assertEquals("curl", got)
    }

    @Test fun idleBodyDetectsNothing() {
        assertNull(run({ FrameFeatures() }, 5.0))
    }
}

class FatigueMonitorTest {
    @Test fun firesOnSlowdownOnce() {
        val f = FatigueMonitor()
        assertFalse(f.add(100.0))
        assertFalse(f.add(98.0))
        assertFalse(f.add(97.0))
        assertTrue(f.add(60.0))     // 4th rep, way below the early best
        assertTrue(f.loss > 0.20)
        assertFalse(f.add(50.0))    // warns only once
    }
}

class SessionEngineTest {
    /** Symmetric 15-joint skeleton; knee x-offset bends the knees. */
    private fun skeleton(kneeOffset: Double): Skeleton {
        val lm = { x: Double, y: Double -> Landmark(x, y, 1.0) }
        return listOf(
            lm(0.50, 0.10),                       // nose
            lm(0.48, 0.10), lm(0.52, 0.10),       // ears
            lm(0.45, 0.25), lm(0.55, 0.25),       // shoulders
            lm(0.42, 0.37), lm(0.58, 0.37),       // elbows
            lm(0.41, 0.48), lm(0.59, 0.48),       // wrists
            lm(0.46, 0.50), lm(0.54, 0.50),       // hips
            lm(0.46 - kneeOffset, 0.70),          // left knee (bends forward)
            lm(0.54 + kneeOffset, 0.70),          // right knee
            lm(0.46, 0.90), lm(0.54, 0.90),       // ankles
        )
    }

    @Test fun countsSquatsFromSkeletons() {
        val engine = SessionEngine("squat")
        var t = 0.0
        var reps = 0
        while (t < 9.5) {           // 3rd rep locks out ~8.8 s + smoothing lag
            // knee offset 0 (standing, ~180°) <-> 0.24 (deep, <100°)
            val phase = (1 - cos(2 * PI * t / 3.0)) / 2
            val out = engine.process(skeleton(0.24 * phase), t)
            if (out.repEvent != null) reps += 1
            t += 1.0 / 30
        }
        assertEquals(3, reps)
        assertEquals(3, engine.hud.repCount)
        val record = engine.finish(9.0, "2026-01-01T00:00:00")
        assertEquals(3, record.summary.reps)
        assertEquals("squat", record.exercise)
        assertNotNull(record.summary.avgScore)
    }

    @Test fun autoDetectLocksThenCounts() {
        val engine = SessionEngine("auto")
        var t = 0.0
        var detectedCue = false
        while (t < 12.0) {
            val phase = (1 - cos(2 * PI * t / 3.0)) / 2
            val out = engine.process(skeleton(0.24 * phase), t)
            if (out.spokenCues.any { it.startsWith("Detected") }) {
                detectedCue = true
            }
            t += 1.0 / 30
        }
        assertTrue(detectedCue)
        assertEquals("squat", engine.exercise)
        assertTrue(engine.hud.repCount >= 1)
    }
}

// Geometry primitives + One Euro smoothing.
// Coordinate convention throughout core: normalized image coordinates with
// the origin at the TOP-LEFT and y growing DOWN — identical to the desktop
// prototype and iOS CoachCore. MediaPipe already delivers this convention.
package com.fekitech.gymcoach.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class P2(val x: Double, val y: Double)

/** Angle ABC in degrees at vertex [b]. */
fun jointAngle(a: P2, b: P2, c: P2): Double {
    val bax = a.x - b.x; val bay = a.y - b.y
    val bcx = c.x - b.x; val bcy = c.y - b.y
    val denom = sqrt(bax * bax + bay * bay) * sqrt(bcx * bcx + bcy * bcy)
    if (denom < 1e-9) return 180.0
    val cosang = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / denom))
    return acos(cosang) * 180.0 / PI
}

/**
 * Angle of the segment (top -> bottom) vs the vertical axis, in degrees.
 * 0 = perfectly vertical. Uses y-down image coordinates (-y is "up").
 */
fun segmentVsVertical(top: P2, bottom: P2): Double {
    val vx = top.x - bottom.x; val vy = top.y - bottom.y
    val n = sqrt(vx * vx + vy * vy)
    if (n < 1e-9) return 0.0
    val cosang = max(-1.0, min(1.0, -vy / n))
    return acos(cosang) * 180.0 / PI
}

/** Adaptive low-pass filter: smooth when slow, responsive when fast. */
class OneEuroFilter(
    private val minCutoff: Double = 1.0,
    private val beta: Double = 0.02,
    private val dCutoff: Double = 1.0,
) {
    private var xPrev: Double? = null
    private var dxPrev = 0.0
    private var tPrev: Double? = null

    private fun alpha(cutoff: Double, dt: Double): Double {
        val tau = 1.0 / (2 * PI * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    fun filter(x: Double, t: Double): Double {
        val xp = xPrev
        val tp = tPrev
        if (xp == null || tp == null) {
            xPrev = x; dxPrev = 0.0; tPrev = t
            return x
        }
        val dt = max(t - tp, 1e-6)
        val dx = (x - xp) / dt
        val aD = alpha(dCutoff, dt)
        val dxHat = aD * dx + (1 - aD) * dxPrev
        val cutoff = minCutoff + beta * abs(dxHat)
        val a = alpha(cutoff, dt)
        val xHat = a * x + (1 - a) * xp
        xPrev = xHat; dxPrev = dxHat; tPrev = t
        return xHat
    }
}

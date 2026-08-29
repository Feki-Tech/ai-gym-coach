// The trained exercise classifier — the SAME gated, versioned TinyMLP the
// desktop trains (pose_coach.py --train-classifier), exported as portable
// JSON (--export-model) and run here without any ML framework: ~1.5k
// parameters, a matrix multiply, a ReLU and a softmax. Parity with the
// Python engine is pinned by data/parity_fixtures.json (window_feature and
// mlp sections).
package com.fekitech.gymcoach.core

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Fixed-size feature vector over a window of frames — port of the desktop's
 * window_features: per-channel mean/std/min/max (population std!) plus
 * torso-normalized shoulder & wrist travel. Channel order = FEAT_KEYS.
 */
object WindowFeatures {
    const val NDIM = 38

    private fun channels(f: FrameFeatures): DoubleArray = doubleArrayOf(
        f.trunk, f.knee, f.elbow, f.hip, f.shoY, f.wriY, f.torso,
        if (f.overhead) 1.0 else 0.0, f.kneeSplit)

    private const val TORSO = 6      // index of "torso" in FEAT_KEYS order
    private const val SHO_Y = 4
    private const val WRI_Y = 5

    fun of(frames: List<FrameFeatures>): DoubleArray {
        require(frames.isNotEmpty()) { "window_features needs >= 1 frame" }
        val rows = frames.map { channels(it) }
        val n = rows.size
        val nc = rows[0].size
        val mean = DoubleArray(nc)
        val mn = DoubleArray(nc) { Double.POSITIVE_INFINITY }
        val mx = DoubleArray(nc) { Double.NEGATIVE_INFINITY }
        for (r in rows) for (c in 0 until nc) {
            mean[c] += r[c]
            mn[c] = min(mn[c], r[c])
            mx[c] = max(mx[c], r[c])
        }
        for (c in 0 until nc) mean[c] /= n
        val std = DoubleArray(nc)
        for (r in rows) for (c in 0 until nc) {
            val d = r[c] - mean[c]
            std[c] += d * d
        }
        for (c in 0 until nc) std[c] = kotlin.math.sqrt(std[c] / n)
        val torso = max(mean[TORSO], 1e-3)
        val out = DoubleArray(NDIM)
        for (c in 0 until nc) {
            out[c] = mean[c]
            out[nc + c] = std[c]
            out[2 * nc + c] = mn[c]
            out[3 * nc + c] = mx[c]
        }
        out[4 * nc] = (mx[SHO_Y] - mn[SHO_Y]) / torso
        out[4 * nc + 1] = (mx[WRI_Y] - mn[WRI_Y]) / torso
        return out
    }
}

/** Two-layer MLP inference: (x-mu)/sd -> ReLU hidden -> softmax classes. */
class TinyMlp(
    val classes: List<String>,
    val minProba: Double,
    private val w1: Array<DoubleArray>,   // [NDIM][hidden]
    private val b1: DoubleArray,
    private val w2: Array<DoubleArray>,   // [hidden][classes]
    private val b2: DoubleArray,
    private val mu: DoubleArray,
    private val sd: DoubleArray,
    val modelVersion: String = "unknown",
) {
    fun predict(x: DoubleArray): DoubleArray {
        val xn = DoubleArray(x.size) { (x[it] - mu[it]) / sd[it] }
        val h = DoubleArray(b1.size) { j ->
            var acc = b1[j]
            for (i in xn.indices) acc += xn[i] * w1[i][j]
            max(0.0, acc)
        }
        val z = DoubleArray(b2.size) { k ->
            var acc = b2[k]
            for (j in h.indices) acc += h[j] * w2[j][k]
            acc
        }
        val zmax = z.max()
        var tot = 0.0
        val e = DoubleArray(z.size) { exp(z[it] - zmax).also { v -> tot += v } }
        return DoubleArray(e.size) { e[it] / tot }
    }

}

/**
 * AutoDetector with the rule-based vote swapped for the trained MLP — same
 * sliding window, vote cadence, and 3-agreeing-votes lock-in as the desktop.
 */
class MlDetector(private val model: TinyMlp) : AutoDetector() {
    override fun classify(): String? {
        val p = model.predict(WindowFeatures.of(windowFrames()))
        var ci = 0
        for (i in p.indices) if (p[i] > p[ci]) ci = i
        return if (p[ci] >= model.minProba) model.classes[ci] else null
    }
}

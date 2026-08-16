// Skeleton model + per-frame body angles.
// 15 joints — the same subset the iOS CoachCore uses; PoseAnalyzer maps
// MediaPipe's 33 BlazePose landmarks down to these.
package com.fekitech.gymcoach.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

enum class Joint {
    NOSE,
    LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
}

data class Landmark(val x: Double, val y: Double, val confidence: Double) {
    val p2: P2 get() = P2(x, y)
}

/** Fixed-size list indexed by [Joint.ordinal]. */
typealias Skeleton = List<Landmark>

operator fun Skeleton.get(j: Joint): Landmark = this[j.ordinal]

const val VIS_MIN = 0.5

/**
 * One Euro per coordinate per joint, holding the last good value while a
 * joint drops below the visibility threshold.
 */
class SkeletonSmoother {
    private val filters = List(Joint.entries.size) {
        Pair(OneEuroFilter(), OneEuroFilter())
    }
    private var last: Skeleton? = null

    fun update(pts: Skeleton, t: Double): Skeleton {
        val out = pts.mapIndexed { i, lm ->
            val l = last
            if (lm.confidence < VIS_MIN && l != null &&
                l[i].confidence >= VIS_MIN
            ) {
                l[i]                              // hold last good value
            } else {
                lm.copy(
                    x = filters[i].first.filter(lm.x, t),
                    y = filters[i].second.filter(lm.y, t),
                )
            }
        }
        last = out
        return out
    }
}

enum class SignalKey { KNEE, HIP, ELBOW, BODY_LINE }

/** All per-frame features used by the FSM and the form rules. */
data class BodyAngles(
    val side: String,
    val knee: Double,
    val hip: Double,
    val elbow: Double,
    val trunkLean: Double,
    val upperArmSwing: Double,
    val bodyLine: Double,             // 180 = straight
    val elbowFlare: Double,
    val neck: Double,
    val valgusRatio: Double,          // < 1 => knees caving in
    val wristYDiff: Double,
    val noseAboveWrists: Double,
) {
    fun value(key: SignalKey): Double = when (key) {
        SignalKey.KNEE -> knee
        SignalKey.HIP -> hip
        SignalKey.ELBOW -> elbow
        SignalKey.BODY_LINE -> bodyLine
    }
}

fun pickSide(pts: Skeleton): String {
    val lJoints = listOf(Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_HIP,
                         Joint.LEFT_KNEE, Joint.LEFT_ANKLE)
    val rJoints = listOf(Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW,
                         Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE)
    val left = lJoints.sumOf { pts[it].confidence } / 5
    val right = rJoints.sumOf { pts[it].confidence } / 5
    return if (left >= right) "L" else "R"
}

fun bodyAngles(pts: Skeleton): BodyAngles {
    val s = pickSide(pts)
    val ear = if (s == "L") Joint.LEFT_EAR else Joint.RIGHT_EAR
    val sho = if (s == "L") Joint.LEFT_SHOULDER else Joint.RIGHT_SHOULDER
    val elb = if (s == "L") Joint.LEFT_ELBOW else Joint.RIGHT_ELBOW
    val wri = if (s == "L") Joint.LEFT_WRIST else Joint.RIGHT_WRIST
    val hip = if (s == "L") Joint.LEFT_HIP else Joint.RIGHT_HIP
    val kne = if (s == "L") Joint.LEFT_KNEE else Joint.RIGHT_KNEE
    val ank = if (s == "L") Joint.LEFT_ANKLE else Joint.RIGHT_ANKLE

    var valgus = 1.0
    if (min(
            min(pts[Joint.LEFT_KNEE].confidence, pts[Joint.RIGHT_KNEE].confidence),
            min(pts[Joint.LEFT_ANKLE].confidence, pts[Joint.RIGHT_ANKLE].confidence),
        ) > VIS_MIN
    ) {
        val kneeW = abs(pts[Joint.LEFT_KNEE].x - pts[Joint.RIGHT_KNEE].x)
        val ankleW = max(abs(pts[Joint.LEFT_ANKLE].x - pts[Joint.RIGHT_ANKLE].x), 1e-4)
        valgus = kneeW / ankleW
    }
    var wristYDiff = 0.0
    var noseAboveWrists = 1.0
    if (min(pts[Joint.LEFT_WRIST].confidence, pts[Joint.RIGHT_WRIST].confidence) > VIS_MIN) {
        wristYDiff = abs(pts[Joint.LEFT_WRIST].y - pts[Joint.RIGHT_WRIST].y)
        noseAboveWrists =
            (pts[Joint.LEFT_WRIST].y + pts[Joint.RIGHT_WRIST].y) / 2 - pts[Joint.NOSE].y
    }

    return BodyAngles(
        side = s,
        knee = jointAngle(pts[hip].p2, pts[kne].p2, pts[ank].p2),
        hip = jointAngle(pts[sho].p2, pts[hip].p2, pts[kne].p2),
        elbow = jointAngle(pts[sho].p2, pts[elb].p2, pts[wri].p2),
        trunkLean = segmentVsVertical(top = pts[sho].p2, bottom = pts[hip].p2),
        upperArmSwing = segmentVsVertical(top = pts[sho].p2, bottom = pts[elb].p2),
        bodyLine = jointAngle(pts[sho].p2, pts[hip].p2, pts[ank].p2),
        elbowFlare = jointAngle(pts[hip].p2, pts[sho].p2, pts[elb].p2),
        neck = if (pts[ear].confidence > VIS_MIN)
            jointAngle(pts[ear].p2, pts[sho].p2, pts[hip].p2) else 180.0,
        valgusRatio = valgus,
        wristYDiff = wristYDiff,
        noseAboveWrists = noseAboveWrists,
    )
}

/** Skeleton edges for overlay drawing. */
val skeletonEdges: List<Pair<Joint, Joint>> = listOf(
    Joint.LEFT_SHOULDER to Joint.RIGHT_SHOULDER, Joint.LEFT_HIP to Joint.RIGHT_HIP,
    Joint.LEFT_SHOULDER to Joint.LEFT_HIP, Joint.RIGHT_SHOULDER to Joint.RIGHT_HIP,
    Joint.LEFT_SHOULDER to Joint.LEFT_ELBOW, Joint.LEFT_ELBOW to Joint.LEFT_WRIST,
    Joint.RIGHT_SHOULDER to Joint.RIGHT_ELBOW, Joint.RIGHT_ELBOW to Joint.RIGHT_WRIST,
    Joint.LEFT_HIP to Joint.LEFT_KNEE, Joint.LEFT_KNEE to Joint.LEFT_ANKLE,
    Joint.RIGHT_HIP to Joint.RIGHT_KNEE, Joint.RIGHT_KNEE to Joint.RIGHT_ANKLE,
)

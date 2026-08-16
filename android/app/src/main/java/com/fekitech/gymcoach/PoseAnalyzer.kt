// MediaPipe Tasks pose landmarker in LIVE_STREAM mode, mapped down to the
// 15-joint core skeleton (roadmap: "Android app (MediaPipe Tasks, Kotlin)").
package com.fekitech.gymcoach

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import com.fekitech.gymcoach.core.Joint
import com.fekitech.gymcoach.core.Landmark
import com.fekitech.gymcoach.core.Skeleton
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/** BlazePose index for each core joint (33-landmark topology). */
private val BLAZE_INDEX = mapOf(
    Joint.NOSE to 0,
    Joint.LEFT_EAR to 7, Joint.RIGHT_EAR to 8,
    Joint.LEFT_SHOULDER to 11, Joint.RIGHT_SHOULDER to 12,
    Joint.LEFT_ELBOW to 13, Joint.RIGHT_ELBOW to 14,
    Joint.LEFT_WRIST to 15, Joint.RIGHT_WRIST to 16,
    Joint.LEFT_HIP to 23, Joint.RIGHT_HIP to 24,
    Joint.LEFT_KNEE to 25, Joint.RIGHT_KNEE to 26,
    Joint.LEFT_ANKLE to 27, Joint.RIGHT_ANKLE to 28,
)

class PoseAnalyzer(
    context: Context,
    private val mirrored: Boolean,           // front camera preview is mirrored
    private val onSkeleton: (Skeleton, Double) -> Unit,
) {
    private val landmarker: PoseLandmarker

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
            .build()
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result: PoseLandmarkerResult, _ ->
                toSkeleton(result)?.let {
                    onSkeleton(it, result.timestampMs() / 1000.0)
                }
            }
            .setErrorListener { /* dropped frame — keep going */ }
            .build()
        landmarker = PoseLandmarker.createFromOptions(context, options)
    }

    /** Feed one CameraX frame. Closes the proxy. */
    fun analyze(imageProxy: ImageProxy, timestampMs: Long) {
        val rotation = imageProxy.imageInfo.rotationDegrees
        var bitmap = imageProxy.toBitmap()
        imageProxy.close()
        if (rotation != 0) {
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width,
                                         bitmap.height, m, true)
        }
        landmarker.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
    }

    private fun toSkeleton(result: PoseLandmarkerResult): Skeleton? {
        val all = result.landmarks().firstOrNull() ?: return null
        if (all.size < 29) return null
        return Joint.entries.map { j ->
            val lm = all[BLAZE_INDEX.getValue(j)]
            val x = if (mirrored) 1.0 - lm.x().toDouble() else lm.x().toDouble()
            Landmark(
                x = x,
                y = lm.y().toDouble(),
                confidence = lm.visibility().orElse(1.0f).toDouble(),
            )
        }
    }

    fun close() = landmarker.close()
}

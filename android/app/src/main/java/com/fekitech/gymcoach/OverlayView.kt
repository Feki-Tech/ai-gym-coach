// Skeleton overlay, drawn in normalized image coordinates mapped with the
// same center-crop (FILL_CENTER) transform the PreviewView applies.
package com.fekitech.gymcoach

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.fekitech.gymcoach.core.Skeleton
import com.fekitech.gymcoach.core.VIS_MIN
import com.fekitech.gymcoach.core.skeletonEdges
import kotlin.math.max

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var skeleton: Skeleton? = null
    private var srcW = 1f
    private var srcH = 1f

    private val bonePaint = Paint().apply {
        color = Color.parseColor("#4DD0E1")
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }
    private val jointPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    fun update(sk: Skeleton?, sourceWidth: Int, sourceHeight: Int) {
        skeleton = sk
        srcW = max(sourceWidth.toFloat(), 1f)
        srcH = max(sourceHeight.toFloat(), 1f)
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val sk = skeleton ?: return
        // FILL_CENTER: uniform scale to cover the view, centered.
        val scale = max(width / srcW, height / srcH)
        val dx = (width - srcW * scale) / 2f
        val dy = (height - srcH * scale) / 2f
        fun px(nx: Double) = (nx.toFloat() * srcW) * scale + dx
        fun py(ny: Double) = (ny.toFloat() * srcH) * scale + dy

        for ((a, b) in skeletonEdges) {
            val la = sk[a.ordinal]
            val lb = sk[b.ordinal]
            if (la.confidence < VIS_MIN || lb.confidence < VIS_MIN) continue
            canvas.drawLine(px(la.x), py(la.y), px(lb.x), py(lb.y), bonePaint)
        }
        for (lm in sk) {
            if (lm.confidence < VIS_MIN) continue
            canvas.drawCircle(px(lm.x), py(lm.y), 10f, jointPaint)
        }
    }
}

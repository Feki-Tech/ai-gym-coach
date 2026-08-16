// Single-screen workout session: camera preview + skeleton overlay + HUD,
// exercise picker (incl. auto-detect), spoken cues, JSON session log with the
// same shape as the desktop's workout_log.json — stored app-private,
// local-first like everywhere else in this project.
package com.fekitech.gymcoach

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.fekitech.gymcoach.core.SessionEngine
import com.fekitech.gymcoach.core.SessionRecord
import com.fekitech.gymcoach.core.Skeleton
import com.fekitech.gymcoach.core.displayName
import com.fekitech.gymcoach.core.exerciseOrder
import com.fekitech.gymcoach.core.specs
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var hudExercise: TextView
    private lateinit var hudReps: TextView
    private lateinit var hudCue: TextView

    private var analyzer: PoseAnalyzer? = null
    private var engine = SessionEngine("auto")
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var sessionStartMs = 0L
    private var frameW = 480
    private var frameH = 640
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
            if (ok) startCamera()
            else Toast.makeText(this, R.string.camera_needed,
                                Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        hudExercise = findViewById(R.id.hud_exercise)
        hudReps = findViewById(R.id.hud_reps)
        hudCue = findViewById(R.id.hud_cue)

        val choices = listOf("auto") + exerciseOrder
        val spinner = findViewById<Spinner>(R.id.exercise_spinner)
        spinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            choices.map { if (it == "auto") getString(R.string.auto_detect)
                          else displayName(it) })
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?,
                                        pos: Int, id: Long) {
                restartSession(choices[pos])
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        findViewById<Button>(R.id.finish_button).setOnClickListener {
            finishSession()
        }

        tts = TextToSpeech(this, this)
        sessionStartMs = SystemClock.uptimeMillis()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    override fun onInit(status: Int) {
        ttsReady = status == TextToSpeech.SUCCESS
        if (ttsReady) tts?.language = Locale.US
    }

    private fun restartSession(exercise: String) {
        engine = SessionEngine(exercise)
        sessionStartMs = SystemClock.uptimeMillis()
        val hint = specs[exercise]?.cameraHint ?: getString(R.string.hint_auto)
        hudCue.text = hint
        hudReps.text = "0"
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analyzer?.close()
            analyzer = PoseAnalyzer(this, mirrored = true) { sk, t ->
                onSkeleton(sk, t)
            }
            analysis.setAnalyzer(analysisExecutor) { proxy ->
                val rot = proxy.imageInfo.rotationDegrees
                frameW = if (rot % 180 == 0) proxy.width else proxy.height
                frameH = if (rot % 180 == 0) proxy.height else proxy.width
                analyzer?.analyze(proxy, SystemClock.uptimeMillis())
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onSkeleton(sk: Skeleton, t: Double) {
        val out = engine.process(sk, t)
        runOnUiThread {
            overlay.update(sk, frameW, frameH)
            hudExercise.text = when {
                out.hud.detecting -> getString(R.string.detecting)
                out.hud.exercise != null -> displayName(out.hud.exercise!!)
                else -> ""
            }
            hudReps.text = out.hud.plankHold?.let {
                getString(R.string.hold_s, it)
            } ?: out.hud.repCount.toString()
            if (out.hud.cue.isNotEmpty()) hudCue.text = out.hud.cue
            for (cue in out.spokenCues) speak(cue)
        }
    }

    private fun speak(msg: String) {
        if (ttsReady) tts?.speak(msg, TextToSpeech.QUEUE_ADD, null,
                                 msg.hashCode().toString())
    }

    private fun finishSession() {
        val durationS =
            (SystemClock.uptimeMillis() - sessionStartMs) / 1000.0
        val started = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss",
                                       Locale.US).format(Date())
        val record = engine.finish(durationS, started)
        appendLog(record)
        val summary = getString(
            R.string.session_done, record.summary.reps,
            record.summary.avgScore?.let { "%.0f".format(it) } ?: "–")
        speak(summary)
        Toast.makeText(this, summary, Toast.LENGTH_LONG).show()
        restartSession("auto")
    }

    /** Append to files/workout_log.json — same record shape as the desktop. */
    private fun appendLog(r: SessionRecord) {
        val file = File(filesDir, "workout_log.json")
        val arr = if (file.exists()) JSONArray(file.readText()) else JSONArray()
        arr.put(JSONObject().apply {
            put("started", r.started)
            put("exercise", r.exercise)
            put("duration_s", r.durationS)
            put("reps", JSONArray().apply {
                r.reps.forEach { rep ->
                    put(JSONObject().apply {
                        put("n", rep.n); put("score", rep.score)
                        put("eccentric_s", rep.eccentricS)
                        put("concentric_s", rep.concentricS)
                        put("min_angle", rep.minAngle)
                        put("velocity", rep.velocity)
                        put("faults", JSONArray(rep.faults))
                    })
                }
            })
            r.plank?.let {
                put("plank", JSONObject().apply {
                    put("total_hold_s", it.totalHoldS)
                    put("best_streak_s", it.bestStreakS)
                })
            }
            put("summary", JSONObject().apply {
                put("reps", r.summary.reps)
                put("avg_score", r.summary.avgScore)
                put("avg_concentric_s", r.summary.avgConcentricS)
                put("fault_counts", JSONObject(r.summary.faultCounts))
                put("velocity_loss_pct", r.summary.velocityLossPct)
            })
        })
        file.writeText(arr.toString(2))
    }

    override fun onDestroy() {
        super.onDestroy()
        analyzer?.close()
        tts?.shutdown()
        analysisExecutor.shutdown()
    }
}

// Exercise definitions — thresholds identical to the desktop prototype
// and iOS CoachCore.
package com.fekitech.gymcoach.core

enum class ExerciseMode { REPS, HOLD }

data class ExerciseSpec(
    val name: String,
    val signal: SignalKey,            // angle driving the FSM (down, then up)
    val startBelow: Double = 0.0,     // below this => rep started
    val bottomBelow: Double = 0.0,    // deep enough for full ROM
    val lockoutAbove: Double = 0.0,   // back above this => rep complete
    val concentricPhase: String = "ascent", // angle direction of the lift
    val minRepS: Double = 0.8,
    val minConcentricS: Double = 0.6, // faster => "slow down"
    val mode: ExerciseMode = ExerciseMode.REPS,
    val cameraHint: String = "Place the phone to see your full body from the side.",
)

val specs: Map<String, ExerciseSpec> = mapOf(
    "squat" to ExerciseSpec("squat", SignalKey.KNEE, startBelow = 150.0,
        bottomBelow = 100.0, lockoutAbove = 165.0,
        cameraHint = "Side view or 45°, whole body in frame."),
    "pushup" to ExerciseSpec("pushup", SignalKey.ELBOW, startBelow = 140.0,
        bottomBelow = 95.0, lockoutAbove = 155.0, minConcentricS = 0.4),
    "bench" to ExerciseSpec("bench", SignalKey.ELBOW, startBelow = 140.0,
        bottomBelow = 90.0, lockoutAbove = 160.0,
        cameraHint = "Side view, head to hips in frame."),
    "deadlift" to ExerciseSpec("deadlift", SignalKey.HIP, startBelow = 150.0,
        bottomBelow = 100.0, lockoutAbove = 165.0),
    "lunge" to ExerciseSpec("lunge", SignalKey.KNEE, startBelow = 150.0,
        bottomBelow = 110.0, lockoutAbove = 165.0,
        cameraHint = "Side view or 45°, front knee visible."),
    "shoulder_press" to ExerciseSpec("shoulder_press", SignalKey.ELBOW,
        startBelow = 150.0, bottomBelow = 100.0, lockoutAbove = 160.0,
        cameraHint = "Face the camera, arms fully in frame."),
    "curl" to ExerciseSpec("curl", SignalKey.ELBOW, startBelow = 140.0,
        bottomBelow = 70.0, lockoutAbove = 155.0,
        concentricPhase = "descent", minConcentricS = 0.5),
    "pullup" to ExerciseSpec("pullup", SignalKey.ELBOW, startBelow = 140.0,
        bottomBelow = 80.0, lockoutAbove = 160.0,
        concentricPhase = "descent",
        cameraHint = "Face the camera, bar and body in frame."),
    "plank" to ExerciseSpec("plank", SignalKey.BODY_LINE,
        mode = ExerciseMode.HOLD),
)

/** Display order for pickers. */
val exerciseOrder = listOf("squat", "pushup", "bench", "deadlift", "lunge",
                           "shoulder_press", "curl", "pullup", "plank")

fun displayName(exercise: String): String =
    exercise.replace('_', ' ')
        .split(' ')
        .joinToString(" ") { w -> w.replaceFirstChar { it.uppercase() } }

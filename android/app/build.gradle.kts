import java.net.URI

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.fekitech.gymcoach"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.fekitech.gymcoach"
        minSdk = 26 // XML-only adaptive icon; MediaPipe Tasks needs 24+
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false // demo distribution is the debug APK
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.core:core-ktx:1.13.1")
    // Pose landmarks — MediaPipe Tasks Vision (roadmap: "MediaPipe Tasks, Kotlin")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    testImplementation("junit:junit:4.13.2")
}

// The pose model is fetched at build time (like the desktop's ensure_model),
// never committed: ~5 MB binary, and *.task is gitignored repo-wide.
val poseModelFile = layout.projectDirectory
    .file("src/main/assets/pose_landmarker_lite.task").asFile
val poseModelUrl = "https://storage.googleapis.com/mediapipe-models/" +
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

val downloadPoseModel = tasks.register("downloadPoseModel") {
    outputs.file(poseModelFile)
    doLast {
        if (!poseModelFile.exists()) {
            poseModelFile.parentFile.mkdirs()
            logger.lifecycle("Downloading pose model -> $poseModelFile")
            URI(poseModelUrl).toURL().openStream().use { input ->
                poseModelFile.outputStream().use { input.copyTo(it) }
            }
        }
    }
}
tasks.named("preBuild") { dependsOn(downloadPoseModel) }

// Camera capture + Apple Vision body-pose detection.
// Emits CoachCore Skeletons in top-left-origin normalized coordinates.

import AVFoundation
import Vision
import CoachCore
import CoreGraphics

final class CameraService: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "gymcoach.camera")
    private let request = VNDetectHumanBodyPoseRequest()
    private var configured = false

    /// Called on the camera queue with the detected skeleton (nil when no
    /// person is visible) and the pixel-buffer size for overlay mapping.
    var onFrame: ((Skeleton?, CGSize) -> Void)?

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard granted, let self else { return }
            self.queue.async {
                self.configure()
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configure() {
        guard !configured else { return }
        configured = true
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)
        if let conn = output.connection(with: .video) {
            // deliver upright portrait buffers so overlay math stays simple
            if #available(iOS 17.0, *) {
                if conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            } else if conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
            }
        }
        session.commitConfiguration()
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let size = CGSize(width: CVPixelBufferGetWidth(pb),
                          height: CVPixelBufferGetHeight(pb))
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
        guard (try? handler.perform([request])) != nil,
              let obs = request.results?.first else {
            onFrame?(nil, size)
            return
        }
        onFrame?(Self.skeleton(from: obs), size)
    }

    static let jointMap: [(Joint, VNHumanBodyPoseObservation.JointName)] = [
        (.nose, .nose), (.leftEar, .leftEar), (.rightEar, .rightEar),
        (.leftShoulder, .leftShoulder), (.rightShoulder, .rightShoulder),
        (.leftElbow, .leftElbow), (.rightElbow, .rightElbow),
        (.leftWrist, .leftWrist), (.rightWrist, .rightWrist),
        (.leftHip, .leftHip), (.rightHip, .rightHip),
        (.leftKnee, .leftKnee), (.rightKnee, .rightKnee),
        (.leftAnkle, .leftAnkle), (.rightAnkle, .rightAnkle),
    ]

    /// Vision uses a bottom-left origin; CoachCore uses top-left (y-down).
    static func skeleton(from obs: VNHumanBodyPoseObservation) -> Skeleton {
        var skel = [Landmark](repeating: Landmark(x: 0, y: 0, confidence: 0),
                              count: Joint.allCases.count)
        guard let pts = try? obs.recognizedPoints(.all) else { return skel }
        for (joint, name) in jointMap {
            if let p = pts[name] {
                skel[joint.rawValue] = Landmark(x: Double(p.location.x),
                                                y: 1 - Double(p.location.y),
                                                confidence: Double(p.confidence))
            }
        }
        return skel
    }
}

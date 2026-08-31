// Camera capture + Apple Vision body-pose detection.
// Emits CoachCore Skeletons in top-left-origin normalized coordinates.
//
// The camera is selectable: back / front / ultra-wide / telephoto, and on
// iOS 17+ external cameras plugged into the USB-C port (a webcam on a
// tripod sees your whole squat better than a propped-up phone). The choice
// persists; the front camera is mirrored like a gym mirror, with the
// buffer mirrored the same way so the skeleton overlay stays aligned.

import AVFoundation
import Vision
import Combine
import CoachCore
import CoreGraphics

struct CameraInfo: Identifiable, Equatable {
    let id: String                 // uniqueID
    let name: String
    let position: AVCaptureDevice.Position
    let isExternal: Bool

    var label: String {
        if isExternal { return name }
        switch position {
        case .front: return String(format: NSLocalizedString("%@ (front)", comment: ""), name)
        case .back: return name
        default: return name
        }
    }
}

final class CameraService: NSObject, ObservableObject,
                           AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    @Published private(set) var current: CameraInfo?
    @Published private(set) var available: [CameraInfo] = []

    private let queue = DispatchQueue(label: "gymcoach.camera")
    private let request = VNDetectHumanBodyPoseRequest()
    private let output = AVCaptureVideoDataOutput()
    private var configured = false
    private static let prefKey = "camera.uid"

    /// Called on the camera queue with the detected skeleton (nil when no
    /// person is visible) and the pixel-buffer size for overlay mapping.
    var onFrame: ((CoachCore.Skeleton?, CGSize) -> Void)?

    // MARK: - discovery

    static func discover() -> [AVCaptureDevice] {
        var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera,
                                                   .builtInUltraWideCamera,
                                                   .builtInTelephotoCamera]
        if #available(iOS 17.0, *) {
            types.append(.external)          // USB-C webcams (iPad, newer iPhones)
        }
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: types, mediaType: .video, position: .unspecified)
        return discovery.devices
    }

    private static func info(_ d: AVCaptureDevice) -> CameraInfo {
        var external = false
        if #available(iOS 17.0, *) { external = d.deviceType == .external }
        return CameraInfo(id: d.uniqueID, name: d.localizedName,
                          position: d.position, isExternal: external)
    }

    func refreshAvailable() {
        let infos = Self.discover().map(Self.info)
        DispatchQueue.main.async { self.available = infos }
    }

    // MARK: - lifecycle

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard granted, let self else { return }
            self.queue.async {
                self.configureIfNeeded()
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    /// Switch to the camera with this uniqueID (persisted for next time).
    func select(_ id: String) {
        UserDefaults.standard.set(id, forKey: Self.prefKey)
        queue.async {
            guard self.configured,
                  let device = Self.discover().first(where: { $0.uniqueID == id })
            else { return }
            self.attach(device)
        }
    }

    private func preferredDevice() -> AVCaptureDevice? {
        let devices = Self.discover()
        if let want = UserDefaults.standard.string(forKey: Self.prefKey),
           let d = devices.first(where: { $0.uniqueID == want }) {
            return d
        }
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video,
                                       position: .back) ?? devices.first
    }

    private func configureIfNeeded() {
        guard !configured else { return }
        configured = true
        refreshAvailable()
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
        if let device = preferredDevice() { attach(device) }
    }

    /// Swap the session's input to `device` (camera queue only).
    private func attach(_ device: AVCaptureDevice) {
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        guard let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        // externals may not do 720p; fall back rather than fail silently
        if !session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .high
        } else {
            session.sessionPreset = .hd1280x720
        }
        if let conn = output.connection(with: .video) {
            // deliver upright portrait buffers so overlay math stays simple
            if #available(iOS 17.0, *) {
                if conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            } else if conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
            }
            // gym-mirror view on the front camera; the preview layer mirrors
            // itself for front cameras, so mirroring the buffer keeps the
            // skeleton overlay aligned with what the athlete sees
            if conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = (device.position == .front)
            }
        }
        session.commitConfiguration()
        let info = Self.info(device)
        DispatchQueue.main.async { self.current = info }
    }

    // MARK: - frames

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

    static let jointMap: [(CoachCore.Joint, VNHumanBodyPoseObservation.JointName)] = [
        (.nose, .nose), (.leftEar, .leftEar), (.rightEar, .rightEar),
        (.leftShoulder, .leftShoulder), (.rightShoulder, .rightShoulder),
        (.leftElbow, .leftElbow), (.rightElbow, .rightElbow),
        (.leftWrist, .leftWrist), (.rightWrist, .rightWrist),
        (.leftHip, .leftHip), (.rightHip, .rightHip),
        (.leftKnee, .leftKnee), (.rightKnee, .rightKnee),
        (.leftAnkle, .leftAnkle), (.rightAnkle, .rightAnkle),
    ]

    /// Vision uses a bottom-left origin; CoachCore uses top-left (y-down).
    static func skeleton(from obs: VNHumanBodyPoseObservation) -> CoachCore.Skeleton {
        var skel = [CoachCore.Landmark](
            repeating: CoachCore.Landmark(x: 0, y: 0, confidence: 0),
            count: CoachCore.Joint.allCases.count)
        guard let pts = try? obs.recognizedPoints(.all) else { return skel }
        for (joint, name) in jointMap {
            if let p = pts[name] {
                skel[joint.rawValue] = CoachCore.Landmark(
                    x: Double(p.location.x),
                    y: 1 - Double(p.location.y),
                    confidence: Double(p.confidence))
            }
        }
        return skel
    }
}

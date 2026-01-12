# camera_with_pose_tasks.py
# RealSense D435i Camera Manager with MediaPipe Pose (Tasks API)
# 集成成功的 demo 到 D435i 相机系统

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import threading
import queue

# MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_AVAILABLE = True
    print("[Pose] MediaPipe Tasks API loaded successfully")
except ImportError as e:
    print(f"[Pose] MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False

# MediaPipe Pose 33 landmarks connections
POSE_CONNECTIONS_33 = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    # Left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    # Left leg
    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (27, 31),
    # Right leg
    (24, 26), (26, 28),
    (28, 30), (30, 32),
    (28, 32),
]


class CameraManagerWithPose:
    """RealSense D435i Camera Manager with MediaPipe Pose (Tasks API)"""

    def __init__(self, width=640, height=480, fps=30, enable_pose=True, model_path=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_pose = enable_pose and MEDIAPIPE_AVAILABLE

        # Frame queues (thread-safe)
        self.rgb_queue = queue.Queue(maxsize=2)
        self.depth_queue = queue.Queue(maxsize=2)

        # Control flags
        self.is_running = False
        self.thread = None

        # RealSense pipeline
        self.pipeline = None
        self.config = None

        # MediaPipe Pose Landmarker
        self.landmarker = None
        self.timestamp_ms = 0

        if self.enable_pose:
            if not MEDIAPIPE_AVAILABLE:
                print("[Pose] ⚠️ 姿态检测已禁用（MediaPipe 不可用）")
                self.enable_pose = False
            else:
                self._init_pose_landmarker(model_path)

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def _init_pose_landmarker(self, model_path):
        """Initialize MediaPipe Pose Landmarker (Tasks API)"""
        try:
            # 默认模型路径
            if model_path is None:
                model_path = "library/pose_landmarker_full.task"

            # 检查模型文件
            if not os.path.exists(model_path):
                print(f"[Pose] ⚠️ 模型文件不存在: {model_path}")
                print("[Pose] 请下载模型文件:")
                print(
                    "  URL: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
                print(f"  保存到: {model_path}")
                self.enable_pose = False
                return

            # 创建 PoseLandmarker
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            print("[Pose] MediaPipe PoseLandmarker initialized successfully")
            print(f"[Pose] Model: {model_path}")

        except Exception as e:
            print(f"[Pose] ⚠️ 初始化失败: {e}")
            self.enable_pose = False
            self.landmarker = None

    def start(self):
        """Start camera capture thread"""
        if self.is_running:
            print("[Camera] Already running")
            return

        try:
            # Create pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Enable streams
            self.config.enable_stream(rs.stream.depth, self.width, self.height,
                                      rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height,
                                      rs.format.bgr8, self.fps)

            # Start pipeline
            self.pipeline.start(self.config)

            # Start capture thread
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

            print(f"[Camera] Started successfully ({self.width}x{self.height} @ {self.fps}fps)")
            if self.enable_pose:
                print("[Camera] Pose estimation enabled (Tasks API)")
            else:
                print("[Camera] Pose estimation disabled")

        except Exception as e:
            print(f"[Camera] Failed to start: {e}")
            self.is_running = False
            raise

    def stop(self):
        """Stop camera capture"""
        if not self.is_running:
            return

        self.is_running = False

        if self.thread:
            self.thread.join(timeout=2.0)

        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"[Camera] Error stopping pipeline: {e}")

        if self.landmarker:
            self.landmarker.close()

        print("[Camera] Stopped")

    def _make_mp_image(self, rgb_np):
        """Create MediaPipe Image object (version-robust)"""
        try:
            if hasattr(mp, "Image") and hasattr(mp, "ImageFormat"):
                return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
            if hasattr(vision, "Image") and hasattr(vision, "ImageFormat"):
                return vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_np)
        except Exception as e:
            print(f"[Pose] Error creating MP Image: {e}")
        return None

    def _draw_landmarks_points(self, image_bgr, landmarks33, radius=3):
        """Draw all 33 landmarks as circles"""
        h, w = image_bgr.shape[:2]
        for i, lm in enumerate(landmarks33):
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Green circles for keypoints
            cv2.circle(image_bgr, (cx, cy), radius, (0, 255, 0), -1)

    def _draw_connections(self, image_bgr, landmarks33, thickness=2):
        """Draw skeleton connections (lines) between landmarks"""
        h, w = image_bgr.shape[:2]
        for a, b in POSE_CONNECTIONS_33:
            la, lb = landmarks33[a], landmarks33[b]
            ax, ay = int(la.x * w), int(la.y * h)
            bx, by = int(lb.x * w), int(lb.y * h)
            # Orange lines for skeleton
            cv2.line(image_bgr, (ax, ay), (bx, by), (0, 165, 255), thickness)

    def _draw_pose_info(self, image_bgr, has_pose):
        """Draw pose detection status"""
        if has_pose:
            cv2.putText(image_bgr, "Person Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image_bgr, "No Person Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _process_pose(self, rgb_image):
        """Process pose estimation on RGB image"""
        if not self.enable_pose or self.landmarker is None:
            return rgb_image

        try:
            # Convert BGR to RGB
            rgb_for_mp = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = self._make_mp_image(rgb_for_mp)
            if mp_image is None:
                return rgb_image

            # Detect pose (VIDEO mode requires increasing timestamps)
            self.timestamp_ms = int(time.time() * 1000)
            result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

            # Draw results
            annotated_image = rgb_image.copy()

            if result.pose_landmarks:
                landmarks33 = result.pose_landmarks[0]

                # Draw skeleton: connections + keypoints
                self._draw_connections(annotated_image, landmarks33, thickness=2)
                self._draw_landmarks_points(annotated_image, landmarks33, radius=3)
                self._draw_pose_info(annotated_image, has_pose=True)

            else:
                self._draw_pose_info(annotated_image, has_pose=False)

            # Update FPS
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 1.0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()

            # Draw FPS
            cv2.putText(annotated_image, f"Pose FPS: {self.current_fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            return annotated_image

        except Exception as e:
            print(f"[Pose] Processing error: {e}")
            return rgb_image

    def _capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        while self.is_running:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Process pose estimation on RGB image
                color_image_with_pose = self._process_pose(color_image)

                # Apply colormap to depth image
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Update queues (non-blocking)
                try:
                    if self.rgb_queue.full():
                        self.rgb_queue.get_nowait()
                    self.rgb_queue.put_nowait(color_image_with_pose)
                except queue.Full:
                    pass

                try:
                    if self.depth_queue.full():
                        self.depth_queue.get_nowait()
                    self.depth_queue.put_nowait(depth_colormap)
                except queue.Full:
                    pass

            except Exception as e:
                if self.is_running:
                    print(f"[Camera] Capture error: {e}")
                time.sleep(0.1)

    def get_latest_frames(self):
        """Get latest RGB and depth frames"""
        rgb_image = None
        depth_image = None

        try:
            rgb_image = self.rgb_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            depth_image = self.depth_queue.get_nowait()
        except queue.Empty:
            pass

        return rgb_image, depth_image

    def is_active(self):
        """Check if camera is actively running"""
        return self.is_running

    def toggle_pose(self, enable):
        """Enable or disable pose estimation"""
        if not MEDIAPIPE_AVAILABLE:
            print("[Pose] ⚠️ MediaPipe 不可用，无法启用姿态检测")
            return

        if enable and self.landmarker is None:
            self._init_pose_landmarker(None)

        self.enable_pose = enable
        print(f"[Pose] Pose estimation {'enabled' if enable else 'disabled'}")


# Standalone test function
def test_camera_with_pose():
    """Test camera functionality with pose estimation"""
    print("Testing RealSense D435i Camera with MediaPipe Pose (Tasks API)...")

    if not MEDIAPIPE_AVAILABLE:
        print("\n⚠️ 警告: MediaPipe 不可用")
        print("姿态检测功能将被禁用，但相机仍可正常工作\n")

    camera = CameraManagerWithPose(enable_pose=True)

    try:
        camera.start()

        print("\nPress 'q' to quit")
        if MEDIAPIPE_AVAILABLE:
            print("Press 'p' to toggle pose estimation")

        while True:
            rgb, depth = camera.get_latest_frames()

            if rgb is not None and depth is not None:
                images = np.hstack((rgb, depth))
                cv2.imshow('D435i RGB + Pose + Depth', images)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') and MEDIAPIPE_AVAILABLE:
                camera.toggle_pose(not camera.enable_pose)

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera_with_pose()
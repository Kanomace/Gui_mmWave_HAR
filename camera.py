# camera_module.py
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import queue
import time


class CameraManager:
    """RealSense D435i Camera Manager with thread-safe frame queue"""

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        # Frame queues (thread-safe)
        self.rgb_queue = queue.Queue(maxsize=2)
        self.depth_queue = queue.Queue(maxsize=2)

        # Control flags
        self.is_running = False
        self.thread = None

        # RealSense pipeline
        self.pipeline = None
        self.config = None

    def start(self):
        """Start camera capture thread"""
        if self.is_running:
            print("[Camera] Already running")
            return

        try:
            # Create pipeline (exactly like your original code)
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Enable streams (exactly like your original code)
            self.config.enable_stream(rs.stream.depth, self.width, self.height,
                                      rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height,
                                      rs.format.bgr8, self.fps)

            # Start pipeline (exactly like your original code)
            self.pipeline.start(self.config)

            # Start capture thread
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

            print(f"[Camera] Started successfully ({self.width}x{self.height} @ {self.fps}fps)")

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

        print("[Camera] Stopped")

    def _capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        while self.is_running:
            try:
                # Wait for frames (exactly like your original code)
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert to numpy arrays (exactly like your original code)
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap to depth image (exactly like your original code)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Update queues (non-blocking)
                try:
                    # Clear old frames if queue is full
                    if self.rgb_queue.full():
                        self.rgb_queue.get_nowait()
                    self.rgb_queue.put_nowait(color_image)
                except queue.Full:
                    pass

                try:
                    if self.depth_queue.full():
                        self.depth_queue.get_nowait()
                    self.depth_queue.put_nowait(depth_colormap)
                except queue.Full:
                    pass

            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    print(f"[Camera] Capture error: {e}")
                time.sleep(0.1)

    def get_latest_frames(self):
        """
        Get latest RGB and depth frames
        Returns: (rgb_image, depth_colormap) or (None, None) if no frames available
        """
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


# Standalone test function (exactly like your original code)
def test_camera():
    """Test camera functionality independently"""
    print("Testing RealSense D435i Camera...")

    camera = CameraManager()

    try:
        camera.start()

        print("Press 'q' to quit")

        while True:
            rgb, depth = camera.get_latest_frames()

            if rgb is not None and depth is not None:
                # Stack horizontally for display (exactly like your original code)
                images = np.hstack((rgb, depth))
                cv2.imshow('D435i RGB + Depth', images)

            # Press 'q' to quit (exactly like your original code)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()
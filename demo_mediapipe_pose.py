# demo_mediapipe_pose_full33.py
# MediaPipe Tasks API (PoseLandmarker) - tasks-only compatible
# Camera -> FULL 33 landmarks (x,y,z,visibility) -> overlay + save npy/json
# Save ONLY 33 points (no 17-joint mapping).

import os
import time
import json
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def landmarks33_to_numpy(landmarks33):
    """
    landmarks33: list of 33 landmarks, each has x,y,z,visibility (visibility may be missing in some builds)
    return: (33,4) float32 [x,y,z,visibility]
    """
    pts = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks33):
        pts[i, 0] = lm.x
        pts[i, 1] = lm.y
        pts[i, 2] = lm.z
        pts[i, 3] = getattr(lm, "visibility", 1.0)
    return pts


def draw_landmarks_points(image_bgr, landmarks33, radius=3):
    """Draw all 33 landmarks as circles with index labels."""
    h, w = image_bgr.shape[:2]
    for i, lm in enumerate(landmarks33):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image_bgr, (cx, cy), radius, (0, 255, 0), -1)
        # index label
        cv2.putText(
            image_bgr,
            str(i),
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# MediaPipe Pose connections for 33 landmarks (standard topology)
# Each tuple is (start_idx, end_idx)
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


def draw_connections(image_bgr, landmarks33, connections=POSE_CONNECTIONS_33, thickness=2):
    """Draw skeleton connections (lines) between landmarks."""
    h, w = image_bgr.shape[:2]
    for a, b in connections:
        la, lb = landmarks33[a], landmarks33[b]
        ax, ay = int(la.x * w), int(la.y * h)
        bx, by = int(lb.x * w), int(lb.y * h)
        cv2.line(image_bgr, (ax, ay), (bx, by), (0, 200, 255), thickness)


def save_landmarks33(out_dir, frame_idx, pts33, img_shape):
    os.makedirs(out_dir, exist_ok=True)
    h, w = img_shape[:2]

    npy_path = os.path.join(out_dir, f"frame_{frame_idx:06d}_mp33.npy")
    json_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.json")

    np.save(npy_path, pts33)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "frame": int(frame_idx),
                "image_hw": [int(h), int(w)],
                "mp33_xyzw": pts33.tolist(),
                "timestamp": float(time.time()),
                "mediapipe_version": getattr(mp, "__version__", None),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def make_mp_image(rgb_np: np.ndarray):
    """Create a MediaPipe Image object in a version-robust way."""
    if hasattr(mp, "Image") and hasattr(mp, "ImageFormat"):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
    if hasattr(vision, "Image") and hasattr(vision, "ImageFormat"):
        return vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_np)
    raise RuntimeError("Cannot construct MediaPipe Image: mp.Image / vision.Image not found.")


def main():
    # ================= Camera =================
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try changing camera id or removing CAP_DSHOW.")

    # ================= MediaPipe PoseLandmarker =================
    model_path = "library/pose_landmarker_full.task"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Download and place it as:\n"
            f"  models/pose_landmarker_full.task\n"
            f"Download URL:\n"
            f"  https://storage.googleapis.com/mediapipe-models/"
            f"pose_landmarker/pose_landmarker_full/float16/latest/"
            f"pose_landmarker_full.task"
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # ================= Output =================
    out_dir = "outputs/mediapipe_pose33"
    save_every_n_frames = 1
    min_vis_to_save = 0.2

    frame_idx = 0
    saved = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = make_mp_image(rgb)

        # VIDEO mode requires increasing timestamps
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks33 = result.pose_landmarks[0]

            # Draw skeleton: lines + points + indices
            draw_connections(frame_bgr, landmarks33, thickness=2)
            draw_landmarks_points(frame_bgr, landmarks33, radius=3)

            pts33 = landmarks33_to_numpy(landmarks33)
            vis_mean = float(np.mean(pts33[:, 3]))

            if frame_idx % save_every_n_frames == 0 and vis_mean >= min_vis_to_save:
                save_landmarks33(out_dir, frame_idx, pts33, frame_bgr.shape)
                saved += 1

            cv2.putText(
                frame_bgr,
                f"frame={frame_idx} saved={saved} vis_mean={vis_mean:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame_bgr,
                f"frame={frame_idx} (no pose)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("MediaPipe PoseLandmarker 33pts (q to quit)", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[Done] Saved {saved} frames to: {out_dir}")


if __name__ == "__main__":
    main()

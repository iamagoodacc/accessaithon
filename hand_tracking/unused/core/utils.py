"""
Shared utility functions for BSL gesture recognition
"""
import cv2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult


# Renders landmark overlays on the video feed
def render_handle(
    hand_landmarks: HandLandmarkerResult,
    pose_landmarks: PoseLandmarkerResult,
    img: cv2.typing.MatLike
):
    """Render hand and pose landmarks on the image"""
    h, w = img.shape[:2]

    # Draw hand landmarks with index numbers
    for hand in hand_landmarks.hand_landmarks:
        for idx, lm in enumerate(hand):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.putText(img, str(idx), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw pose landmarks as circles
    for pose in pose_landmarks.pose_landmarks:
        for idx, lm in enumerate(pose):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

import cv2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from video import run


def default_handle(
    hand_landmarks: list[list[NormalizedLandmark]],
    face_landmarks: list[list[NormalizedLandmark]],
    pose_landmarks: list[list[NormalizedLandmark]],
    img: cv2.typing.MatLike
):
    h, w = img.shape[:2]

    for hand in hand_landmarks:
        for idx, lm in enumerate(hand):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.putText(img, str(idx), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    for face in face_landmarks:
        for lm in face:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    for pose in pose_landmarks:
        for idx, lm in enumerate(pose):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 4, (255, 0, 0), -1)


if __name__ == '__main__':
    run(default_handle)

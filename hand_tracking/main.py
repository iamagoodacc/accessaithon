from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import cv2
from video import run

def default_handle(hand_landmarks: list[list[NormalizedLandmark]], img: cv2.typing.MatLike):
    for hand_landmarks in hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            cv2.putText(img, str(idx), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),
                        thickness=2)

if __name__ == '__main__':
    run(default_handle)
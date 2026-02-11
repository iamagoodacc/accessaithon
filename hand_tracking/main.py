import cv2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from data import collect_handle
import torch
import numpy as np
from model import RecognitionModel
from collections import deque

from video import run

SIGNS = ["hello", "yes"]
sample_data = np.load(f"data/{SIGNS[0]}.npy")
input_size = sample_data.shape[2]

CONFIDENCE_THRESHOLD = 0.7
FRAME_COUNT = 30
INTERVAL = 15

model = RecognitionModel(
    input_size=input_size,
    hidden_size=128,
    num_layers=2,
    output_size=len(SIGNS),
    dropout=0.0
)
model.load_state_dict(torch.load("recognition_model.pth"))
model.eval()

sequence_buffer = deque([])

def predict(sequence):
    x = torch.tensor(np.array(sequence), dtype=torch.float32)
    x = x.unsqueeze(0)   # (1, 30, input_size)

    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        conf, predicted = torch.max(probabilities, dim=1)

    if conf.item() < CONFIDENCE_THRESHOLD:
        return "unidentified", conf.item()

    return SIGNS[predicted.item()], conf.item()

sequences = []

# this main handle will be called whenever there is new data
def main_handle(
        hand_landmarks: HandLandmarkerResult,
        pose_landmarks: PoseLandmarkerResult,
        img: cv2.typing.MatLike
):
    frame_data = collect_handle(hand_landmarks, pose_landmarks)
    sequence_buffer.append(frame_data)

    if len(sequence_buffer) > FRAME_COUNT + INTERVAL:
        for i in range(INTERVAL):
            sequence_buffer.popleft()
        print(predict(sequence_buffer))

    render_handle(hand_landmarks, pose_landmarks, img)

def render_handle(
    hand_landmarks: HandLandmarkerResult,
    pose_landmarks: PoseLandmarkerResult,
    img: cv2.typing.MatLike
):
    h, w = img.shape[:2]

    for hand in hand_landmarks.hand_landmarks:
        for idx, lm in enumerate(hand):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.putText(img, str(idx), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    for pose in pose_landmarks.pose_landmarks:
        for idx, lm in enumerate(pose):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 4, (255, 0, 0), -1)


if __name__ == '__main__':
    run(main_handle)

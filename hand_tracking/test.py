from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import cv2
import numpy as np
import torch
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from main import render_handle
from data import collect_handle
from model import RecognitionModel
from video import run

SIGNS = ["hello", "yes"]
FRAMES_PER_SEQUENCE = 30
CONFIDENCE_THRESHOLD = 0.7

sequence_buffer = []
recording = False
prediction = "..."
confidence = 0.0

# figure out input size from saved data
sample_data = np.load(f"data/{SIGNS[0]}.npy")
input_size = sample_data.shape[2]

model = RecognitionModel(
    input_size=input_size,
    hidden_size=128,
    num_layers=2,
    output_size=len(SIGNS),
    dropout=0.0
)
model.load_state_dict(torch.load("recognition_model.pth"))
model.eval()

print(f"Model loaded — input_size={input_size}, output_size={len(SIGNS)}")

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

def main_handle(
        hand_landmarks: HandLandmarkerResult,
        pose_landmarks: PoseLandmarkerResult,
        img: cv2.typing.MatLike
):
    global sequence_buffer, recording, prediction, confidence

    if recording:
        try:
            frame_data = collect_handle(hand_landmarks, pose_landmarks)
            sequence_buffer.append(frame_data)

            # recording indicator
            cv2.putText(img, f"RECORDING {len(sequence_buffer)}/{FRAMES_PER_SEQUENCE}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # sequence complete — predict
            if len(sequence_buffer) == FRAMES_PER_SEQUENCE:
                prediction, confidence = predict(sequence_buffer)
                sequence_buffer = []
                recording = False
                print(f"Prediction: {prediction} ({confidence:.0%})")

        except Exception as e:
            print(f"Frame error: {e}")
            sequence_buffer = []
            recording = False
    else:
        # show last prediction
        color = (0, 255, 0) if prediction != "unidentified" else (0, 165, 255)
        cv2.putText(img, f"Sign: {prediction}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(img, f"Confidence: {confidence:.0%}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, "SPACE to record sign | Q to quit", (10, img.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    render_handle(hand_landmarks, pose_landmarks, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and not recording:
        recording = True
        sequence_buffer = []
        prediction = "..."
        confidence = 0.0
        print("Recording started...")
    elif key == ord('q'):
        exit()

if __name__ == "__main__":
    run(main_handle)

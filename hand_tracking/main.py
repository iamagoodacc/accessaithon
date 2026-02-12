import cv2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from data import collect_handle
from utils import render_handle
import torch
import numpy as np
from model import RecognitionModel
from collections import deque
import time
import os
import sys

from video import run

SIGNS = ["hello", "yes"]

# Check if training data exists
if not os.path.exists(f"data/{SIGNS[0]}.npy"):
    print("ERROR: Training data not found!")
    print("Please run 'python collect_data.py' first to collect training data.")
    print("Then run 'python training.py' to train the model.")
    sys.exit(1)

sample_data = np.load(f"data/{SIGNS[0]}.npy")
input_size = sample_data.shape[2]

# Check if model exists
if not os.path.exists("recognition_model.pth"):
    print("ERROR: Trained model not found!")
    print("Please run 'python training.py' to train the model first.")
    sys.exit(1)

# Configuration
CONFIDENCE_THRESHOLD = 0.6  # Lowered for better detection
FRAME_COUNT = 30  # Number of frames per gesture
PREDICTION_INTERVAL = 10  # Predict every N frames (sliding window)
DEBOUNCE_TIME = 2.0  # Seconds to wait before detecting same gesture again
MAX_TEXT_LENGTH = 100  # Maximum characters in text buffer

model = RecognitionModel(
    input_size=input_size,
    hidden_size=128,
    num_layers=2,
    output_size=len(SIGNS),
    dropout=0.0
)
model.load_state_dict(torch.load("recognition_model.pth"))
model.eval()

# Buffers and state
sequence_buffer = deque(maxlen=FRAME_COUNT)  # Automatically removes old frames
text_sequence = []  # Accumulated detected gestures
last_prediction = None
last_prediction_time = 0
frame_count = 0

def predict(sequence):
    """Predict gesture from sequence of frames"""
    if len(sequence) < FRAME_COUNT:
        return None, 0.0

    # Take last FRAME_COUNT frames
    seq = list(sequence)[-FRAME_COUNT:]

    x = torch.tensor(np.array(seq), dtype=torch.float32)
    x = x.unsqueeze(0)   # (1, 30, input_size)

    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        conf, predicted = torch.max(probabilities, dim=1)

    if conf.item() < CONFIDENCE_THRESHOLD:
        return None, conf.item()

    return SIGNS[predicted.item()], conf.item()

# this main handle will be called whenever there is new data
def main_handle(
        hand_landmarks: HandLandmarkerResult,
        pose_landmarks: PoseLandmarkerResult,
        img: cv2.typing.MatLike
):
    global sequence_buffer, text_sequence, last_prediction, last_prediction_time, frame_count

    try:
        # Collect frame data
        frame_data = collect_handle(hand_landmarks, pose_landmarks)
        sequence_buffer.append(frame_data)
        frame_count += 1

        current_prediction = None
        current_confidence = 0.0

        # Predict every PREDICTION_INTERVAL frames once we have enough frames
        if len(sequence_buffer) >= FRAME_COUNT and frame_count % PREDICTION_INTERVAL == 0:
            current_prediction, current_confidence = predict(sequence_buffer)

            # Add to text sequence if valid and not duplicate
            if current_prediction is not None:
                current_time = time.time()
                time_since_last = current_time - last_prediction_time

                # Debouncing: only add if different gesture or enough time has passed
                if (last_prediction != current_prediction or time_since_last > DEBOUNCE_TIME):
                    text_sequence.append(current_prediction)
                    last_prediction = current_prediction
                    last_prediction_time = current_time

                    # Limit text length
                    if len(text_sequence) > MAX_TEXT_LENGTH:
                        text_sequence.pop(0)

                    print(f"✓ Detected: {current_prediction} ({current_confidence:.1%})")

        # Display on screen
        render_ui(img, current_prediction if frame_count % PREDICTION_INTERVAL == 0 else last_prediction,
                 current_confidence if frame_count % PREDICTION_INTERVAL == 0 else 0.0)

    except Exception as e:
        print(f"Error in main_handle: {e}")
        # Don't clear buffer on error, just skip this frame

    render_handle(hand_landmarks, pose_landmarks, img)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Clear text
        text_sequence = []
        last_prediction = None
        print("Text cleared")
    elif key == ord('r'):  # Reset buffer
        sequence_buffer.clear()
        frame_count = 0
        print("Buffer reset")

def render_ui(img, current_prediction, confidence):
    """Render UI elements on the image"""
    h, w = img.shape[:2]

    # Semi-transparent overlay for text area
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Current prediction status
    buffer_status = f"Buffer: {len(sequence_buffer)}/{FRAME_COUNT}"
    cv2.putText(img, buffer_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if current_prediction:
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(img, f"Current: {current_prediction} ({confidence:.0%})",
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(img, "Detecting...", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    # Accumulated text sequence
    text_display = " ".join(text_sequence[-10:]) if text_sequence else "[No gestures detected]"
    cv2.putText(img, f"Text: {text_display}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Full text at bottom (if long)
    if len(text_sequence) > 10:
        full_text = " ".join(text_sequence)
        cv2.putText(img, f"Full: ...{full_text[-50:]}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Instructions
    cv2.putText(img, "C: Clear | R: Reset | Q: Quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


if __name__ == '__main__':
    run(main_handle)

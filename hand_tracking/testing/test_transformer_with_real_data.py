"""
Live inference — opens webcam and identifies signs in real time.
Press Q to quit.
"""
import os
import sys
import time
import collections
import torch
import cv2
import mediapipe as mp

os.environ["GLOG_minloglevel"] = "2"

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from core.transformer import CtcRecognitionModel
from core.data import collect_handle, NUM_LANDMARKS_IN_HAND, POSE_LANDMARKS_IDX_LIST, DIMENSIONS
from training.transformer_data import draw_landmarks, download_models

MODEL_PATH = "model.pt"

VOCABULARY = {
    0: "blank",
    1: "Hello",
    2: "I",
    3: "You",
    4: "Want",
    5: "Apple",
}

LHAND_FEATURES = NUM_LANDMARKS_IN_HAND * DIMENSIONS
RHAND_FEATURES = NUM_LANDMARKS_IN_HAND * DIMENSIONS
POSE_FEATURES  = len(POSE_LANDMARKS_IDX_LIST) * DIMENSIONS

# how many frames to accumulate before running inference
# at ~30fps, 60 frames = ~2 seconds of signing
WINDOW_SIZE = 240


def decode(logits: torch.Tensor) -> list[str]:
    """Greedy CTC decode — collapse repeats and remove blanks."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    indices   = log_probs.argmax(dim=-1).squeeze(0).tolist()
    decoded   = []
    prev      = None
    for idx in indices:
        if idx != prev:
            if idx != 0:
                decoded.append(VOCABULARY.get(idx, f"?{idx}"))
        prev = idx
    return decoded


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"No model found at {MODEL_PATH} — run train.py first")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model      = CtcRecognitionModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def run_inference():
    download_models()

    model = load_model()
    print("Model loaded. Starting webcam...")

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    )

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # rolling window of recent frames
        lhand_buffer = collections.deque(maxlen=WINDOW_SIZE)
        rhand_buffer = collections.deque(maxlen=WINDOW_SIZE)
        pose_buffer  = collections.deque(maxlen=WINDOW_SIZE)

        prediction   = []   # current prediction shown on screen
        last_infer   = time.time()
        INFER_EVERY  = 0.5  # run inference every 0.5 seconds

        print("Signing... Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = hand_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            frame = draw_landmarks(frame, hand_result, pose_result)

            try:
                flat = collect_handle(hand_result, pose_result)
                lhand = flat[:LHAND_FEATURES]
                rhand = flat[LHAND_FEATURES:LHAND_FEATURES + RHAND_FEATURES]
                pose  = flat[LHAND_FEATURES + RHAND_FEATURES:]
                lhand_buffer.append(lhand)
                rhand_buffer.append(rhand)
                pose_buffer.append(pose)
            except ValueError:
                pass  # no detection this frame

            # run inference every INFER_EVERY seconds if we have enough frames
            now = time.time()
            if now - last_infer >= INFER_EVERY and len(lhand_buffer) >= 10:
                with torch.no_grad():
                    lhand_t = torch.tensor(list(lhand_buffer), dtype=torch.float32).unsqueeze(0)
                    rhand_t = torch.tensor(list(rhand_buffer), dtype=torch.float32).unsqueeze(0)
                    pose_t  = torch.tensor(list(pose_buffer),  dtype=torch.float32).unsqueeze(0)

                    logits     = model(lhand_t, rhand_t, pose_t)
                    prediction = decode(logits)

                last_infer = now

            # display prediction on screen
            pred_text = " ".join(prediction) if prediction else "..."
            cv2.putText(frame, pred_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Buffer: {len(lhand_buffer)}/{WINDOW_SIZE} frames", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Press Q to quit", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Sign Language Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()

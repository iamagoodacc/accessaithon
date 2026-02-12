from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import cv2
import numpy as np
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from utils import render_handle
from data import collect_handle
from video import run
import os

SIGNS = ["hello", "yes"]
FRAMES_PER_SEQUENCE = 30
NUM_SEQUENCES = 20

sequences = []
current_sequence = []
collecting = False
sequence_count = 0
sign_index = 0        # which sign we're currently collecting

def main_handle(
        hand_landmarks: HandLandmarkerResult,
        pose_landmarks: PoseLandmarkerResult,
        img: cv2.typing.MatLike
):
    global collecting, current_sequence, sequences, sequence_count, sign_index

    # all signs collected
    if sign_index >= len(SIGNS):
        cv2.putText(img, "ALL SIGNS COLLECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        render_handle(hand_landmarks, pose_landmarks, img)
        cv2.waitKey(1)
        return

    current_sign = SIGNS[sign_index]

    if collecting:
        try:
            frame_data = collect_handle(hand_landmarks, pose_landmarks)
            current_sequence.append(frame_data)

            cv2.putText(img, f"RECORDING {len(current_sequence)}/{FRAMES_PER_SEQUENCE}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(current_sequence) == FRAMES_PER_SEQUENCE:
                sequences.append(current_sequence.copy())
                sequence_count += 1
                current_sequence = []
                collecting = False
                print(f"[{current_sign}] Sequence {sequence_count}/{NUM_SEQUENCES} collected")

                # done collecting this sign
                if sequence_count == NUM_SEQUENCES:
                    data = np.array(sequences)
                    os.makedirs("data", exist_ok=True)
                    np.save(f"data/{current_sign}.npy", data)
                    print(f"Saved data/{current_sign}.npy with shape {data.shape}")

                    # move to next sign
                    sequences = []
                    sequence_count = 0
                    sign_index += 1

                    if sign_index < len(SIGNS):
                        print(f"Now collecting: {SIGNS[sign_index]}")

        except Exception as e:
            print(f"Error collecting frame: {e}")
            current_sequence = []
            collecting = False

    # status display
    cv2.putText(img, f"Sign: {current_sign}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Collected: {sequence_count}/{NUM_SEQUENCES}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "SPACE to record | Q to quit", (10, img.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    render_handle(hand_landmarks, pose_landmarks, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and not collecting and sequence_count < NUM_SEQUENCES:
        collecting = True
        current_sequence = []
        print(f"Recording [{current_sign}] sequence {sequence_count + 1}...")
    elif key == ord('q'):
        # save whatever we have so far
        if sequences:
            data = np.array(sequences)
            os.makedirs("data", exist_ok=True)
            np.save(f"data/{current_sign}.npy", data)
            print(f"Saved {sequence_count} sequences to data/{current_sign}.npy")
        exit()

if __name__ == "__main__":
    run(main_handle)

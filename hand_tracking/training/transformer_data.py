"""
Records sign language data using mediapipe and saves into FrameData .pt files.
Run this script once per sentence you want to record.
"""
import os
import time
import urllib.request
import torch
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from core.data import collect_handle, NUM_LANDMARKS_IN_HAND, POSE_LANDMARKS_IDX_LIST, DIMENSIONS
import sys
def draw_landmarks(frame, hand_result, pose_result):
    """Draw hand and pose landmarks onto the frame using OpenCV directly."""
    h, w = frame.shape[:2]

    # hand connections from the new tasks API
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),         # thumb
        (0,5),(5,6),(6,7),(7,8),         # index
        (0,9),(9,10),(10,11),(11,12),    # middle
        (0,13),(13,14),(14,15),(15,16),  # ring
        (0,17),(17,18),(18,19),(19,20),  # pinky
        (5,9),(9,13),(13,17),            # palm
    ]

    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),         # face left
        (0,4),(4,5),(5,6),(6,8),         # face right
        (9,10),                          # mouth
        (11,12),                         # shoulders
        (11,13),(13,15),                 # left arm
        (12,14),(14,16),                 # right arm
    ]

    # draw pose
    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            # draw connections
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                    continue
                start = pose_landmarks[start_idx]
                end   = pose_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w),   int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # draw joints
            for lm in pose_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    # draw hands
    if hand_result.hand_landmarks:
        for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            # different color per hand
            label = handedness[0].category_name  # "Left" or "Right"
            color = (0, 255, 0) if label == "Left" else (255, 0, 0)

            # draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end   = hand_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w),   int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            # draw joints
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)

            # label which hand
            wrist = hand_landmarks[0]
            cv2.putText(frame, label, (int(wrist.x * w), int(wrist.y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ── vocabulary ────────────────────────────────────────────────────────────────
# index 0 is reserved for CTC blank — real words start at 1
VOCABULARY = {
    1: "Hello",
    2: "I",
    3: "You",
    4: "Want",
    5: "Apple",
}
VOCABULARY_INVERSE = {v: k for k, v in VOCABULARY.items()}

# ── feature sizes (derived from collect_handle layout) ────────────────────────
LHAND_FEATURES = NUM_LANDMARKS_IN_HAND * DIMENSIONS   # 21 * 3 = 63
RHAND_FEATURES = NUM_LANDMARKS_IN_HAND * DIMENSIONS   # 21 * 3 = 63
POSE_FEATURES  = len(POSE_LANDMARKS_IDX_LIST) * DIMENSIONS  # 17 * 3 = 51

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── mediapipe model urls ──────────────────────────────────────────────────────
MEDIAPIPE_MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def download_models():
    """Download mediapipe model files if they are not already present."""
    for filename, url in MEDIAPIPE_MODELS.items():
        if os.path.exists(filename):
            continue
        print(f"Downloading {filename}...")
        try:
            # show a simple progress indicator
            def reporthook(block, block_size, total):
                downloaded = block * block_size
                if total > 0:
                    pct = min(downloaded / total * 100, 100)
                    print(f"\r  {pct:.1f}%", end="", flush=True)
            urllib.request.urlretrieve(url, filename, reporthook)
            print(f"\r  Done → {filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {filename}: {e}")


def split_flat_features(flat: list[float]) -> tuple[list[float], list[float], list[float]]:
    """
    collect_handle returns a flat list: [lhand(63), rhand(63), pose(51)]
    Split it back into the three parts.
    """
    lhand = flat[:LHAND_FEATURES]
    rhand = flat[LHAND_FEATURES:LHAND_FEATURES + RHAND_FEATURES]
    pose  = flat[LHAND_FEATURES + RHAND_FEATURES:]
    return lhand, rhand, pose


def get_sentence_labels() -> tuple[list[int], str]:
    """
    Ask the user to type the sentence they are about to sign.
    Returns the label indices and a human-readable sentence string.
    """
    print("\nAvailable vocabulary:")
    for idx, word in VOCABULARY.items():
        print(f"  {idx}: {word}")

    print("\nType the words you will sign, separated by spaces.")
    print("Example: Hello Thank you Yes")
    raw = input("> ").strip()

    words = raw.split()
    labels = []
    for word in words:
        # allow typing index or word name
        if word.isdigit() and int(word) in VOCABULARY:
            labels.append(int(word))
        elif word in VOCABULARY_INVERSE:
            labels.append(VOCABULARY_INVERSE[word])
        else:
            raise ValueError(f"Unknown word: '{word}'. Must be one of {list(VOCABULARY.values())}")

    sentence = " ".join(VOCABULARY[l] for l in labels)
    return labels, sentence


def record_sentence(
    hand_landmarker: HandLandmarker,
    pose_landmarker: PoseLandmarker,
    sentence: str,
    countdown: int = 3,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Opens webcam, counts down, then records frames until the user presses Q.
    Returns three lists of per-frame features: lhand, rhand, pose.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    lhand_frames = []
    rhand_frames = []
    pose_frames  = []

    print(f"\nGet ready to sign: '{sentence}'")
    print(f"Starting in {countdown} seconds... Press Q to stop recording.")

    # countdown
    start = time.time()
    while time.time() - start < countdown:
        ret, frame = cap.read()
        if not ret:
            continue
        remaining = countdown - int(time.time() - start)
        cv2.putText(frame, f"Starting in {remaining}...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Sign: {sentence}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Recording", frame)
        cv2.waitKey(1)

    print("Recording! Press Q to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert to mediapipe image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # run landmarkers
        hand_result = hand_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)

        frame = draw_landmarks(frame, hand_result, pose_result)

        try:
            flat = collect_handle(hand_result, pose_result)
            lhand, rhand, pose = split_flat_features(flat)
            lhand_frames.append(lhand)
            rhand_frames.append(rhand)
            pose_frames.append(pose)
            status = f"Frames: {len(lhand_frames)}"
        except ValueError:
            # no hands or pose detected — skip this frame
            status = f"Frames: {len(lhand_frames)} (no detection)"

        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Signing: {sentence}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to stop", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return lhand_frames, rhand_frames, pose_frames


def save_sample(
    lhand_frames: list[list[float]],
    rhand_frames: list[list[float]],
    pose_frames:  list[list[float]],
    labels: list[int],
    sentence: str,
):
    """
    Converts recorded frames into tensors and saves as a .pt file
    compatible with SignLangDataset.
    """
    nframes = len(lhand_frames)
    if nframes == 0:
        raise ValueError("No frames recorded — nothing to save")

    # shape: (nframes, features) — matches what SignLangDataset expects
    lhand_tensor  = torch.tensor(lhand_frames, dtype=torch.float32)  # (nframes, 63)
    rhand_tensor  = torch.tensor(rhand_frames, dtype=torch.float32)  # (nframes, 63)
    pose_tensor   = torch.tensor(pose_frames,  dtype=torch.float32)  # (nframes, 51)
    labels_tensor = torch.tensor(labels, dtype=torch.long)           # (num_signs,)

    assert lhand_tensor.shape == (nframes, LHAND_FEATURES)
    assert rhand_tensor.shape == (nframes, RHAND_FEATURES)
    assert pose_tensor.shape  == (nframes, POSE_FEATURES)

    # unique filename using timestamp so repeated recordings don't overwrite each other
    timestamp = int(time.time())
    safe_sentence = sentence.replace(" ", "_").replace(",", "")
    filename = f"{safe_sentence}_{timestamp}.pt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # save as FrameData — must match TypedDict exactly
    torch.save({
        'nframes':        nframes,
        'lhand_features': lhand_tensor,
        'rhand_features': rhand_tensor,
        'pose_features':  pose_tensor,
        'labels':         labels_tensor,
    }, filepath)

    print(f"Saved {nframes} frames → {filepath}")
    return filepath


def main():
    # download model files if not present
    download_models()

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    )

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

        while True:
            try:
                labels, sentence = get_sentence_labels()
            except ValueError as e:
                print(f"Error: {e}")
                continue

            print(f"\nYou will sign: '{sentence}'  →  labels={labels}")
            confirm = input("Press Enter to start, or type 'skip' to re-enter: ").strip()
            if confirm.lower() == 'skip':
                continue

            lhand_frames, rhand_frames, pose_frames = record_sentence(
                hand_landmarker, pose_landmarker, sentence
            )

            if len(lhand_frames) < 10:
                print(f"Only {len(lhand_frames)} frames recorded — too short, discarding.")
                continue

            save_sample(lhand_frames, rhand_frames, pose_frames, labels, sentence)

            again = input("\nRecord another? (y/n): ").strip().lower()
            if again != 'y':
                break

    print("Done recording.")


if __name__ == "__main__":
    main()

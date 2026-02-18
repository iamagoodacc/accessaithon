"""
Automated Training Pipeline
============================
Processes pre-recorded video files to extract landmarks and train the gesture recognition model.

Expected folder structure:
    videos/
        hello/
            video1.mp4
            video2.mp4
            ...
        yes/
            clip_a.avi
            clip_b.mp4
            ...
        thanks/
            recording1.mp4
            ...

Each subfolder name becomes a sign label.
Each video should contain one performance of that gesture.

Usage:
    python auto_train.py                          # uses ./videos/ folder
    python auto_train.py --video_dir ./my_videos  # custom folder
    python auto_train.py --frames 30 --epochs 100 # custom settings
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
import mediapipe as mp
import urllib.request

# ──────────────────────────────────────────────
#  Resolve paths — hand_tracking/ contains the core modules
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HAND_TRACKING_DIR = os.path.join(PROJECT_ROOT, "hand_tracking")

# Add hand_tracking/ to sys.path so we can import core.*
sys.path.insert(0, HAND_TRACKING_DIR)

from core.data import collect_handle
from core.model import RecognitionModel, train


# ──────────────────────────────────────────────
#  MediaPipe setup (same models as video.py)
# ──────────────────────────────────────────────
TASKS_DIR = "tasks"

MODELS = {
    "hand": (
        os.path.join(TASKS_DIR, "hand_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose": (
        os.path.join(TASKS_DIR, "pose_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def download_models():
    os.makedirs(TASKS_DIR, exist_ok=True)
    for name, (path, url) in MODELS.items():
        if not os.path.exists(path):
            print(f"  Downloading {name} model...")
            urllib.request.urlretrieve(url, path)
            print(f"  {name} model downloaded!")


def create_detectors():
    """Create and return MediaPipe hand + pose detectors."""
    hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=MODELS["hand"][0],
                delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
    )

    pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=MODELS["pose"][0],
                delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
    )
    return hand_detector, pose_detector


# ──────────────────────────────────────────────
#  Video processing
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_landmarks_from_video(video_path, hand_detector, pose_detector, target_frames):
    """
    Read a video file, run MediaPipe on each frame, and return landmark sequences.

    If the video has more frames than target_frames, we sample evenly.
    If fewer, we pad by repeating the last frame.

    Returns a list of sequences, where each sequence is target_frames long.
    For short videos (<target_frames), returns one padded sequence.
    For long videos, returns one evenly-sampled sequence.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    WARNING: Could not open {video_path}, skipping.")
        return []

    # Read all frames and extract landmarks
    all_frame_data = []
    frame_idx = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)

        try:
            frame_data = collect_handle(hand_result, pose_result)
            all_frame_data.append(frame_data)
        except ValueError:
            # No pose detected in this frame — skip it
            pass

        frame_idx += 1

    cap.release()

    if len(all_frame_data) == 0:
        print(f"    WARNING: No valid frames extracted from {video_path}")
        return []

    # Resample to exactly target_frames
    num_raw = len(all_frame_data)

    if num_raw >= target_frames:
        # Evenly sample target_frames from the video
        indices = np.linspace(0, num_raw - 1, target_frames, dtype=int)
        sequence = [all_frame_data[i] for i in indices]
    else:
        # Pad by repeating last frame
        sequence = all_frame_data + [all_frame_data[-1]] * (target_frames - num_raw)

    return [sequence]


def process_video_folder(video_dir, target_frames):
    """
    Scan the video directory and extract landmarks from all videos.

    Returns:
        signs: list of sign names (subfolder names)
        all_data: dict mapping sign_name -> list of sequences
    """
    signs = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ])

    if not signs:
        print(f"ERROR: No subfolders found in {video_dir}/")
        print(f"Expected structure:")
        print(f"  {video_dir}/")
        print(f"    hello/")
        print(f"      video1.mp4")
        print(f"    yes/")
        print(f"      video1.mp4")
        sys.exit(1)

    print(f"\nFound {len(signs)} sign(s): {signs}")

    # Set up MediaPipe
    print("\nLoading MediaPipe models...")
    download_models()
    hand_detector, pose_detector = create_detectors()

    all_data = {}

    for sign in signs:
        sign_dir = os.path.join(video_dir, sign)
        video_files = sorted([
            f for f in os.listdir(sign_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])

        if not video_files:
            print(f"\n  WARNING: No video files found in {sign_dir}/, skipping.")
            continue

        print(f"\n  Processing '{sign}' — {len(video_files)} video(s)")
        sequences = []

        for i, vf in enumerate(video_files):
            video_path = os.path.join(sign_dir, vf)
            print(f"    [{i+1}/{len(video_files)}] {vf}...", end=" ")

            seqs = extract_landmarks_from_video(video_path, hand_detector, pose_detector, target_frames)
            sequences.extend(seqs)

            if seqs:
                print(f"OK ({len(seqs)} sequence(s))")
            else:
                print("SKIPPED (no valid data)")

        if sequences:
            all_data[sign] = sequences
            print(f"  Total for '{sign}': {len(sequences)} sequences of {target_frames} frames")
        else:
            print(f"  WARNING: No valid data for '{sign}'")

    # Cleanup
    hand_detector.close()
    pose_detector.close()

    return signs, all_data


# ──────────────────────────────────────────────
#  Save extracted data
# ──────────────────────────────────────────────
def save_data(all_data, data_dir="data"):
    """Save extracted sequences as .npy files (same format as collect_data.py)."""
    os.makedirs(data_dir, exist_ok=True)

    for sign, sequences in all_data.items():
        arr = np.array(sequences)  # shape: (num_sequences, target_frames, num_features)
        filepath = os.path.join(data_dir, f"{sign}.npy")
        np.save(filepath, arr)
        print(f"  Saved {filepath}  —  shape: {arr.shape}")


# ──────────────────────────────────────────────
#  Train model
# ──────────────────────────────────────────────
def augment_sequence(seq, num_augmented=10):
    """
    Generate augmented copies of a landmark sequence to improve generalization.
    seq: shape (frames, features) — one sample
    Returns: list of augmented sequences
    """
    augmented = []
    frames, features = seq.shape

    for _ in range(num_augmented):
        aug = seq.copy()

        # 1. Gaussian noise — simulates slight hand position differences
        noise_scale = np.random.uniform(0.005, 0.02)
        aug += np.random.normal(0, noise_scale, aug.shape)

        # 2. Temporal jitter — randomly drop & duplicate frames to simulate speed variation
        if frames > 5 and np.random.random() < 0.5:
            n_drop = np.random.randint(1, max(2, frames // 6))
            drop_indices = np.random.choice(frames, n_drop, replace=False)
            keep = np.delete(np.arange(frames), drop_indices)
            # Resample back to original frame count
            resample_indices = np.linspace(0, len(keep) - 1, frames, dtype=int)
            aug = aug[keep][resample_indices]

        # 3. Spatial scaling — simulates different hand sizes / distances from camera
        scale = np.random.uniform(0.85, 1.15)
        aug *= scale

        # 4. Small random translation (shift all landmarks)
        shift = np.random.normal(0, 0.01, features)
        aug += shift

        augmented.append(aug)

    return augmented


def train_model(signs, data_dir="data", epochs=100, lr=0.001, hidden_size=128, num_layers=2, dropout=0.2):
    """Load .npy data and train the LSTM model."""
    x_data = []
    y_data = []

    for label_idx, sign in enumerate(signs):
        filepath = os.path.join(data_dir, f"{sign}.npy")
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping '{sign}'")
            continue

        data = np.load(filepath)
        original_count = len(data)

        # Augment each sample to create more training variety
        augmented_samples = list(data)  # keep originals
        for sample in data:
            augmented_samples.extend(augment_sequence(sample, num_augmented=10))

        data = np.array(augmented_samples)
        x_data.append(data)
        y_data.extend([label_idx] * len(data))
        print(f"  Loaded '{sign}': {original_count} original -> {len(data)} samples (with augmentation)")

    if not x_data:
        print("ERROR: No training data to load!")
        sys.exit(1)

    x_data = np.concatenate(x_data, axis=0)
    y_data = np.array(y_data)

    # Shuffle the data
    indices = np.random.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]

    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.long)

    print(f"\n  Training data shape: {x_train.shape}")
    print(f"  Labels shape:        {y_train.shape}")
    print(f"  Classes:             {len(signs)}")

    input_size = x_train.shape[2]
    output_size = len(signs)

    model = RecognitionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    )

    print(f"\n  Training for {epochs} epochs...")
    train(
        model=model,
        num_epochs=epochs,
        learning_rate=lr,
        x_train=x_train,
        y_train=y_train
    )

    model_path = os.path.join(HAND_TRACKING_DIR, "recognition_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved to {model_path}")

    # Save signs list so api.py and main.py can load it dynamically
    signs_path = os.path.join(HAND_TRACKING_DIR, "signs.json")
    with open(signs_path, "w") as f:
        json.dump(signs, f)
    print(f"  Signs list saved to {signs_path}")

    return model


# ──────────────────────────────────────────────
#  Update SIGNS in project files
# ──────────────────────────────────────────────
def update_signs_in_files(signs):
    """
    Update the SIGNS list in main.py, training.py, api.py, and collect_data.py
    so they stay in sync with the trained model.
    """
    signs_str = repr(signs)
    target_files = [
        os.path.join(HAND_TRACKING_DIR, "main.py"),
        os.path.join(HAND_TRACKING_DIR, "training", "training.py"),
        os.path.join(HAND_TRACKING_DIR, "training", "collect_data.py"),
        os.path.join(HAND_TRACKING_DIR, "api.py"),
    ]

    for filename in target_files:
        if not os.path.exists(filename):
            continue

        with open(filename, "r") as f:
            content = f.read()

        # Find and replace the SIGNS = [...] line
        import re
        new_content, count = re.subn(
            r'^SIGNS\s*=\s*\[.*?\]',
            f'SIGNS = {signs_str}',
            content,
            count=1,
            flags=re.MULTILINE
        )

        if count > 0:
            with open(filename, "w") as f:
                f.write(new_content)
            print(f"  Updated SIGNS in {filename}")
        else:
            print(f"  WARNING: Could not find SIGNS line in {filename}")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Automated BSL gesture training from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example folder structure:
  videos/
    hello/
      vid1.mp4
      vid2.mp4
    yes/
      clip1.mp4
    thanks/
      recording1.mp4

Each subfolder = one sign. Each video = one sample of that gesture.
More videos per sign = better accuracy. Aim for 15-30+ videos per sign.
        """
    )
    default_data_dir = os.path.join(HAND_TRACKING_DIR, "data")
    parser.add_argument("--video_dir", default="videos", help="Path to video folder (default: ./videos)")
    parser.add_argument("--data_dir", default=default_data_dir, help=f"Where to save .npy files (default: {default_data_dir})")
    parser.add_argument("--frames", type=int, default=30, help="Frames per sequence (default: 30)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--skip_extract", action="store_true", help="Skip extraction, use existing .npy files")
    parser.add_argument("--skip_train", action="store_true", help="Only extract data, skip training")
    parser.add_argument("--update_signs", action="store_true", help="Update SIGNS list in project files")

    args = parser.parse_args()

    print("=" * 60)
    print("  Automated BSL Gesture Training Pipeline")
    print("=" * 60)

    # Step 1: Extract landmarks from videos
    if not args.skip_extract:
        print(f"\n[Step 1/3] Extracting landmarks from videos in '{args.video_dir}/'")
        signs, all_data = process_video_folder(args.video_dir, args.frames)

        if not all_data:
            print("ERROR: No data extracted from any videos!")
            sys.exit(1)

        # Only keep signs that have data
        signs = [s for s in signs if s in all_data]

        print(f"\n[Step 2/3] Saving extracted data to '{args.data_dir}/'")
        save_data(all_data, args.data_dir)
    else:
        print("\n[Step 1/3] Skipping extraction (using existing .npy files)")
        print("[Step 2/3] Skipping save")
        # Discover signs from existing .npy files
        if not os.path.exists(args.data_dir):
            print(f"ERROR: Data directory '{args.data_dir}' not found!")
            sys.exit(1)
        signs = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(args.data_dir)
            if f.endswith(".npy")
        ])
        if not signs:
            print("ERROR: No .npy files found!")
            sys.exit(1)
        print(f"  Found existing data for: {signs}")

    # Step 3: Train
    if not args.skip_train:
        print(f"\n[Step 3/3] Training model")
        train_model(
            signs=signs,
            data_dir=args.data_dir,
            epochs=args.epochs,
            lr=args.lr,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        print("\n[Step 3/3] Skipping training")

    # Optional: update SIGNS in project files
    if args.update_signs:
        print(f"\n[Bonus] Updating SIGNS list in project files")
        update_signs_in_files(signs)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
    print(f"\nSigns trained: {signs}")
    if not args.skip_train:
        print(f"Model saved to: {os.path.join(HAND_TRACKING_DIR, 'recognition_model.pth')}")
        print(f"Signs list saved to: {os.path.join(HAND_TRACKING_DIR, 'signs.json')}")
    print(f"Data saved to: {args.data_dir}")
    print(f"\nTo run live detection:  cd {HAND_TRACKING_DIR} && python main.py")


if __name__ == "__main__":
    main()

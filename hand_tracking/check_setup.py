"""
BSL to Text Converter - Setup Verification Script
Run this to check if your system is ready for real-time detection
"""

import os
import sys

def check_file(filepath, name, required=True):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else ("[MISSING]" if required else "[WARN]")
    print(f"{status} {name}: {filepath}")
    return exists

def check_module(module_name):
    """Check if a Python module is installed"""
    try:
        __import__(module_name)
        print(f"[OK] {module_name} installed")
        return True
    except ImportError:
        print(f"[MISSING] {module_name} NOT installed")
        return False

def main():
    print("=" * 60)
    print("BSL to Text Converter - Setup Verification")
    print("=" * 60)

    all_good = True

    # Check required Python files
    print("\n[*] Checking Python files...")
    required_files = [
        ("main.py", "Main application"),
        ("model.py", "Model definition"),
        ("data.py", "Data processing"),
        ("video.py", "Video capture"),
        ("training.py", "Training script"),
        ("collect_data.py", "Data collection"),
    ]

    for file, name in required_files:
        if not check_file(file, name):
            all_good = False

    # Check MediaPipe models
    print("\n[*] Checking MediaPipe models...")
    models = [
        ("hand_landmarker.task", "Hand detector model"),
        ("pose_landmarker.task", "Pose detector model"),
    ]

    for file, name in models:
        check_file(file, name, required=False)  # Will auto-download if missing

    # Check required Python packages
    print("\n[*] Checking Python packages...")
    packages = [
        "torch",
        "cv2",
        "numpy",
        "mediapipe",
    ]

    for package in packages:
        module_name = "cv2" if package == "cv2" else package
        if not check_module(module_name):
            all_good = False

    # Check training data
    print("\n[*] Checking training data...")
    data_dir = "data"
    has_data = False

    if os.path.exists(data_dir):
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if npy_files:
            has_data = True
            print(f"[OK] Found {len(npy_files)} data file(s):")
            for f in npy_files:
                filepath = os.path.join(data_dir, f)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"   - {f} ({size_mb:.2f} MB)")
        else:
            print(f"[WARN] No .npy files in {data_dir}/ folder")
    else:
        print(f"[WARN] Directory '{data_dir}' not found")

    # Check trained model
    print("\n[*] Checking trained model...")
    model_file = "recognition_model.pth"
    has_model = check_file(model_file, "Trained model")

    if has_model:
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")

    # Summary and next steps
    print("\n" + "=" * 60)
    print("SETUP STATUS")
    print("=" * 60)

    if not all_good:
        print("[ERROR] Missing required packages!")
        print("\n>> Install missing packages:")
        print("   pip install -r requirements.txt")
        return False

    if not has_data:
        print("[WARN] No training data found!")
        print("\n>> Next step: Collect training data")
        print("   python collect_data.py")
        print("\n   Instructions:")
        print("   1. Press SPACE to record each gesture")
        print("   2. Perform gesture consistently")
        print("   3. Record 20 sequences per gesture")
        return False

    if not has_model:
        print("[WARN] No trained model found!")
        print("\n>> Next step: Train the model")
        print("   python training.py")
        print("\n   This will:")
        print("   - Load training data from data/ folder")
        print("   - Train LSTM model (100 epochs)")
        print("   - Save model as recognition_model.pth")
        return False

    print("[SUCCESS] ALL CHECKS PASSED!")
    print("\n>> Ready to run real-time detection:")
    print("   python main.py")
    print("\n>> Controls:")
    print("   - C: Clear text")
    print("   - R: Reset buffer")
    print("   - Q: Quit")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Error during verification: {e}")
        sys.exit(1)

from collections.abc import Callable
import cv2
import mediapipe as mp
import os
import urllib.request
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

MODELS = {
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    # "face": (
    #     "face_landmarker.task",
    #     "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    # ),
    "pose": (
        "pose_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def download_models():
    for name, (path, url) in MODELS.items():
        if not os.path.exists(path):
            print(f"Downloading {name} model...")
            urllib.request.urlretrieve(url, path)
            print(f"{name} model downloaded!")


def run(handle: Callable[
    [
        HandLandmarkerResult,  # hand_landmarks
        PoseLandmarkerResult,  # pose_landmarks
        cv2.typing.MatLike
    ],
    None
]):
    download_models()

    cap = cv2.VideoCapture(0)

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

    while True:
        success, img = cap.read()

        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            hand_result = hand_detector.detect(mp_image)
            pose_result = pose_detector.detect(mp_image)

            handle(
                hand_result,
                pose_result,
                img
            )

            cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

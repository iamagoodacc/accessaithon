from collections.abc import Callable

import cv2
import mediapipe as mp
import os
import urllib.request

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


def run(handle: Callable[[list[list[NormalizedLandmark]], list[list[NormalizedLandmark]], cv2.typing.MatLike], None]):
    hand_model_path = 'hand_landmarker.task'
    if not os.path.exists(hand_model_path):
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, hand_model_path)
        print("Model downloaded successfully!")

    face_model_path = 'face_detector.model'
    if not os.path.exists(face_model_path):
        print("Downloading face recognition model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, "face_landmarker.task")
        print("Model downloaded successfully!")

    cap = cv2.VideoCapture(0)

    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task', delegate=mp.tasks.BaseOptions.Delegate.CPU)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    face_base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
    face_options = mp.tasks.vision.FaceLandmarkerOptions(base_options=face_base_options)
    face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)

    while True:
        success, img = cap.read()

        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            hand_detection_result = hand_detector.detect(mp_image)
            face_detection_result = face_detector.detect(mp_image)

            if hand_detection_result.hand_landmarks and face_detection_result.face_landmarks:
                handle(hand_detection_result.hand_landmarks, face_detection_result.face_landmarks, img)

            cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
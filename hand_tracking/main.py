import cv2
import mediapipe as mp
import os
import urllib.request

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded successfully!")

cap = cv2.VideoCapture(0)

base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task', delegate=mp.tasks.BaseOptions.Delegate.CPU)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

while True:
    success, img = cap.read()

    if success:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    cv2.putText(img, str(idx), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=2)

        cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
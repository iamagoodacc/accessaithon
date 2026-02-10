from typing import Iterable, TypeVar

import cv2
import torch
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

DIMENSIONS = 3
NUM_LANDMARKS_IN_HAND = 21

POSE_LANDMARKS_HEAD_START_IDX = 0
POSE_LANDMARKS_HEAD_END_IDX = 10

# we use the head, shoulders, and arm joints
POSE_LANDMARKS_SHOULDER_IDX = [11, 13]
POSE_LANDMARKS_ARMS_IDX = [13, 14, 15, 16]
POSE_LANDMARKS_IDX_LIST = list(range(POSE_LANDMARKS_HEAD_START_IDX, POSE_LANDMARKS_HEAD_END_IDX + 1)) + POSE_LANDMARKS_SHOULDER_IDX + POSE_LANDMARKS_ARMS_IDX

T = TypeVar('T')
def flatten(xss: Iterable[Iterable[T]]) -> list[T]:
    return [x
     for xs in xss
     for x in xs
     ]

def collect_handle(
        hand_landmarks: HandLandmarkerResult,
        pose_landmarks: PoseLandmarkerResult,
        img: cv2.typing.MatLike
):
    # layout: left hand_landmarks + right hand_landmarks + pose landmarks
    landmarks = []

    handedness: list[list[Category]] = hand_landmarks.handedness
    left = None
    right = None

    for landmarks, hand in zip(hand_landmarks.hand_landmarks, handedness):
        label = hand[0].category_name  # "Left" or "Right"
        if label == "Left":
            left = landmarks
        else:
            right = landmarks

    lefthand_landmarks = []
    if left is None:
        lefthand_landmarks = [0] * NUM_LANDMARKS_IN_HAND * DIMENSIONS
    else:
        lefthand_landmarks = flatten(map(lambda landmark: [landmark.x, landmark.y, landmark.z], left))

    assert len(lefthand_landmarks) == NUM_LANDMARKS_IN_HAND * DIMENSIONS

    righthand_landmarks = []
    if right is None:
        righthand_landmarks = [0] * NUM_LANDMARKS_IN_HAND * DIMENSIONS
    else:
        righthand_landmarks = flatten(map(lambda landmark: [landmark.x, landmark.y, landmark.z], left))

    assert len(righthand_landmarks) == NUM_LANDMARKS_IN_HAND * DIMENSIONS

    used_pose_landmarks = flatten(map(lambda landmark: [landmark.x, landmark.y, landmark.z], map(lambda idx: pose_landmarks.pose_landmarks[idx], POSE_LANDMARKS_IDX_LIST)))

    assert len(used_pose_landmarks) == len(POSE_LANDMARKS_IDX_LIST) * DIMENSIONS

    landmarks = lefthand_landmarks + righthand_landmarks + used_pose_landmarks

    return
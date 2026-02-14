"""
Extracts and normalizes landmark features (left hand, right hand, head/shoulders/arms from pose — all relative to a base position)
"""

from typing import Iterable, TypeVar

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
) -> list[float]:
    """ layout: left hand_landmarks + right hand_landmarks + pose landmarks """

    # Check if pose landmarks are available
    if not pose_landmarks.pose_landmarks or len(pose_landmarks.pose_landmarks) == 0:
        raise ValueError("No pose landmarks detected")

    used_pose_landmarks = list(map(lambda landmark: [landmark.x, landmark.y, landmark.z], map(lambda idx: pose_landmarks.pose_landmarks[0][idx], POSE_LANDMARKS_IDX_LIST)))
    # all relative to this location
    base_position = used_pose_landmarks[0]

    assert len(used_pose_landmarks) == len(POSE_LANDMARKS_IDX_LIST)

    handedness: list[list[Category]] = hand_landmarks.handedness
    left = None
    right = None

    for landmarks, hand in zip(hand_landmarks.hand_landmarks, handedness):
        label = hand[0].category_name  # "Left" or "Right"
        if label == "Left":
            left = landmarks
        else:
            right = landmarks

    if left is None:
        # Fix: Create separate zero lists for each landmark (not reference to same list)
        left_hand_landmarks = [[0, 0, 0] for _ in range(NUM_LANDMARKS_IN_HAND)]
    else:
        left_hand_landmarks = list(map(lambda landmark: [landmark.x, landmark.y, landmark.z], left))

    assert len(left_hand_landmarks) == NUM_LANDMARKS_IN_HAND

    if right is None:
        # Fix: Create separate zero lists for each landmark (not reference to same list)
        right_hand_landmarks = [[0, 0, 0] for _ in range(NUM_LANDMARKS_IN_HAND)]
    else:
        right_hand_landmarks = list(map(lambda landmark: [landmark.x, landmark.y, landmark.z], right))

    assert len(right_hand_landmarks) == NUM_LANDMARKS_IN_HAND

    landmarks = left_hand_landmarks + right_hand_landmarks + used_pose_landmarks

    return flatten(map(lambda landmark: map(lambda pair: pair[0] - pair[1], zip(landmark,base_position)), landmarks))
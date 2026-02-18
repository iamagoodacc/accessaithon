/**
 * Port of hand_tracking/core/data.py
 * Extracts and normalizes landmark features from MediaPipe results
 */

import type {
  HandLandmarkerResult,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

const DIMENSIONS = 3;
const NUM_LANDMARKS_IN_HAND = 21;

const POSE_LANDMARKS_HEAD_START_IDX = 0;
const POSE_LANDMARKS_HEAD_END_IDX = 10;

const POSE_LANDMARKS_SHOULDER_IDX = [11, 13];
const POSE_LANDMARKS_ARMS_IDX = [13, 14, 15, 16];

const POSE_LANDMARKS_IDX_LIST = [
  ...Array.from(
    { length: POSE_LANDMARKS_HEAD_END_IDX - POSE_LANDMARKS_HEAD_START_IDX + 1 },
    (_, i) => i + POSE_LANDMARKS_HEAD_START_IDX
  ),
  ...POSE_LANDMARKS_SHOULDER_IDX,
  ...POSE_LANDMARKS_ARMS_IDX,
];

export const FEATURE_LENGTH =
  NUM_LANDMARKS_IN_HAND * DIMENSIONS * 2 + POSE_LANDMARKS_IDX_LIST.length * DIMENSIONS;

/**
 * Extract normalized landmark features from hand and pose results.
 * Returns a flat array of floats matching the Python backend's format.
 * Returns null if pose landmarks are not available.
 */
export function extractLandmarks(
  handResult: HandLandmarkerResult,
  poseResult: PoseLandmarkerResult
): number[] | null {
  // Need pose landmarks
  if (!poseResult.landmarks || poseResult.landmarks.length === 0) {
    return null;
  }

  // Need at least one hand detected — without hands the model can't distinguish signs
  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    return null;
  }

  const poseLandmarks = poseResult.landmarks[0];

  // Extract used pose landmarks as [x, y, z]
  const usedPoseLandmarks = POSE_LANDMARKS_IDX_LIST.map((idx) => {
    const lm = poseLandmarks[idx];
    return [lm.x, lm.y, lm.z];
  });

  // Base position: first used pose landmark (top of head)
  const basePosition = usedPoseLandmarks[0];

  // Determine left/right hands
  let leftHandLandmarks: number[][] | null = null;
  let rightHandLandmarks: number[][] | null = null;

  if (handResult.handednesses && handResult.landmarks) {
    for (let i = 0; i < handResult.handednesses.length; i++) {
      const label = handResult.handednesses[i][0].categoryName;
      const landmarks = handResult.landmarks[i].map((lm) => [lm.x, lm.y, lm.z]);

      if (label === "Left") {
        leftHandLandmarks = landmarks;
      } else {
        rightHandLandmarks = landmarks;
      }
    }
  }

  // Zero-fill missing hands
  if (!leftHandLandmarks) {
    leftHandLandmarks = Array.from({ length: NUM_LANDMARKS_IN_HAND }, () => [0, 0, 0]);
  }
  if (!rightHandLandmarks) {
    rightHandLandmarks = Array.from({ length: NUM_LANDMARKS_IN_HAND }, () => [0, 0, 0]);
  }

  // Combine: left hand + right hand + pose
  const allLandmarks = [
    ...leftHandLandmarks,
    ...rightHandLandmarks,
    ...usedPoseLandmarks,
  ];

  // Flatten and subtract base position
  const features: number[] = [];
  for (const landmark of allLandmarks) {
    for (let d = 0; d < DIMENSIONS; d++) {
      features.push(landmark[d] - basePosition[d]);
    }
  }

  return features;
}

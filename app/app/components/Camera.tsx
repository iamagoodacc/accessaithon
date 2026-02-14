"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import type {
  HandLandmarkerResult,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { extractLandmarks, FEATURE_LENGTH } from "../lib/landmarks";

const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

interface CameraProps {
  /** Called each frame with the extracted landmark feature vector */
  onFrame?: (features: number[]) => void;
  /** Whether collection is actively running */
  collecting?: boolean;
}

export default function Camera({ onFrame, collecting = false }: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<string>("Initializing...");
  const [ready, setReady] = useState(false);

  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const animationRef = useRef<number>(0);

  // Initialize MediaPipe models
  useEffect(() => {
    let cancelled = false;

    async function init() {
      setStatus("Loading MediaPipe models...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      if (cancelled) return;

      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
        numHands: 2,
        runningMode: "VIDEO",
      });

      const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
      });

      if (cancelled) {
        handLandmarker.close();
        poseLandmarker.close();
        return;
      }

      handLandmarkerRef.current = handLandmarker;
      poseLandmarkerRef.current = poseLandmarker;

      setStatus("Starting camera...");

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setReady(true);
          setStatus("Ready");
        }
      } catch {
        setStatus("Camera access denied");
      }
    }

    init();

    return () => {
      cancelled = true;
      handLandmarkerRef.current?.close();
      poseLandmarkerRef.current?.close();
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((t) => t.stop());
      }
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  // Detection loop
  const detect = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const handLandmarker = handLandmarkerRef.current;
    const poseLandmarker = poseLandmarkerRef.current;

    if (!video || !canvas || !handLandmarker || !poseLandmarker) {
      animationRef.current = requestAnimationFrame(detect);
      return;
    }

    if (video.readyState < 2) {
      animationRef.current = requestAnimationFrame(detect);
      return;
    }

    const now = performance.now();
    const handResult: HandLandmarkerResult =
      handLandmarker.detectForVideo(video, now);
    const poseResult: PoseLandmarkerResult =
      poseLandmarker.detectForVideo(video, now);

    // Draw overlays
    const ctx = canvas.getContext("2d")!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drawingUtils = new DrawingUtils(ctx);

    // Draw hand landmarks
    if (handResult.landmarks) {
      for (const landmarks of handResult.landmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          HandLandmarker.HAND_CONNECTIONS,
          { color: "#00FF88", lineWidth: 2 }
        );
        drawingUtils.drawLandmarks(landmarks, {
          color: "#FF3366",
          lineWidth: 1,
          radius: 3,
        });
      }
    }

    // Draw pose landmarks (just upper body connections)
    if (poseResult.landmarks) {
      for (const landmarks of poseResult.landmarks) {
        drawingUtils.drawLandmarks(landmarks, {
          color: "#4488FF",
          lineWidth: 1,
          radius: 2,
        });
      }
    }

    // Extract features and send to parent
    if (collecting && onFrame) {
      const features = extractLandmarks(handResult, poseResult);
      if (features) {
        onFrame(features);
      }
    }

    animationRef.current = requestAnimationFrame(detect);
  }, [collecting, onFrame]);

  useEffect(() => {
    if (ready) {
      animationRef.current = requestAnimationFrame(detect);
    }
    return () => cancelAnimationFrame(animationRef.current);
  }, [ready, detect]);

  return (
    <div className="camera-container">
      <div className="camera-wrapper">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ transform: "scaleX(-1)" }}
        />
        <canvas
          ref={canvasRef}
          style={{ transform: "scaleX(-1)" }}
        />
      </div>
      {!ready && (
        <div className="camera-overlay">
          <p>{status}</p>
        </div>
      )}
    </div>
  );
}

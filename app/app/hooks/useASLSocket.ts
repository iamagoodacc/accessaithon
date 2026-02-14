"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = "ws://localhost:5000";

export interface Prediction {
  prediction: string;
  confidence: number;
  textSequence: string[];
}

interface ServerMessage {
  type: "prediction" | "status" | "error" | "stats" | "ack";
  prediction?: string;
  confidence?: number;
  text_sequence?: string[];
  status?: string;
  message?: string;
  frame_count?: number;
  queue_size?: number;
}

export function useASLSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Disconnected");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setConnected(true);
      setStatusMessage("Connected to backend");
    };

    ws.onmessage = (event) => {
      const msg: ServerMessage = JSON.parse(event.data);

      if (msg.type === "prediction") {
        setPrediction({
          prediction: msg.prediction ?? "",
          confidence: msg.confidence ?? 0,
          textSequence: msg.text_sequence ?? [],
        });
      } else if (msg.type === "status") {
        setStatusMessage(msg.message ?? msg.status ?? "");
      } else if (msg.type === "error") {
        setStatusMessage(`Error: ${msg.message}`);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setStatusMessage("Disconnected — reconnecting...");
      wsRef.current = null;
      reconnectTimeout.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      // onclose will fire after this, handling reconnect
    };

    wsRef.current = ws;
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
    setStatusMessage("Disconnected");
  }, []);

  // Send a single frame of landmark features
  const sendFrame = useCallback((features: number[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: "frame", data: features }));
    }
  }, []);

  // Reset the server's frame buffer
  const resetBuffer = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: "reset" }));
    }
  }, []);

  // Clear the accumulated text on the server
  const clearText = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: "clear" }));
    }
    setPrediction(null);
  }, []);

  // Auto-connect on mount, cleanup on unmount
  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return {
    connected,
    statusMessage,
    prediction,
    sendFrame,
    resetBuffer,
    clearText,
  };
}

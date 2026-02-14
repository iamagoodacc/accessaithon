"use client";

import { useState, useCallback } from "react";
import Camera from "./components/Camera";
import { useASLSocket } from "./hooks/useASLSocket";

export default function Home() {
  const [collecting, setCollecting] = useState(true);
  const { connected, statusMessage, prediction, sendFrame, clearText } =
    useASLSocket();

  const translatedText = prediction?.textSequence.join(" ") ?? "";

  const handleFrame = useCallback(
    (features: number[]) => {
      sendFrame(features);
    },
    [sendFrame]
  );

  const handleClear = () => {
    clearText();
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>ASL Translator</h1>
        <p className="subtitle">American Sign Language to Text</p>
      </header>

      <main className="app-main">
        <div className="camera-section">
          <Camera onFrame={handleFrame} collecting={collecting} />
          <div className="camera-controls">
            <button
              onClick={() => setCollecting(!collecting)}
              className={`btn ${collecting ? "btn-active" : "btn-inactive"}`}
            >
              {collecting ? "Pause Detection" : "Resume Detection"}
            </button>
            <span
              className={`connection-status ${connected ? "connected" : "disconnected"}`}
            >
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {prediction && (
          <div className="prediction-banner">
            <span className="prediction-label">Detected:</span>
            <span className="prediction-value">{prediction.prediction}</span>
            <span className="prediction-confidence">
              {(prediction.confidence * 100).toFixed(0)}%
            </span>
          </div>
        )}

        <div className="translation-section">
          <label htmlFor="translation-output">Translation</label>
          <textarea
            id="translation-output"
            className="translation-box"
            value={translatedText}
            readOnly
            placeholder="ASL translations will appear here..."
            rows={6}
          />
          <div className="translation-controls">
            <button onClick={handleClear} className="btn btn-secondary">
              Clear
            </button>
            <span className="hint">{statusMessage}</span>
          </div>
        </div>
      </main>
    </div>
  );
}

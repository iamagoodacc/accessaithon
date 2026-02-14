# Gesture Recognition TCP API (Push-Based Architecture)

This API provides an **asynchronous TCP server** with a **push-based architecture** for real-time gesture recognition. Clients post frame data continuously, and the server automatically pushes prediction results when ready.

## 🎯 How It Works

1. **Client posts frames** → Frames are added to a queue
2. **Queue reaches threshold** → (30 frames + 15 frames = 45 frames total)
3. **Server processes prediction** → Uses first 30 frames for recognition
4. **Server pushes result** → Client receives prediction automatically
5. **Window slides** → Remove 15 oldest frames, ready for next prediction

This design decouples frame posting from prediction, allowing clients to stream frames at full speed while the server processes predictions asynchronously.

## Features

✨ **Push Architecture** - Server sends predictions when ready, not on request
🔄 **Sliding Window** - Processes predictions every 15 frames after initial 30
⚡ **Non-Blocking** - Clients can post frames without waiting for responses
📊 **Real-Time Streaming** - Handle continuous video streams efficiently
🎯 **Automatic Predictions** - No manual triggering required
🔒 **Thread-Safe** - Per-client isolation with async locks
📡 **Bi-directional** - Server can push multiple message types

## Architecture

```
┌──────────────┐
│    Client    │
│              │
│  Post frames │
│  continuously│
└──────┬───────┘
       │ Frames (non-blocking)
       ▼
┌──────────────────────────────┐
│      Server Session          │
│                              │
│  Frame Queue (deque)         │
│  └─ Max: 30 + 15 = 45 frames│
│                              │
│  When size >= 45:            │
│  ├─ Take first 30 frames     │
│  ├─ Run prediction           │
│  ├─ Push result to client    │
│  └─ Remove oldest 15 frames  │
└──────┬───────────────────────┘
       │ Predictions (pushed)
       ▼
┌──────────────┐
│    Client    │
│              │
│   Listens    │
│   for msgs   │
└──────────────┘
```

## Configuration Constants

```python
SEQUENCE_LENGTH = 30      # Frames needed for one prediction
PREDICTION_STRIDE = 15    # Process prediction every N frames
# Queue fills to: 30 + 15 = 45 frames before first prediction
```

These can be adjusted in `api.py` based on your needs:
- **Faster predictions**: Decrease `PREDICTION_STRIDE` (e.g., 10)
- **More context**: Increase `SEQUENCE_LENGTH` (e.g., 40)
- **Trade-off**: Smaller stride = more predictions but more CPU usage

## Setup

1. **Install dependencies**:
   ```bash
   pip install python-dotenv torch numpy
   ```

2. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```

3. **Configure settings** (optional):
   ```
   PORT=5000              # Server port
   HOST=0.0.0.0          # Bind address
   MAX_CONNECTIONS=10     # Max concurrent clients
   WORKER_THREADS=4       # Thread pool size for inference
   ```

4. **Ensure model is trained**:
   ```bash
   python collect_data.py  # Collect training data
   python training.py      # Train the model
   ```

## Running the Server

```bash
python api.py
```

Output:
```
[STARTING] Gesture Recognition API Server (Async Push Model)
[LISTENING] on 0.0.0.0:5000
[MODEL] Loaded with 2 signs: ['hello', 'yes']
[CONFIG] Sequence length: 30, Prediction stride: 15
[READY] Waiting for connections...
```

## Message Types

### From Client → Server

#### 1. Post Frame
```json
{
  "command": "frame",
  "data": [0.1, 0.2, 0.3, ...]  // Array of floats
}
```

#### 2. Reset Buffer
```json
{
  "command": "reset"
}
```

#### 3. Clear Text
```json
{
  "command": "clear"
}
```

#### 4. Get Status
```json
{
  "command": "status"
}
```

#### 5. Get Server Stats
```json
{
  "command": "server_stats"
}
```

### From Server → Client (Pushed)

#### 1. Prediction (automatic)
```json
{
  "type": "prediction",
  "prediction": "hello",
  "confidence": 0.87,
  "frame_count": 45,
  "text_sequence": ["hello", "yes"],
  "queue_size": 30
}
```

#### 2. Status Update
```json
{
  "type": "status",
  "status": "connected",
  "message": "Session started, ready to receive frames",
  "frame_count": 0,
  "queue_size": 0
}
```

#### 3. Frame Acknowledgment (optional)
```json
{
  "type": "ack",
  "frame_added": 42,
  "queue_size": 42
}
```

#### 4. Error
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Using the Python Client

### Basic Usage

```python
from push_client_example import AsyncPushGestureClient
import asyncio
import numpy as np

async def main():
    client = AsyncPushGestureClient(host='localhost', port=5000)
    await client.connect()
    
    # Post frames - predictions will be pushed automatically
    for i in range(50):
        frame_data = np.random.rand(100)
        await client.send_frame(frame_data)
        await asyncio.sleep(0.033)  # ~30 FPS
    
    # Wait for pending predictions
    await asyncio.sleep(2)
    await client.disconnect()

asyncio.run(main())
```

### Using Callbacks

```python
async def on_prediction(message):
    """Called when server pushes a prediction"""
    print(f"Detected: {message['prediction']} "
          f"({message['confidence']:.1%})")
    print(f"Text: {message['text_sequence']}")

async def on_status(message):
    """Called when server sends status update"""
    print(f"Status: {message['message']}")

client = AsyncPushGestureClient(host='localhost', port=5000)
client.set_prediction_callback(on_prediction)
client.set_status_callback(on_status)

await client.connect()

# Post frames...
for i in range(60):
    await client.send_frame(np.random.rand(100))
    await asyncio.sleep(0.033)

await client.disconnect()
```

### Continuous Streaming

```python
async def stream_frames(duration_seconds=10):
    """Stream frames for a specified duration"""
    
    async def on_prediction(msg):
        # Update UI with latest prediction
        print(f"\rCurrent: {msg['prediction']} | "
              f"Text: {' '.join(msg['text_sequence'])}", 
              end='', flush=True)
    
    client = AsyncPushGestureClient(host='localhost', port=5000)
    client.set_prediction_callback(on_prediction)
    
    await client.connect()
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        frame_data = get_camera_frame()  # Your frame source
        await client.send_frame(frame_data)
        await asyncio.sleep(0.033)
    
    await client.disconnect()
```

## Running Examples

```bash
python push_client_example.py
```

Available examples:
1. **Simple frame posting** - Basic usage
2. **Using callbacks** - Handle predictions with callbacks
3. **Continuous streaming** - Live display of predictions
4. **Multiple clients** - Concurrent client connections
5. **Commands** - Reset, clear, and status commands

## Advantages of Push Architecture

### Traditional (Request-Response)
```
Client: Send frame → Wait for response → Get prediction
        ↓ BLOCKS ↓
        (can't send next frame until response received)
```

### Push Architecture (This API)
```
Client: Send frame → Send frame → Send frame → ...
        ↓ NON-BLOCKING ↓
Server: Push prediction when ready (async)
```

**Benefits:**
- ✅ No blocking on frame submission
- ✅ Full-speed video streaming
- ✅ Server decides when to predict
- ✅ Efficient resource utilization
- ✅ Multiple predictions can be in flight
- ✅ Client code is simpler

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Frames to first prediction | 45 (30 + 15) |
| Subsequent predictions | Every 15 frames |
| Max queue size per client | 45 frames |
| Client posting speed | Unlimited (non-blocking) |
| Prediction latency | 50-100ms |
| Concurrent clients | 10+ (configurable) |

### Frame Timeline Example

```
Frame:  1  2  3  ... 30 31 32 ... 45 46 ... 60 61 ...
        ↓              ↓              ↓        ↓
Queue:  Filling...    Still filling  Predict! Predict!
                                     ↓        ↓
                                     Push     Push
                                     result   result
```

## Configuration

### Environment Variables

```bash
PORT=5000              # Server port
HOST=0.0.0.0          # Bind address (0.0.0.0 = all interfaces)
MAX_CONNECTIONS=10     # Max concurrent clients
WORKER_THREADS=4       # Thread pool size
```

### Performance Tuning

**For low-latency predictions:**
```python
SEQUENCE_LENGTH = 20   # Fewer frames needed
PREDICTION_STRIDE = 10 # Predict more frequently
# First prediction at 30 frames, then every 10
```

**For high-accuracy predictions:**
```python
SEQUENCE_LENGTH = 40   # More context
PREDICTION_STRIDE = 20 # Less frequent predictions
# First prediction at 60 frames, then every 20
```

**For real-time streaming:**
```python
SEQUENCE_LENGTH = 30   # Standard (1 second at 30 FPS)
PREDICTION_STRIDE = 15 # Good balance (0.5 seconds)
# Recommended for most use cases
```

## Integration with Video Processing

```python
import cv2
from data import collect_handle
from push_client_example import AsyncPushGestureClient
import asyncio

client = AsyncPushGestureClient(host='localhost', port=5000)

async def on_prediction(message):
    """Handle predictions from server"""
    global current_prediction
    current_prediction = message['prediction']
    print(f"Detected: {current_prediction}")

async def init():
    """Initialize client"""
    client.set_prediction_callback(on_prediction)
    await client.connect()

def process_video():
    """Process video frames"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract features from frame
        frame_data = collect_handle(hand_landmarks, pose_landmarks)
        
        # Post to server (non-blocking)
        asyncio.create_task(client.send_frame(frame_data))
        
        # Display
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run
loop = asyncio.new_event_loop()
loop.run_until_complete(init())
loop.run_in_executor(None, process_video)
loop.run_until_complete(client.disconnect())
```

## Troubleshooting

### "No predictions received"
- Wait for 45 frames (30 + 15) before first prediction
- Check that frame data format matches model input size
- Verify confidence threshold isn't too high

### "Queue size stuck at 45"
- This is normal - queue maintains sliding window
- Predictions are still being processed every 15 frames

### "Predictions too slow"
- Decrease `PREDICTION_STRIDE` (e.g., 10 instead of 15)
- Increase `WORKER_THREADS` for more parallel inference
- Check CPU/GPU utilization

### "Predictions too frequent"
- Increase `PREDICTION_STRIDE` (e.g., 20 instead of 15)
- Adjust `DEBOUNCE_TIME` to filter duplicates

## Security Considerations

⚠️ **No authentication or encryption** - For production:

- **TLS/SSL**: Wrap connections with encryption
- **Authentication**: Add token-based auth
- **Rate Limiting**: Limit frames per client
- **Input Validation**: Validate frame data size/format
- **Resource Limits**: Monitor memory per client
- **Firewall**: Restrict network access

## Comparison to Previous Architecture

| Feature | Old (Request-Response) | New (Push) |
|---------|----------------------|-----------|
| Client blocks on frame? | Yes | No |
| Prediction timing | Client-controlled | Server-controlled |
| Code complexity | Higher | Lower |
| Streaming performance | Limited | Optimal |
| Latency variance | High | Low |
| Resource efficiency | Moderate | High |

## Advanced: Custom Prediction Logic

Modify the prediction worker in `api.py`:

```python
async def prediction_worker(self):
    """Custom prediction logic"""
    while self.is_running:
        async with self.lock:
            queue_size = len(self.frame_queue)
        
        # Custom trigger: predict when queue >= CUSTOM_SIZE
        if queue_size >= CUSTOM_SIZE:
            # Your custom logic here
            prediction, confidence = await self.predict(queue_copy)
            await self.send_prediction(prediction, confidence, frame_num)
        
        await asyncio.sleep(0.01)
```

## License

MIT License - See LICENSE file for details

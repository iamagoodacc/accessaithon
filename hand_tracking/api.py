import asyncio
import json
import numpy as np
import torch
from model import RecognitionModel
from collections import deque
import time
import os
import sys
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configuration from .env
PORT = int(os.getenv('PORT', 5000))
HOST = os.getenv('HOST', '0.0.0.0')
MAX_CONNECTIONS = int(os.getenv('MAX_CONNECTIONS', 10))
WORKER_THREADS = int(os.getenv('WORKER_THREADS', 4))

SIGNS = ["hello", "yes"]

# Check if training data exists
if not os.path.exists(f"data/{SIGNS[0]}.npy"):
    print("ERROR: Training data not found!")
    print("Please run 'python collect_data.py' first to collect training data.")
    print("Then run 'python training.py' to train the model.")
    sys.exit(1)

sample_data = np.load(f"data/{SIGNS[0]}.npy")
input_size = sample_data.shape[2]

# Check if model exists
if not os.path.exists("recognition_model.pth"):
    print("ERROR: Trained model not found!")
    print("Please run 'python training.py' to train the model first.")
    sys.exit(1)

# Configuration constants
CONFIDENCE_THRESHOLD = 0.6
SEQUENCE_LENGTH = 30  # Number of frames needed for prediction
PREDICTION_STRIDE = 15  # Process prediction every N frames after SEQUENCE_LENGTH
DEBOUNCE_TIME = 2.0
MAX_TEXT_LENGTH = 100

# Load model
model = RecognitionModel(
    input_size=input_size,
    hidden_size=128,
    num_layers=2,
    output_size=len(SIGNS),
    dropout=0.0
)
model.load_state_dict(torch.load("recognition_model.pth"))
model.eval()

# Thread pool for CPU-intensive tasks (model inference)
executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)

print(f"Model loaded successfully. Input size: {input_size}")
print(f"Configuration: SEQUENCE_LENGTH={SEQUENCE_LENGTH}, PREDICTION_STRIDE={PREDICTION_STRIDE}")


class GestureRecognitionSession:
    """Handles gesture recognition state for a single client connection"""
    
    def __init__(self, session_id, writer):
        self.session_id = session_id
        self.writer = writer
        self.frame_queue = deque(maxlen=SEQUENCE_LENGTH + PREDICTION_STRIDE)
        self.text_sequence = []
        self.last_prediction = None
        self.last_prediction_time = 0
        self.frame_count = 0
        self.frames_since_last_prediction = 0
        self.lock = asyncio.Lock()
    
    def predict_sync(self, sequence):
        """Synchronous prediction (runs in thread pool)"""
        if len(sequence) < SEQUENCE_LENGTH:
            return None, 0.0

        # Take first SEQUENCE_LENGTH frames for prediction
        seq = list(sequence)[:SEQUENCE_LENGTH]

        x = torch.tensor(np.array(seq), dtype=torch.float32)
        x = x.unsqueeze(0)  # (1, 30, input_size)

        with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output, dim=1)
            conf, predicted = torch.max(probabilities, dim=1)

        if conf.item() < CONFIDENCE_THRESHOLD:
            return None, conf.item()

        return SIGNS[predicted.item()], conf.item()
    
    async def predict(self, sequence):
        """Async wrapper for prediction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.predict_sync, sequence)
    
    async def send_prediction(self, prediction, confidence, frame_count):
        """Send prediction result to client"""
        try:
            message = {
                'type': 'prediction',
                'prediction': prediction,
                'confidence': confidence,
                'frame_count': frame_count,
                'text_sequence': self.text_sequence.copy(),
                'queue_size': len(self.frame_queue)
            }
            await send_message(self.writer, message)
        except Exception as e:
            print(f"[Session {self.session_id}] Error sending prediction: {e}")
    
    async def send_status(self, status_type, message_text):
        """Send status message to client"""
        try:
            message = {
                'type': 'status',
                'status': status_type,
                'message': message_text,
                'frame_count': self.frame_count,
                'queue_size': len(self.frame_queue)
            }
            await send_message(self.writer, message)
        except Exception as e:
            print(f"[Session {self.session_id}] Error sending status: {e}")
    
    async def add_frame(self, frame_data):
        """Add a frame to the queue and check if prediction should be triggered"""
        async with self.lock:
            self.frame_queue.append(frame_data)
            self.frame_count += 1
            self.frames_since_last_prediction += 1
            
            queue_size = len(self.frame_queue)
            should_predict = False
            
            # Check if we should trigger a prediction
            if queue_size >= SEQUENCE_LENGTH + PREDICTION_STRIDE:
                if self.frames_since_last_prediction >= PREDICTION_STRIDE:
                    should_predict = True
                    # Copy queue for prediction
                    queue_copy = list(self.frame_queue)
        
        # Run prediction outside the lock if needed
        if should_predict:
            # Run prediction on the first SEQUENCE_LENGTH frames
            prediction, confidence = await self.predict(queue_copy)
            
            # Update state
            async with self.lock:
                if prediction is not None:
                    current_time = time.time()
                    time_since_last = current_time - self.last_prediction_time

                    # Debouncing: only add if different gesture or enough time has passed
                    if (self.last_prediction != prediction or time_since_last > DEBOUNCE_TIME):
                        self.text_sequence.append(prediction)
                        self.last_prediction = prediction
                        self.last_prediction_time = current_time

                        # Limit text length
                        if len(self.text_sequence) > MAX_TEXT_LENGTH:
                            self.text_sequence.pop(0)

                        print(f"[Session {self.session_id}] ✓ Detected: {prediction} ({confidence:.1%}) at frame {self.frame_count}")
            
            # Send prediction to client
            await self.send_prediction(prediction, confidence, self.frame_count)
            
            # Remove oldest PREDICTION_STRIDE frames to slide the window
            async with self.lock:
                for _ in range(PREDICTION_STRIDE):
                    if len(self.frame_queue) > 0:
                        self.frame_queue.popleft()
                
                # Reset the counter
                self.frames_since_last_prediction = 0
        
        async with self.lock:
            return {
                'frame_added': self.frame_count,
                'queue_size': len(self.frame_queue)
            }
    
    async def start(self):
        """Initialize session"""
        await self.send_status('connected', 'Session started, ready to receive frames')
    
    async def stop(self):
        """Cleanup session"""
        pass  # No worker to stop
    
    async def reset_buffer(self):
        """Reset the frame buffer"""
        async with self.lock:
            self.frame_queue.clear()
            self.frame_count = 0
            self.frames_since_last_prediction = 0
        await self.send_status('reset', 'Buffer reset')
    
    async def clear_text(self):
        """Clear accumulated text"""
        async with self.lock:
            self.text_sequence = []
            self.last_prediction = None
        await self.send_status('cleared', 'Text sequence cleared')
    
    async def get_status(self):
        """Get current status"""
        async with self.lock:
            return {
                'frame_count': self.frame_count,
                'queue_size': len(self.frame_queue),
                'text_sequence': self.text_sequence.copy(),
                'frames_since_last_prediction': self.frames_since_last_prediction
            }


class ConnectionPool:
    """Manages active connections with limits"""
    
    def __init__(self, max_connections):
        self.max_connections = max_connections
        self.active_connections = 0
        self.session_counter = 0
        self.lock = asyncio.Lock()
    
    async def can_accept(self):
        """Check if we can accept a new connection"""
        async with self.lock:
            return self.active_connections < self.max_connections
    
    async def add_connection(self):
        """Add a new connection"""
        async with self.lock:
            self.active_connections += 1
            self.session_counter += 1
            return self.session_counter
    
    async def remove_connection(self):
        """Remove a connection"""
        async with self.lock:
            self.active_connections = max(0, self.active_connections - 1)
    
    async def get_stats(self):
        """Get connection statistics"""
        async with self.lock:
            return {
                "active": self.active_connections,
                "max": self.max_connections,
                "total_sessions": self.session_counter
            }


# Global connection pool
connection_pool = ConnectionPool(MAX_CONNECTIONS)


async def receive_message(reader):
    """Receive a length-prefixed JSON message"""
    # Receive length (4 bytes)
    length_bytes = await reader.readexactly(4)
    if not length_bytes:
        return None
    
    msg_length = int.from_bytes(length_bytes, byteorder='big')
    
    # Receive message
    data = await reader.readexactly(msg_length)
    return json.loads(data.decode('utf-8'))


async def send_message(writer, message_dict):
    """Send a length-prefixed JSON message"""
    message_json = json.dumps(message_dict)
    message_bytes = message_json.encode('utf-8')
    message_length = len(message_bytes).to_bytes(4, byteorder='big')
    
    writer.write(message_length + message_bytes)
    await writer.drain()


async def handle_client(reader, writer):
    """Handle a single client connection asynchronously"""
    addr = writer.get_extra_info('peername')
    
    # Check if we can accept this connection
    if not await connection_pool.can_accept():
        print(f"[REJECTED] {addr} - Server at capacity")
        await send_message(writer, {
            'type': 'error',
            'message': 'Server at maximum capacity'
        })
        writer.close()
        await writer.wait_closed()
        return
    
    session_id = await connection_pool.add_connection()
    print(f"[NEW CONNECTION] Session {session_id} from {addr}")
    
    session = GestureRecognitionSession(session_id, writer)
    await session.start()  # Start the prediction worker
    
    try:
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(receive_message(reader), timeout=300.0)
                
                if message is None:
                    break
                
                command = message.get('command', 'frame')
                
                if command == 'frame':
                    # Add frame data to queue
                    frame_data = np.array(message['data'])
                    result = await session.add_frame(frame_data)
                    
                    # Send acknowledgment (optional, can be removed for performance)
                    await send_message(writer, {
                        'type': 'ack',
                        'frame_added': result['frame_added'],
                        'queue_size': result['queue_size']
                    })
                    
                elif command == 'reset':
                    # Reset buffer
                    await session.reset_buffer()
                    
                elif command == 'clear':
                    # Clear text sequence
                    await session.clear_text()
                    
                elif command == 'status':
                    # Get current status
                    status = await session.get_status()
                    await send_message(writer, {
                        'type': 'status',
                        **status
                    })
                    
                elif command == 'server_stats':
                    # Get server statistics
                    stats = await connection_pool.get_stats()
                    await send_message(writer, {
                        'type': 'stats',
                        'stats': stats
                    })
                    
                else:
                    await send_message(writer, {
                        'type': 'error',
                        'message': f'Unknown command: {command}'
                    })
                    
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] Session {session_id} - No activity for 5 minutes")
                await send_message(writer, {
                    'type': 'error',
                    'message': 'Connection timeout'
                })
                break
                
            except json.JSONDecodeError as e:
                await send_message(writer, {
                    'type': 'error',
                    'message': f'Invalid JSON: {str(e)}'
                })
                
            except Exception as e:
                print(f"[ERROR] Session {session_id}: {e}")
                await send_message(writer, {
                    'type': 'error',
                    'message': str(e)
                })
    
    except asyncio.IncompleteReadError:
        print(f"[DISCONNECTED] Session {session_id} - Connection closed by client")
    except Exception as e:
        print(f"[ERROR] Session {session_id}: {e}")
    
    finally:
        await session.stop()  # Stop the prediction worker
        await connection_pool.remove_connection()
        writer.close()
        await writer.wait_closed()
        print(f"[DISCONNECTED] Session {session_id} from {addr}")


async def start_server():
    """Start the async TCP server"""
    server = await asyncio.start_server(
        handle_client, 
        HOST, 
        PORT
    )
    
    addr = server.sockets[0].getsockname()
    print(f"[STARTING] Gesture Recognition API Server (Async Push Model)")
    print(f"[LISTENING] on {addr[0]}:{addr[1]}")
    print(f"[MODEL] Loaded with {len(SIGNS)} signs: {SIGNS}")
    print(f"[POOL] Max connections: {MAX_CONNECTIONS}")
    print(f"[WORKERS] Thread pool size: {WORKER_THREADS}")
    print(f"[CONFIG] Sequence length: {SEQUENCE_LENGTH}, Prediction stride: {PREDICTION_STRIDE}")
    print(f"[READY] Waiting for connections...")
    print("\nProtocol:")
    print("  - Client posts frames continuously")
    print("  - Server pushes predictions when ready (every 15 frames after 30 frames)")
    print("  - Commands: 'frame', 'reset', 'clear', 'status', 'server_stats'")
    print("  - Frame format: {'command': 'frame', 'data': [list of floats]}")
    print("\nMessage Types from Server:")
    print("  - 'prediction': Gesture detected")
    print("  - 'status': Status updates")
    print("  - 'ack': Frame acknowledgment")
    print("  - 'error': Error messages")
    print("\nPress Ctrl+C to stop the server\n")
    
    async with server:
        await server.serve_forever()


def main():
    """Main entry point"""
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\n[SHUTTING DOWN] Server stopped")
    finally:
        executor.shutdown(wait=True)
        print("[CLEANUP] Thread pool shut down")


if __name__ == '__main__':
    main()

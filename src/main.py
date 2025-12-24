import uvicorn
import argparse
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2

app = FastAPI()

import subprocess
import time

def get_camera_frame():
    # Start rpicam-vid as a subprocess streaming to a TCP port
    # We use a random high port or fixed one. Let's use 8888.
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--inline",
        "--listen",
        "-o", "tcp://0.0.0.0:8888",
        "--codec", "mjpeg",
        "--width", "640",
        "--height", "480",
        "--framerate", "15", # Lower framerate to reduce load
        "--vflip"
    ]
    
    print(f"Starting camera process: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Give it a moment to start
    time.sleep(2)
    
    # Connect OpenCV to the TCP stream
    print("Connecting OpenCV to tcp://127.0.0.1:8888")
    camera = cv2.VideoCapture("tcp://127.0.0.1:8888")

    if not camera.isOpened():
        print("Error: Could not connect to camera stream.")
        process.terminate()
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Failed to read frame from stream.")
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()
        process.terminate()
        print("Camera process terminated.")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(get_camera_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(public_path, "index.html"))

# Mount the public directory relative to this file
public_path = os.path.join(os.path.dirname(__file__), "public")
app.mount("/public", StaticFiles(directory=public_path), name="public")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")

    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
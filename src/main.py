import uvicorn
import argparse
import os
import threading
import subprocess
import time
import cv2


from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from cat_detector import CatDetector

class VideoCamera:
    def __init__(self):
        self.frame = None
        self.raw_frame = None
        self.condition = threading.Condition()
        self.running = False
        self.thread = None
        self.process = None
        self.camera = None

    def start(self):
        if self.running:
            return
        self.running = True
        
        # Start rpicam-vid as a subprocess streaming to a TCP port
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--inline",
            "--listen",
            "-o", "tcp://127.0.0.1:8888",
            "--codec", "mjpeg",
            "--width", "640",
            "--height", "480",
            "--framerate", "10",
            "--vflip",
        ]
        
        print(f"Starting camera process: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Give it a moment to start
        time.sleep(2)
        
        # Connect OpenCV to the TCP stream
        print("Connecting OpenCV to tcp://127.0.0.1:8888")
        self.camera = cv2.VideoCapture("tcp://127.0.0.1:8888")

        if not self.camera.isOpened():
            print("Error: Could not connect to camera stream.")
            self.stop()
            return

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()
        if self.process:
            self.process.terminate()
        print("Camera process terminated.")

    def update(self):
        print("Camera update loop started.")
        while self.running:
            if self.camera and self.camera.isOpened():
                success, frame = self.camera.read()
                if success:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        with self.condition:
                            self.frame = buffer.tobytes()
                            self.raw_frame = frame
                            self.condition.notify_all()
                else:
                    print("Error: Failed to read frame from stream.")
                    break
            else:
                break
        print("Camera update loop ended.")

camera = VideoCamera()
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global detector
    camera.start()
    detector = CatDetector()
    yield
    # Shutdown
    camera.stop()

app = FastAPI(lifespan=lifespan)


def get_camera_frame():
    while True:
        with camera.condition:
            camera.condition.wait()
            frame = camera.frame
        
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_processed_frame():
    while True:
        with camera.condition:
            camera.condition.wait()
            raw_frame = camera.raw_frame
        
        if raw_frame is not None:
            annotated_image, _ = detector.detect(raw_frame)
            if annotated_image is not None:
                ret, buffer = cv2.imencode('.jpg', annotated_image)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(get_camera_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed_processed")
async def video_feed_processed():
    return StreamingResponse(get_processed_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

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

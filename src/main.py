import uvicorn
import argparse
import os
import threading
import subprocess
import time
import cv2
import board
import psutil
import json
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from pydantic import BaseModel


from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from cat_detector import CatDetector

# Servo setup
pca = None
servo_pan = None
servo_tilt = None
servo_trigger = None

class TurretController:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(__file__), "turret_config.json")
        self.load_config()
        
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        self.tracking_enabled = False
        self.center_x = 320
        self.center_y = 240
        self.deadzone = 40
        
        # State for PID
        self.prev_error_x = 0
        self.prev_error_y = 0

    def load_config(self):
        defaults = {
            "pan_invert": False,
            "tilt_invert": False,
            "kp_pan": 0.02,
            "kd_pan": 0.005,
            "kp_tilt": 0.02,
            "kd_tilt": 0.005,
            "trigger_rest_angle": 45,
            "trigger_fire_angle": 180
        }
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = {**defaults, **json.load(f)}
            else:
                self.config = defaults
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = defaults

    def save_config(self, new_config):
        try:
            self.config.update(new_config)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def update(self, target_pos):
        global servo_pan, servo_tilt
        if not self.tracking_enabled or target_pos is None:
            # Reset PID state when not tracking
            self.prev_error_x = 0
            self.prev_error_y = 0
            return

        x, y = target_pos
        
        # Calculate error
        # Camera X: 0 (Left) -> 640 (Right)
        # Camera Y: 0 (Top) -> 480 (Bottom)
        
        # Error > 0 means target is to the Left/Top of center
        error_x = self.center_x - x 
        error_y = self.center_y - y 

        # Update Pan
        if abs(error_x) > self.deadzone:
            p_term = error_x * self.config["kp_pan"]
            d_term = (error_x - self.prev_error_x) * self.config["kd_pan"]
            delta = p_term + d_term
            
            if self.config["pan_invert"]:
                delta = -delta
            self.pan_angle += delta
            self.pan_angle = max(0, min(180, self.pan_angle))
            if servo_pan:
                servo_pan.angle = self.pan_angle
        self.prev_error_x = error_x

        # Update Tilt
        if abs(error_y) > self.deadzone:
            p_term = error_y * self.config["kp_tilt"]
            d_term = (error_y - self.prev_error_y) * self.config["kd_tilt"]
            delta = p_term + d_term
            
            if self.config["tilt_invert"]:
                delta = -delta
            self.tilt_angle += delta
            self.tilt_angle = max(0, min(180, self.tilt_angle))
            if servo_tilt:
                servo_tilt.angle = self.tilt_angle
        self.prev_error_y = error_y

    def set_manual(self, pan, tilt):
        global servo_pan, servo_tilt
        self.pan_angle = pan
        self.tilt_angle = tilt
        if servo_pan:
            servo_pan.angle = self.pan_angle
        if servo_tilt:
            servo_tilt.angle = self.tilt_angle

turret_controller = TurretController()

def init_servos():
    global pca, servo_pan, servo_tilt, servo_trigger
    try:
        print("Initializing servos...")
        i2c = board.I2C()
        pca = PCA9685(i2c)
        pca.frequency = 50
        servo_pan = servo.Servo(pca.channels[0])
        servo_tilt = servo.Servo(pca.channels[1])
        servo_trigger = servo.Servo(pca.channels[15])
        
        # Set initial position
        servo_pan.angle = 90
        servo_tilt.angle = 90
        servo_trigger.angle = turret_controller.config["trigger_rest_angle"]
        print("Servos initialized.")
    except Exception as e:
        print(f"Error initializing servos: {e}")

def deinit_servos():
    global pca
    if pca:
        print("Deinitializing servos...")
        pca.deinit()

class VideoCamera:
    def __init__(self):
        self.frame = None
        self.raw_frame = None
        self.capture_time = 0
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
            "--framerate", "15",
            "--vflip",
            "--hflip",
            "--autofocus-mode", "manual",
            "--lens-position", "0.0",
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
                    capture_time = time.time()
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        with self.condition:
                            self.frame = buffer.tobytes()
                            self.raw_frame = frame
                            self.capture_time = capture_time
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
    init_servos()
    yield
    # Shutdown
    camera.stop()
    deinit_servos()

app = FastAPI(lifespan=lifespan)


class ServoRequest(BaseModel):
    pan: float
    tilt: float

class TrackingRequest(BaseModel):
    enabled: bool

class ConfigRequest(BaseModel):
    pan_invert: bool
    tilt_invert: bool
    kp_pan: float
    kd_pan: float
    kp_tilt: float
    kd_tilt: float
    trigger_rest_angle: float = 45
    trigger_fire_angle: float = 180

@app.get("/config")
async def get_config():
    return turret_controller.config

@app.post("/config")
async def update_config(request: ConfigRequest):
    turret_controller.save_config(request.dict())
    return {"status": "ok", "config": turret_controller.config}

@app.post("/fire")
async def fire_turret():
    global servo_trigger
    if servo_trigger:
        try:
            # Fire sequence
            fire_angle = turret_controller.config["trigger_fire_angle"]
            rest_angle = turret_controller.config["trigger_rest_angle"]
            
            servo_trigger.angle = fire_angle
            time.sleep(0.5) # Hold for 0.5s
            servo_trigger.angle = rest_angle
            
            return {"status": "ok", "message": "Fired"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Trigger servo not initialized"}

@app.post("/set_tracking")
async def set_tracking(request: TrackingRequest):
    turret_controller.tracking_enabled = request.enabled
    return {"status": "ok", "enabled": turret_controller.tracking_enabled}

@app.post("/control_servos")
async def control_servos(request: ServoRequest):
    global servo_pan, servo_tilt
    if servo_pan and servo_tilt:
        try:
            # Clamp values
            pan = max(0, min(180, request.pan))
            tilt = max(0, min(180, request.tilt))
            
            # Disable tracking when manual control is used
            turret_controller.tracking_enabled = False
            turret_controller.set_manual(pan, tilt)
            
            return {"status": "ok", "pan": pan, "tilt": tilt}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Servos not initialized"}

@app.get("/system_stats")
async def system_stats():
    cpu_temp = None
    try:
        # Try psutil first (works on some systems)
        temps = psutil.sensors_temperatures()
        if 'cpu_thermal' in temps:
            cpu_temp = temps['cpu_thermal'][0].current
        elif 'coretemp' in temps:
             cpu_temp = temps['coretemp'][0].current
    except:
        pass

    if cpu_temp is None:
        try:
            # Fallback to reading file on Raspberry Pi
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                cpu_temp = float(f.read()) / 1000.0
        except:
            pass

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used": psutil.virtual_memory().used,
        "memory_total": psutil.virtual_memory().total,
        "cpu_temp": cpu_temp
    }


class TargetingSystem:
    def __init__(self):
        self.detection_start_time = None
        self.last_detection_time = 0
        self.dwell_threshold = 2.0
        self.animation_duration = 0.5
        self.locked = False

    def update(self, is_detected, current_time):
        if is_detected:
            if self.detection_start_time is None:
                self.detection_start_time = current_time
            self.last_detection_time = current_time
        else:
            # Reset if lost for more than 0.5s
            if current_time - self.last_detection_time > 0.5:
                self.detection_start_time = None
                self.locked = False

    def draw(self, image, center_pos, current_time):
        if self.detection_start_time is None or center_pos is None:
            return

        dwell_time = current_time - self.detection_start_time
        
        if dwell_time > self.dwell_threshold:
            self.locked = True
            # Animation phase (0.0 to 1.0)
            anim_time = dwell_time - self.dwell_threshold
            progress = min(1.0, anim_time / self.animation_duration)
            
            # Box size shrinking from 150 to 60
            start_size = 150
            end_size = 60
            current_size = int(start_size - (start_size - end_size) * progress)
            
            x, y = center_pos
            
            color = (0, 0, 255) # Red
            thickness = 2
            
            top_left = (x - current_size // 2, y - current_size // 2)
            bottom_right = (x + current_size // 2, y + current_size // 2)
            
            cv2.rectangle(image, top_left, bottom_right, color, thickness)
            
            if progress >= 1.0:
                # Draw crosshair inside
                cv2.line(image, (x - 10, y), (x + 10, y), color, 1)
                cv2.line(image, (x, y - 10), (x, y + 10), color, 1)
                cv2.putText(image, "TARGET LOCKED", (x - 60, y + current_size // 2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def get_camera_frame():
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        with camera.condition:
            camera.condition.wait()
            raw_frame = camera.raw_frame
        
        if raw_frame is not None:
            # Calculate FPS
            fps_counter += 1
            now = time.time()
            if now - fps_start_time > 1.0:
                fps = fps_counter / (now - fps_start_time)
                fps_counter = 0
                fps_start_time = now
            
            # Draw FPS
            display_frame = raw_frame.copy()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def get_processed_frame():
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    targeting_system = TargetingSystem()
    
    while True:
        with camera.condition:
            camera.condition.wait()
            raw_frame = camera.raw_frame
            capture_time = camera.capture_time
        
        if raw_frame is not None:
            annotated_image, (cat_pos, confidence) = detector.detect(raw_frame)
            
            # Update turret controller
            if cat_pos:
                turret_controller.update(cat_pos)

            # Update targeting system
            now = time.time()
            targeting_system.update(confidence > 0, now)

            if annotated_image is not None:
                # Draw targeting animation
                if cat_pos:
                    targeting_system.draw(annotated_image, cat_pos, now)

                # Calculate FPS
                fps_counter += 1
                if now - fps_start_time > 1.0:
                    fps = fps_counter / (now - fps_start_time)
                    fps_counter = 0
                    fps_start_time = now
                
                # Calculate Latency
                latency_ms = (now - capture_time) * 1000
                
                # Draw FPS and Latency
                cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_image, f"Lat: {latency_ms:.0f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
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

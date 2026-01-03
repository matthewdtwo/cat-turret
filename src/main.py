import uvicorn
import argparse
import os
import threading
import subprocess
import time
import asyncio
import cv2
import board
import psutil
import json
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from pylx16a.lx16a import LX16A, ServoTimeoutError
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


class ServoOutput:
    def __init__(self, rate_hz: float = 50.0):
        self._rate_hz = rate_hz
        self._servo_pan = None
        self._servo_tilt = None
        self._lock = threading.Lock()
        self._target_pan = None
        self._target_tilt = None
        self._stop_event = threading.Event()
        self._thread = None

    def attach(self, servo_pan_obj, servo_tilt_obj):
        with self._lock:
            self._servo_pan = servo_pan_obj
            self._servo_tilt = servo_tilt_obj

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def is_ready(self) -> bool:
        with self._lock:
            return self._servo_pan is not None and self._servo_tilt is not None

    def set_target(self, pan: float, tilt: float):
        with self._lock:
            self._target_pan = float(pan)
            self._target_tilt = float(tilt)

    def _run(self):
        period = 1.0 / max(1e-6, float(self._rate_hz))
        next_deadline = time.monotonic()
        last_pan = None
        last_tilt = None

        while not self._stop_event.is_set():
            with self._lock:
                servo_pan_obj = self._servo_pan
                servo_tilt_obj = self._servo_tilt
                target_pan = self._target_pan
                target_tilt = self._target_tilt

            if servo_pan_obj is not None and servo_tilt_obj is not None:
                try:
                    if target_pan is not None and target_pan != last_pan:
                        servo_pan_obj.move(target_pan)
                        last_pan = target_pan
                    if target_tilt is not None and target_tilt != last_tilt:
                        servo_tilt_obj.move(target_tilt)
                        last_tilt = target_tilt
                except Exception:
                    pass

            next_deadline += period
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_deadline = time.monotonic()


servo_output = ServoOutput(rate_hz=60.0)

class TurretController:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(__file__), "turret_config.json")
        self.load_config()
        
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        self.center_x = 320
        self.center_y = 240
        self.deadzone = 60
        self.calibration_mode = False
        
        # State for PID
        self.prev_error_x = 0
        self.prev_error_y = 0
        
        # State for Search/Return
        self.state = "IDLE" # IDLE, TRACKING, SEARCHING, RETURNING
        self.search_vel_pan = 0
        self.search_vel_tilt = 0
        self.last_pan = 90
        self.last_tilt = 90

    def load_config(self):
        def map_range(x, in_min=0, in_max=1000, out_min=0, out_max=240):
            return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

        # Limits from serial_bus_servo_control.py
        # Pan: 110 (anticlockwise) to 863 (clockwise)
        pan_min_deg = map_range(110)
        pan_max_deg = map_range(863)
        
        # Tilt: 595 (down) to 836 (up)
        tilt_min_deg = map_range(595)
        tilt_max_deg = map_range(836)

        defaults = {
            "pan_invert": False,
            "tilt_invert": False,
            "kp_pan": 0.02,
            "kd_pan": 0.005,
            "kp_tilt": 0.02,
            "kd_tilt": 0.005,
            "pan_min": pan_min_deg,
            "pan_max": pan_max_deg,
            "tilt_min": tilt_min_deg,
            "tilt_max": tilt_max_deg,
            "home_pan": (pan_min_deg + pan_max_deg) / 2,
            "home_tilt": (tilt_min_deg + tilt_max_deg) / 2,
            "trigger_rest_angle": 45,
            "trigger_fire_angle": 140,
            "tracking_enabled": False,
            "armed": False
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

    def save_config(self, new_config=None):
        try:
            if new_config:
                self.config.update(new_config)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def update(self, target_pos):
        # Safety: Don't move if not armed
        if not self.config.get("armed", False):
            return

        tracking_enabled = self.config.get("tracking_enabled", False)

        if not tracking_enabled:
            # Reset PID state when not tracking, unless returning home
            if self.state != "RETURNING":
                self.prev_error_x = 0
                self.prev_error_y = 0
                self.state = "IDLE"
                return

        if tracking_enabled and target_pos:
            x, y = target_pos
            
            # Bias the target position up by 20 pixels so the camera aims slightly above the target
            y -= 50
            
            # Safety check for extreme values (assuming 640x480 resolution with some margin)
            if not (-200 < x < 840) or not (-200 < y < 680):
                return

            # Initialize state on first frame of tracking to prevent jumping
            if self.state != "TRACKING":
                self.prev_error_x = self.center_x - x
                self.prev_error_y = self.center_y - y
                self.last_pan = self.pan_angle
                self.last_tilt = self.tilt_angle
                self.search_vel_pan = 0
                self.search_vel_tilt = 0

            self.state = "TRACKING"
            
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
                self.pan_angle = max(self.config["pan_min"], min(self.config["pan_max"], self.pan_angle))
            self.prev_error_x = error_x

            # Update Tilt
            if abs(error_y) > self.deadzone:
                p_term = error_y * self.config["kp_tilt"]
                d_term = (error_y - self.prev_error_y) * self.config["kd_tilt"]
                delta = p_term + d_term
                
                if self.config["tilt_invert"]:
                    delta = -delta
                self.tilt_angle += delta
                self.tilt_angle = max(self.config["tilt_min"], min(self.config["tilt_max"], self.tilt_angle))
            self.prev_error_y = error_y

            if servo_output.is_ready():
                servo_output.set_target(self.pan_angle, self.tilt_angle)
            
            # Update velocity estimate
            curr_vel_pan = self.pan_angle - self.last_pan
            curr_vel_tilt = self.tilt_angle - self.last_tilt
            alpha = 0.2
            self.search_vel_pan = (1 - alpha) * self.search_vel_pan + alpha * curr_vel_pan
            self.search_vel_tilt = (1 - alpha) * self.search_vel_tilt + alpha * curr_vel_tilt
            self.last_pan = self.pan_angle
            self.last_tilt = self.tilt_angle
            
        else:
            # Lost tracking logic
            if self.state == "TRACKING":
                self.state = "SEARCHING"
                # If velocity is negligible, skip search
                if abs(self.search_vel_pan) < 0.05 and abs(self.search_vel_tilt) < 0.05:
                    self.state = "RETURNING"
            
            if self.state == "SEARCHING":
                self.pan_angle += self.search_vel_pan
                self.tilt_angle += self.search_vel_tilt
                
                # Decay velocity to prevent overshoot
                self.search_vel_pan *= 0.95
                self.search_vel_tilt *= 0.95
                
                pan_min = self.config["pan_min"]
                pan_max = self.config["pan_max"]
                tilt_min = self.config["tilt_min"]
                tilt_max = self.config["tilt_max"]
                
                hit_limit = False
                if self.pan_angle <= pan_min or self.pan_angle >= pan_max: hit_limit = True
                if self.tilt_angle <= tilt_min or self.tilt_angle >= tilt_max: hit_limit = True
                
                self.pan_angle = max(pan_min, min(pan_max, self.pan_angle))
                self.tilt_angle = max(tilt_min, min(tilt_max, self.tilt_angle))

                if servo_output.is_ready():
                    servo_output.set_target(self.pan_angle, self.tilt_angle)
                
                # Stop searching if hit limit or stopped
                if hit_limit or (abs(self.search_vel_pan) < 0.01 and abs(self.search_vel_tilt) < 0.01):
                    self.state = "RETURNING"
            
            elif self.state == "RETURNING":
                pan_target = self.config.get("home_pan", 90)
                tilt_target = self.config.get("home_tilt", 90)
                speed = 0.2
                
                if abs(self.pan_angle - pan_target) > speed:
                    self.pan_angle += speed if pan_target > self.pan_angle else -speed
                if abs(self.tilt_angle - tilt_target) > speed:
                    self.tilt_angle += speed if tilt_target > self.tilt_angle else -speed

                if servo_output.is_ready():
                    servo_output.set_target(self.pan_angle, self.tilt_angle)
                
                if abs(self.pan_angle - pan_target) < 1 and abs(self.tilt_angle - tilt_target) < 1:
                    self.state = "IDLE"

    def set_manual(self, pan, tilt):
        self.pan_angle = pan
        self.tilt_angle = tilt
        if servo_output.is_ready():
            servo_output.set_target(self.pan_angle, self.tilt_angle)

turret_controller = TurretController()

def init_servos():
    global pca, servo_pan, servo_tilt, servo_trigger
    try:
        print("Initializing servos...")
        LX16A.initialize("/dev/ttyUSB0")
        
        try:
            servo_pan = LX16A(1)
            servo_tilt = LX16A(2)
            servo_pan.servo_mode()
            servo_tilt.servo_mode()
        except ServoTimeoutError as e:
            print(f"Servo {e.id_} is not responding.")
            # Don't return here, try to init trigger servo anyway
        
        # Initialize PCA9685 for trigger servo
        try:
            i2c = board.I2C()
            pca = PCA9685(i2c)
            pca.frequency = 50
            # Channel 16 is index 15
            servo_trigger = servo.Servo(pca.channels[15])
            
            # Set to rest position
            rest_angle = turret_controller.config.get("trigger_rest_angle", 45)
            servo_trigger.angle = rest_angle
            print("Trigger servo initialized.")
        except Exception as e:
            print(f"Error initializing trigger servo: {e}")
            servo_trigger = None
        
        # Set initial position
        if servo_pan and servo_tilt:
            servo_output.attach(servo_pan, servo_tilt)
            servo_output.start()
            
            # Move to home position
            home_pan = turret_controller.config.get("home_pan", 90)
            home_tilt = turret_controller.config.get("home_tilt", 90)
            servo_output.set_target(home_pan, home_tilt)
        
        print("Servos initialized.")
    except Exception as e:
        print(f"Error initializing servos: {e}")

def deinit_servos():
    global pca, servo_pan, servo_tilt, servo_trigger
    print("Centering servos...")
    try:
        if servo_output.is_ready():
            servo_output.set_target(
                turret_controller.config.get("home_pan", 90),
                turret_controller.config.get("home_tilt", 90),
            )
        time.sleep(0.5)
        
        # Deinit PCA
        if pca:
            pca.deinit()
            
    except Exception as e:
        print(f"Error centering servos: {e}")

    print("Deinitializing servos...")
    servo_output.stop()

class VideoCamera:
    def __init__(self):
        self.frame = None
        self.raw_frame = None
        self.preview_jpeg = None
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
            "-o", "tcp://0.0.0.0:8888",
            "--codec", "mjpeg",
            "--width", "640",
            "--height", "480",
            "--framerate", "25",
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
                    capture_time = time.monotonic()

                    preview_jpeg = None
                    try:
                        preview = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                        ret, buffer = cv2.imencode(
                            '.jpg',
                            preview,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                        )
                        if ret:
                            preview_jpeg = buffer.tobytes()
                    except Exception:
                        preview_jpeg = None

                    with self.condition:
                        self.raw_frame = frame
                        self.capture_time = capture_time
                        if preview_jpeg is not None:
                            self.preview_jpeg = preview_jpeg
                        self.condition.notify_all()
                else:
                    print("Error: Failed to read frame from stream.")
                    break
            else:
                break
        print("Camera update loop ended.")

class ProcessingThread:
    def __init__(self, camera, detector, controller):
        self.camera = camera
        self.detector = detector
        self.controller = controller
        self.running = False
        self.thread = None
        self.latest_processed_frame = None
        self.lock = threading.Lock()
        self.condition = threading.Condition()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _run(self):
        fps_start_time = time.monotonic()
        fps_counter = 0
        fps = 0
        targeting_system = TargetingSystem()

        while self.running:
            with self.camera.condition:
                self.camera.condition.wait()
                raw_frame = self.camera.raw_frame
                capture_time = self.camera.capture_time

            if raw_frame is not None:
                # Run detection and control
                annotated_image, (cat_pos, confidence) = self.detector.detect(raw_frame)
                self.controller.update(cat_pos)

                # Update targeting system
                now = time.monotonic()
                targeting_system.update(confidence > 0, now)

                if annotated_image is not None:
                    if cat_pos:
                        targeting_system.draw(annotated_image, cat_pos, now)

                    # Draw State
                    state_color = (0, 255, 0)
                    if self.controller.state == "SEARCHING": state_color = (0, 255, 255)
                    elif self.controller.state == "RETURNING": state_color = (0, 165, 255)
                    elif self.controller.state == "IDLE": state_color = (200, 200, 200)
                    
                    cv2.putText(annotated_image, f"State: {self.controller.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)

                    # Encode for streaming (only if needed? No, we need it for the generator)
                    # Optimization: Only encode if someone is watching? 
                    # For now, let's encode here but maybe we can optimize later.
                    # Actually, let's store the annotated image and let the generator encode it.
                    # That moves encoding out of the control loop!
                    
                    with self.condition:
                        self.latest_processed_frame = annotated_image
                        self.condition.notify_all()

camera = VideoCamera()
detector = None
processing_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global detector, processing_thread
    camera.start()
    detector = CatDetector()
    init_servos()
    
    processing_thread = ProcessingThread(camera, detector, turret_controller)
    processing_thread.start()
    
    yield
    # Shutdown
    if processing_thread:
        processing_thread.stop()
    if detector:
        detector.stop()
    camera.stop()
    deinit_servos()

app = FastAPI(lifespan=lifespan)


class ServoRequest(BaseModel):
    pan: float
    tilt: float

class TrackingRequest(BaseModel):
    enabled: bool

class ArmedRequest(BaseModel):
    enabled: bool

class DetectionTargetRequest(BaseModel):
    target: str

class SetLimitRequest(BaseModel):
    axis: str
    limit: str

class ConfigRequest(BaseModel):
    pan_invert: bool
    tilt_invert: bool
    kp_pan: float
    kd_pan: float
    kp_tilt: float
    kd_tilt: float
    pan_min: int = 0
    pan_max: int = 240
    tilt_min: int = 0
    tilt_max: int = 240
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
    if not turret_controller.config.get("armed", False):
        return {"status": "error", "message": "Turret is not armed"}

    if servo_trigger:
        try:
            # Fire sequence
            fire_angle = turret_controller.config["trigger_fire_angle"]
            rest_angle = turret_controller.config["trigger_rest_angle"]
            
            servo_trigger.angle = fire_angle
            await asyncio.sleep(0.5)
            servo_trigger.angle = rest_angle
            
            return {"status": "ok", "message": "Fired"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Trigger servo not initialized"}

@app.post("/set_tracking")
async def set_tracking(request: TrackingRequest):
    turret_controller.save_config({"tracking_enabled": request.enabled})
    return {"status": "ok", "enabled": turret_controller.config["tracking_enabled"]}

@app.post("/set_armed")
async def set_armed(request: ArmedRequest):
    turret_controller.save_config({"armed": request.enabled})
    return {"status": "ok", "enabled": turret_controller.config["armed"]}

@app.post("/set_detection_target")
async def set_detection_target(request: DetectionTargetRequest):
    global detector
    if detector:
        detector.set_target_class(request.target)
        return {"status": "ok", "target": detector.target_class}
    return {"status": "error", "message": "Detector not initialized"}

@app.post("/calibration/start")
async def start_calibration():
    turret_controller.calibration_mode = True
    return {"status": "ok", "mode": "calibration"}

@app.post("/calibration/stop")
async def stop_calibration():
    turret_controller.calibration_mode = False
    turret_controller.save_config()
    return {"status": "ok", "mode": "normal", "config": turret_controller.config}

@app.post("/calibration/set_limit")
async def set_limit(request: SetLimitRequest):
    if not turret_controller.calibration_mode:
        return {"status": "error", "message": "Not in calibration mode"}
    
    if request.axis == "pan":
        val = int(turret_controller.pan_angle)
        if request.limit == "min":
            turret_controller.config["pan_min"] = val
        elif request.limit == "max":
            turret_controller.config["pan_max"] = val
    elif request.axis == "tilt":
        val = int(turret_controller.tilt_angle)
        if request.limit == "min":
            turret_controller.config["tilt_min"] = val
        elif request.limit == "max":
            turret_controller.config["tilt_max"] = val
            
    return {"status": "ok", "config": turret_controller.config}

@app.post("/set_home")
async def set_home():
    turret_controller.config["home_pan"] = turret_controller.pan_angle
    turret_controller.config["home_tilt"] = turret_controller.tilt_angle
    turret_controller.save_config()
    return {"status": "ok", "config": turret_controller.config}

@app.post("/go_home")
async def go_home():
    if not turret_controller.config.get("armed", False):
        return {"status": "error", "message": "System not armed"}

    # Disable tracking
    turret_controller.save_config({"tracking_enabled": False})
    
    # Set state to RETURNING to smoothly move home
    turret_controller.state = "RETURNING"
    
    return {"status": "ok", "message": "Returning home"}

def test_range_sequence():
    global servo_pan, servo_tilt
    
    # Disable tracking
    turret_controller.tracking_enabled = False
    turret_controller.save_config({"tracking_enabled": False})
    
    pan_min = turret_controller.config["pan_min"]
    pan_max = turret_controller.config["pan_max"]
    tilt_min = turret_controller.config["tilt_min"]
    tilt_max = turret_controller.config["tilt_max"]
    
    center_pan = (pan_min + pan_max) / 2
    center_tilt = (tilt_min + tilt_max) / 2
    
    # Helper for smooth movement
    def move_servo_smooth(servo, target_angle, current_angle):
        if servo is None: return target_angle
        
        step = 1 if target_angle > current_angle else -1
        start = int(current_angle)
        end = int(target_angle)
        
        if start == end: return target_angle
        
        stop_val = end + 1 if step > 0 else end - 1
        
        for angle in range(start, stop_val, step):
            servo.move(angle)
            time.sleep(0.015) 
        return target_angle

    # Current positions
    curr_pan = turret_controller.pan_angle
    curr_tilt = turret_controller.tilt_angle

    # Pan Sequence
    curr_pan = move_servo_smooth(servo_pan, pan_min, curr_pan)
    time.sleep(0.2)
    curr_pan = move_servo_smooth(servo_pan, pan_max, curr_pan)
    time.sleep(0.2)
    curr_pan = move_servo_smooth(servo_pan, center_pan, curr_pan)
    
    # Tilt Sequence
    curr_tilt = move_servo_smooth(servo_tilt, tilt_min, curr_tilt)
    time.sleep(0.2)
    curr_tilt = move_servo_smooth(servo_tilt, tilt_max, curr_tilt)
    time.sleep(0.2)
    curr_tilt = move_servo_smooth(servo_tilt, center_tilt, curr_tilt)

    # Update controller state
    turret_controller.pan_angle = curr_pan
    turret_controller.tilt_angle = curr_tilt

@app.post("/test_range")
async def test_range_endpoint():
    if not turret_controller.config.get("armed", False):
        return {"status": "error", "message": "System not armed"}

    threading.Thread(target=test_range_sequence).start()
    return {"status": "ok", "message": "Range test started"}

@app.post("/control_servos")
async def control_servos(request: ServoRequest):
    if not turret_controller.config.get("armed", False):
        return {"status": "error", "message": "System not armed"}

    if servo_output.is_ready():
        try:
            # Clamp values
            if turret_controller.calibration_mode:
                pan = max(0, min(240, request.pan))
                tilt = max(0, min(240, request.tilt))
            else:
                pan = max(turret_controller.config["pan_min"], min(turret_controller.config["pan_max"], request.pan))
                tilt = max(turret_controller.config["tilt_min"], min(turret_controller.config["tilt_max"], request.tilt))
            
            # Disable tracking when manual control is used
            if turret_controller.config.get("tracking_enabled", False):
                turret_controller.save_config({"tracking_enabled": False})
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
        "cpu_temp": cpu_temp,
        "tracking_enabled": turret_controller.config.get("tracking_enabled", False),
        "armed": turret_controller.config.get("armed", False),
        "current_pan": turret_controller.pan_angle,
        "current_tilt": turret_controller.tilt_angle
    }


class TargetingSystem:
    def __init__(self):
        self.detection_start_time = None
        self.last_detection_time = 0
        self.dwell_threshold = 2.5
        self.animation_duration = 1
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
    while True:
        with camera.condition:
            camera.condition.wait()
            preview_jpeg = camera.preview_jpeg
        
        if preview_jpeg is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + preview_jpeg + b'\r\n'
            )

def get_processed_frame():
    while True:
        with processing_thread.condition:
            processing_thread.condition.wait()
            frame = processing_thread.latest_processed_frame
        
        if frame is not None:
            ret, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )
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

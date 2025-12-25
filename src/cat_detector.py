import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Import MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class CatTracker:
    def __init__(self):
        # Initialize Kalman Filter
        # State: [x, y, dx, dy]
        # Measurement: [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 5, 0],
                                            [0, 0, 0, 5]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 1
        self.found = False

    def update(self, measurement):
        """Update the filter with a new measurement (x, y)."""
        if not self.found:
            # First detection, initialize state
            self.kf.statePre = np.array([[measurement[0]], [measurement[1]], [0], [0]], np.float32)
            self.kf.statePost = np.array([[measurement[0]], [measurement[1]], [0], [0]], np.float32)
            self.found = True
        
        self.kf.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))

    def predict(self):
        """Predict the next state."""
        if self.found:
            prediction = self.kf.predict()
            return int(prediction[0]), int(prediction[1])
        return None

class CatDetector:
    def __init__(self, model_path='models/efficientdet_lite0.tflite'):
        # Resolve absolute path for the model
        if not os.path.isabs(model_path):
            # Assuming this file is in src/, and models/ is in src/models/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            
        print(f"Loading model from: {model_path}")
        
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            max_results=5, # max cats found
            score_threshold=0.4,
            category_allowlist=['cat']
        )
        self.detector = ObjectDetector.create_from_options(options)
        self.tracker = CatTracker()
        self.start_time = time.time()

    def detect(self, image):
        """
        Detects cats in the image.
        image: numpy array (BGR)
        Returns: (annotated_image, (position, confidence))
        position: (x, y) tuple of the Kalman Filter estimated position
        confidence: int (0-100) score of the detection
        """
        if image is None:
            return None, (None, 0)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        cat_detections = self._process_detections(detection_result)
        
        # Tracking logic
        predicted_pos = self.tracker.predict()
        final_pos = predicted_pos
        confidence = 0
        
        if cat_detections:
            target_cat = None
            if self.tracker.found and predicted_pos is not None:
                # Find closest detection to predicted position to avoid jumping between cats
                def distance(p1, p2):
                    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                target_cat = min(cat_detections, key=lambda x: distance(x[0], predicted_pos))
            else:
                # No track yet, pick the highest confidence one
                target_cat = max(cat_detections, key=lambda x: x[1])

            self.tracker.update(target_cat[0])
            confidence = target_cat[1]
            
            # Use the corrected state after update
            if self.tracker.found:
                final_pos = (int(self.tracker.kf.statePost[0, 0]), int(self.tracker.kf.statePost[1, 0]))
        
        annotated_image = self._draw_detections(image, cat_detections, final_pos)
        
        return annotated_image, (final_pos, confidence)

    def _process_detections(self, detection_result):
        cat_detections = []
        if detection_result.detections:
            print(f"Detections found: {len(detection_result.detections)}")
            for detection in detection_result.detections:
                category = detection.categories[0]
                print(f"Detected: {category.category_name} ({category.score:.2f})")
                if category.category_name == 'cat':
                    bbox = detection.bounding_box
                    center_x = int(bbox.origin_x + (bbox.width / 2))
                    center_y = int(bbox.origin_y + (bbox.height / 2))
                    confidence = category.score
                    cat_detections.append(((center_x, center_y), int(confidence*100)))
        return cat_detections

    def _draw_detections(self, image, detections, predicted_pos=None):
        annotated_image = image.copy()
        
        for (center, confidence) in detections:
            x, y = center
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
            text = f"Cat: {confidence:.2f}"
            cv2.putText(annotated_image, text, (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if predicted_pos:
            cv2.circle(annotated_image, predicted_pos, 5, (255, 0, 0), -1)
            cv2.putText(annotated_image, "KF", (predicted_pos[0] + 10, predicted_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
        return annotated_image

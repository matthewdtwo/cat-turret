

### 2025-12-25 servo control
- cat detection and tracking integrated. preliminary "target lock" functionality.
- tracking uses a PD loop to aim and target lock is based on target dwell time in frame.
- PD values adjustable via UI and stored in JSON locally.
- added pan/tilt servos via pca9685 for testing tracking. 

### 2025-12-24 Initial

- mediapipe on the pi requires python 3.11
- system install package libgl1
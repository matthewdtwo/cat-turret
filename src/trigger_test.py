import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
import time

def main():
    print("Initializing PCA9685 for Trigger Test...")
    try:
        i2c = board.I2C()
        pca = PCA9685(i2c)
        pca.frequency = 50
    except Exception as e:
        print(f"Failed to initialize I2C/PCA9685: {e}")
        return

    # Channel 16 is index 15
    TRIGGER_CHANNEL = 15
    print(f"Setting up trigger servo on channel {TRIGGER_CHANNEL}...")
    trigger_servo = servo.Servo(pca.channels[TRIGGER_CHANNEL])

    # Start at 90 degrees
    print("Setting initial position to 90 degrees...")
    trigger_servo.angle = 90

    print("Waiting for 5 seconds...")
    time.sleep(5)

    print("Running trigger test sequence...")
    
    try:
        for i in range(3):
            print(f"Iteration {i + 1}: Moving to 45 degrees...")
            trigger_servo.angle = 45
            time.sleep(0.5)
            
            print(f"Iteration {i + 1}: Moving to 180 degrees...")
            trigger_servo.angle = 180
            time.sleep(0.5)
            
        print("Test sequence complete.")

        trigger_servo.angle = 45

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("Deinitializing...")
        pca.deinit()

if __name__ == "__main__":
    main()

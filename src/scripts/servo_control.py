import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
import sys

def main():
    print("Initializing PCA9685...")
    try:
        i2c = board.I2C()
        pca = PCA9685(i2c)
        pca.frequency = 50
    except Exception as e:
        print(f"Failed to initialize I2C/PCA9685: {e}")
        return

    print("Setting up servos on channels 0 and 1...")
    servo0 = servo.Servo(pca.channels[0])
    servo1 = servo.Servo(pca.channels[1])

    print("Servo Control CLI")
    print("Enter angles for Servo 0 and Servo 1 separated by a space (e.g., '90 45').")
    print("Type 'q' or 'quit' to exit.")

    try:
        while True:
            user_input = input("Angles (S0 S1)> ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            try:
                parts = user_input.split()
                if len(parts) != 2:
                    print("Error: Please provide exactly two angles.")
                    continue
                
                angle0 = float(parts[0])
                angle1 = float(parts[1])

                if not (0 <= angle0 <= 180) or not (0 <= angle1 <= 180):
                    print("Error: Angles must be between 0 and 180.")
                    continue

                servo0.angle = angle0
                servo1.angle = angle1
                print(f"Set Servo 0 to {angle0}°, Servo 1 to {angle1}°")

            except ValueError:
                print("Error: Invalid number format.")
            except Exception as e:
                print(f"Error setting angles: {e}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("Deinitializing...")
        pca.deinit()

if __name__ == "__main__":
    main()

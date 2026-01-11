from pylx16a.lx16a import *

LX16A.initialize("/dev/ttyUSB0")

def map_range(x, in_min=0, in_max=1000, out_min=0, out_max=240):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def find_servos():

    # check IDs of the connected servos
    for id_ in range(1, 253):
        try:
            servo = LX16A(id_)
            model = servo.get_id()
            print(f"Found servo with ID {id_}, model: {model}")
        except ServoTimeoutError:
            pass  # No servo


def try_servo_movement():
    try:
        servo1 = LX16A(1)   # pan servo
        # servo2 = LX16A(2)   # tilt servo
        servo1.servo_mode()
        # servo2.servo_mode()

        print("Servos initialized in servo mode.")

        # move each relative to the current position 20

        servo1.move(map_range(500))
        # servo2.move(20, 1, True, True)


    except ServoTimeoutError as e:
        print(f"Servo {e.id_} is not responding. Exiting...")
        quit()


if __name__ == "__main__":
    find_servos()
    # try_servo_movement()
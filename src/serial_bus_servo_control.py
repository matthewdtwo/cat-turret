# using Hiwonder LX-16A servo with serial bus control

from math import sin, cos, pi
from pylx16a.lx16a import *
import time


LX16A.initialize("/dev/ttyUSB1)

def map_range(x, in_min=0, in_max=1000, out_min=0, out_max=240):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


try:
    servo1 = LX16A(1)   # pan servo
    servo2 = LX16A(2)   # tilt servo
    servo1.servo_mode()
    servo2.servo_mode()

except ServoTimeoutError as e:
    print(f"Servo {e.id_} is not responding. Exiting...")
    quit()


# pan angle position limits
# 863 - clockwise
# 110 - anticlockwise

# tilt servo position limits
# 836 - up
# 595 - down


servo1.move(map_range(863))
time.sleep(1)
servo1.move(map_range(110))
time.sleep(1)
servo1.move(map_range(486))

time.sleep(2)

servo2.move(map_range(800))
time.sleep(1)
servo2.move(map_range(595))
time.sleep(1)
servo2.move(map_range(655))


# t = 0
# while True:
#     servo1.move(sin(t) * 60 + 60)
#     servo2.move(cos(t) * 60 + 60)

#     time.sleep(0.05)
#     t += 0.1
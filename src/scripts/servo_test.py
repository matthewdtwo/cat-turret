# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time

import board
from adafruit_motor import servo

from adafruit_pca9685 import PCA9685

i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = busio.I2C(board.GP1, board.GP0)    # Pi Pico RP2040

# Create a simple PCA9685 class instance.
print("Initializing PCA9685...")
pca = PCA9685(i2c)
pca.frequency = 50

print("Setting up servo on channel 0 and 1...")
pan_servo = servo.Servo(pca.channels[0])

tilt_servo = servo.Servo(pca.channels[1])

pan_start_angle = 30
pan_stop_angle = 150

tilt_start_angle = 30
tilt_stop_angle = 150

try:
    # We sleep in the loops to give the servo time to move into position.
    print(f"Starting sweep {pan_start_angle}-{pan_stop_angle}...")
    for i in range(pan_start_angle, pan_stop_angle + 1):
        pan_servo.angle = i

        time.sleep(0.03)
    
    print(f"Starting sweep {pan_stop_angle}-{pan_start_angle}...")
    for i in range(pan_stop_angle, pan_start_angle - 1, -1):
        pan_servo.angle = i
        # servo1.angle = i
        time.sleep(0.03)


    print("Done panning")
    print(f"Starting tilt sweep {tilt_start_angle}-{tilt_stop_angle}...")

    for i in range(tilt_start_angle, tilt_stop_angle + 1):
        tilt_servo.angle = i
        time.sleep(0.03)
    print(f"Starting tilt sweep {tilt_stop_angle}-{tilt_start_angle}...")
    for i in range(tilt_stop_angle, tilt_start_angle - 1, -1):
        tilt_servo.angle = i
        time.sleep(0.03)
    print("Done tilting")

    # # You can also specify the movement fractionally.
    # print("Starting fractional movement...")
    # fraction = 0.0
    # while fraction < 1.0:
    #     servo7.fraction = fraction
    #     fraction += 0.01
    #     time.sleep(0.03)
    # print("Done.")

finally:

    # center the servo before exiting
    print("Centering servo...")
    pan_servo.angle = 90
    tilt_servo.angle = 90
    time.sleep(1)


    print("Deinitializing...")
    pca.deinit()
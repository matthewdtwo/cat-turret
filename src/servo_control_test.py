def _prompt_channel() -> int:
	while True:
		raw = input("Servo channel (0-15)> ").strip()
		if raw.lower() in {"q", "quit", "exit"}:
			raise SystemExit(0)
		try:
			channel = int(raw)
		except ValueError:
			print("Invalid channel. Enter an integer 0-15.")
			continue

		if 0 <= channel <= 15:
			return channel

		print("Channel out of range. Enter 0-15.")


def _prompt_angle() -> float | None:
	raw = input("Angle (0-180, 'c' for 90, 'q' to quit)> ").strip()
	if raw.lower() in {"q", "quit", "exit"}:
		return None
	if raw.lower() in {"c", "center", "home"}:
		return 90.0

	try:
		angle = float(raw)
	except ValueError:
		print("Invalid angle. Enter a number 0-180.")
		return 0.0 - 1.0

	if 0.0 <= angle <= 180.0:
		return angle

	print("Angle out of range. Enter 0-180.")
	return 0.0 - 1.0


def main() -> None:
	try:
		import board
		from adafruit_motor import servo
		from adafruit_pca9685 import PCA9685
	except ModuleNotFoundError as exc:
		print(f"Missing dependency: {exc}")
		print("Run this from the project env (from ./src):")
		print("  cd src && uv run ../servo_control_test.py")
		return

	print("Initializing PCA9685...")
	try:
		i2c = board.I2C()
		pca = PCA9685(i2c)
		pca.frequency = 50
	except Exception as exc:
		print(f"Failed to initialize I2C/PCA9685: {exc}")
		return

	servo_obj = None
	try:
		channel = _prompt_channel()
		print(f"Using channel {channel}.")
		servo_obj = servo.Servo(pca.channels[channel])

		while True:
			angle = _prompt_angle()
			if angle is None:
				break
			if angle < 0:
				continue

			try:
				servo_obj.angle = angle
				print(f"Set channel {channel} to {angle}")
			except Exception as exc:
				print(f"Failed to set angle: {exc}")

	except KeyboardInterrupt:
		print("\nExiting...")
	finally:
		try:
			if servo_obj is not None:
				servo_obj.angle = 90
		except Exception:
			pass
		try:
			pca.deinit()
		except Exception:
			pass


if __name__ == "__main__":
	main()


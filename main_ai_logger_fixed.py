
# M5Stack Core2 + Roller485 Lite x2 / UIFlow2 MicroPython
# PD teacher-data logger. Save on device as main.py.
import M5
import time
import math
import gc
from array import array
from machine import I2C, Pin
from unit import Roller485Unit

Kp = 40.0
Kd = 1.0
MAX_CURRENT = 400       # Roller485 command units; actual current not verified
PERIOD_US = 5000        # 5 ms; change to 10000 if timing is unstable
MAX_DT_US = 10000       # stop if period exceeds 10 ms
MAX_SAMPLES = 1000      # 5 seconds at 5 ms
START_ANGLE = 5.0
FALL_ANGLE = 20.0
TAU = 0.5
CALIBRATION_SAMPLES = 200
CSV_FILE = 'pd_training.csv'

WHITE = 0xFFFFFF
CYAN = 0x00FFFF
YELLOW = 0xFFFF00
GREEN = 0x00FF00
BLUE = 0x0000FF
RED = 0xFF0000
MAGENTA = 0xFF00FF
motor1 = None
motor2 = None


def display(color):
    M5.Lcd.fillScreen(color)


def stop_motors():
    for motor in (motor1, motor2):
        if motor is not None:
            try:
                motor.set_motor_max_current(0)
            except Exception as e:
                print('Zero current error:', e)
    for motor in (motor1, motor2):
        if motor is not None:
            try:
                motor.set_motor_output_state(0)
            except Exception as e:
                print('Disable output error:', e)


def run():
    global motor1, motor2
    M5.begin()
    display(WHITE)
    time.sleep_ms(2000)
    i2c = I2C(0, scl=Pin(33), sda=Pin(32), freq=100000)
    devices = i2c.scan()
    print('I2C devices:', devices)
    if 0x64 not in devices or 0x65 not in devices:
        raise RuntimeError('Motor not found')

    motor1 = Roller485Unit(i2c, address=0x64,
                           mode=Roller485Unit.I2C_MODE)
    motor2 = Roller485Unit(i2c, address=0x65,
                           mode=Roller485Unit.I2C_MODE)
    for motor in (motor1, motor2):
        motor.set_motor_output_state(0)
        motor.set_motor_mode(3)
        motor.set_motor_max_current(0)

    angle_sum = 0.0
    gx_sum = 0.0
    for _ in range(CALIBRATION_SAMPLES):
        ax, ay, az = M5.Imu.getAccel()
        gx, gy, gz = M5.Imu.getGyro()
        angle_sum += math.degrees(math.atan2(ay, az))
        gx_sum += gx
        time.sleep_ms(10)
    angle_offset = angle_sum / CALIBRATION_SAMPLES
    gx_bias = gx_sum / CALIBRATION_SAMPLES
    print('Angle offset:', angle_offset)
    print('Gyro bias:', gx_bias)

    # Preallocate bounded RAM buffers. No file writes in control loop.
    theta_log = array('f', [0.0 for _ in range(MAX_SAMPLES)])
    omega_log = array('f', [0.0 for _ in range(MAX_SAMPLES)])
    command_log = array('h', [0 for _ in range(MAX_SAMPLES)])
    time_log = array('I', [0 for _ in range(MAX_SAMPLES)])

    display(CYAN)
    print('Press Button A to start')
    while True:
        M5.update()
        if M5.BtnA.wasPressed():
            break
        time.sleep_ms(20)
    display(YELLOW)
    while True:
        M5.update()
        if not M5.BtnA.isPressed():
            break
        time.sleep_ms(10)
    time.sleep_ms(100)
    M5.update()

    ax, ay, az = M5.Imu.getAccel()
    angle = math.degrees(math.atan2(ay, az))
    theta = angle - angle_offset
    print('Initial theta:', theta)
    if abs(theta) > START_ANGLE:
        print('Stopped: Start angle limit')
        display(RED)
        return

    count = 0
    max_command = 0
    max_theta = 0.0
    max_dt_us = 0
    reason = 'Buffer full'
    gc.collect()
    gc.disable()  # bounded run: <= 5 s, restore in finally
    try:
        motor1.set_motor_max_current(0)
        motor2.set_motor_max_current(0)
        motor1.set_motor_output_state(1)
        motor2.set_motor_output_state(1)
        display(GREEN)
        last_us = time.ticks_us()
        next_us = time.ticks_add(last_us, PERIOD_US)
        start_us = last_us

        while count < MAX_SAMPLES:
            while time.ticks_diff(next_us, time.ticks_us()) > 0:
                pass
            now = time.ticks_us()
            dt_us = time.ticks_diff(now, last_us)
            last_us = now
            if dt_us > max_dt_us:
                max_dt_us = dt_us
            if dt_us <= 0 or dt_us > MAX_DT_US:
                reason = 'Cycle timeout'
                break
            M5.update()
            if M5.BtnA.wasPressed():
                reason = 'Button A'
                break
            dt = dt_us / 1000000.0
            ax, ay, az = M5.Imu.getAccel()
            gx, gy, gz = M5.Imu.getGyro()
            gx -= gx_bias
            accel_angle = math.degrees(math.atan2(ay, az))
            alpha = TAU / (TAU + dt)
            angle = alpha * (angle + gx * dt) + (1.0 - alpha) * accel_angle
            theta = angle - angle_offset
            if abs(theta) > max_theta:
                max_theta = abs(theta)
            if abs(theta) > FALL_ANGLE:
                reason = 'Fall detected'
                break
            u = -Kp * theta - Kd * gx
            if u > MAX_CURRENT:
                u = MAX_CURRENT
            elif u < -MAX_CURRENT:
                u = -MAX_CURRENT
            current = int(u)
            if abs(current) > max_command:
                max_command = abs(current)

            motor1.set_motor_max_current(current)
            motor2.set_motor_max_current(-current)

            # Save actual PD input/output, including command clipping.
            time_log[count] = time.ticks_diff(now, start_us) // 1000
            theta_log[count] = theta
            omega_log[count] = gx
            command_log[count] = current
            count += 1
            next_us = time.ticks_add(next_us, PERIOD_US)
    finally:
        stop_motors()
        gc.enable()

    display(BLUE if reason in ('Button A', 'Buffer full') else RED)
    print('Stopped:', reason)
    print('Samples:', count)
    print('Max command:', max_command)
    print('Max angle:', max_theta)
    print('Max dt (ms):', max_dt_us / 1000)

    # Save ONLY after motor outputs have been disabled.
    if count > 0:
        try:
            with open(CSV_FILE, 'w') as f:
                f.write('time_ms,theta_deg,omega_deg_s,command\n')
                for i in range(count):
                    f.write('{},{:.5f},{:.5f},{}\n'.format(
                        time_log[i], theta_log[i], omega_log[i], command_log[i]))
            print('CSV saved:', CSV_FILE, count, 'rows')
        except Exception as e:
            print('CSV save error:', e)
            display(MAGENTA)


try:
    run()
except Exception as e:
    print('Error:', e)
    stop_motors()
    try:
        display(MAGENTA)
    except Exception:
        pass
finally:
    gc.enable()

while True:
    time.sleep_ms(200)

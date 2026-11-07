# M5Stack Core2 + Roller485 Lite x2 / UIFlow2 MicroPython
# 100% AI, 60-second run, no CSV/RAM sample buffers. Save as main.py.
import M5
import time
import math
import gc
from machine import I2C, Pin
from unit import Roller485Unit

Kp = 40.0
Kd = 1.0
MAX_CURRENT = 400       # Roller485 command units; actual current not verified
PERIOD_US = 10000       # 10 ms
MAX_DT_US = 20000       # stop if period exceeds 20 ms
MAX_SAMPLES = 6000      # 60 seconds at 10 ms
AI_RATIO = 1.00         # 100% AI; PD is safety reference only
MAX_AI_PD_DIFF = 80.0   # stop if AI deviates strongly from PD
START_ANGLE = 5.0
FALL_ANGLE = 20.0
TAU = 0.5
CALIBRATION_SAMPLES = 200

WHITE = 0xFFFFFF
CYAN = 0x00FFFF
YELLOW = 0xFFFF00
GREEN = 0x00FF00
BLUE = 0x0000FF
RED = 0xFF0000
MAGENTA = 0xFF00FF
motor1 = None
motor2 = None


# Trained on first 800 rows of pd_training.csv (chronological split).
# 2 -> 4 tanh -> 1, fixed input scales 5 deg and 200 deg/s.
AI_W1 = [[-0.43181439420669016, 0.49029859004418563, -0.5474875682416946, 0.49654225170292277], [0.3276726015039394, 0.42059541941510203, -0.013318604675247082, 0.515756483077586]]
AI_B1 = [-0.38739629946371107, -0.6429722846257957, -0.995702039163207, 0.8359608387644251]
AI_W2 = [-0.1518221269255342, -0.9160419813232847, 0.26287991572923536, -0.6929072493483193]
AI_B2 = 0.09821041846511148


def ai_predict(theta, omega):
    x0 = theta / 5.0
    x1 = omega / 200.0
    v = AI_B2
    for j in range(4):
        h = math.tanh(x0 * AI_W1[0][j] + x1 * AI_W1[1][j] + AI_B1[j])
        v += h * AI_W2[j]
    return v * 400.0

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
    max_ai_diff = 0.0
    max_blended_command = 0
    sum_ai_diff = 0.0
    reason = '60 seconds completed'
    gc.enable()
    gc.collect()
    # Keep GC enabled for the longer run; collect before enabling motors.
    try:
        motor1.set_motor_max_current(0)
        motor2.set_motor_max_current(0)
        motor1.set_motor_output_state(1)
        motor2.set_motor_output_state(1)
        display(GREEN)
        last_us = time.ticks_us()
        next_us = time.ticks_add(last_us, PERIOD_US)

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
            # AI predicts motor command; PD is computed only for safety comparison.
            ai_u = ai_predict(theta, gx)
            if ai_u > MAX_CURRENT:
                ai_u = MAX_CURRENT
            elif ai_u < -MAX_CURRENT:
                ai_u = -MAX_CURRENT
            if not (-MAX_CURRENT <= ai_u <= MAX_CURRENT):
                reason = 'Invalid AI command'
                break
            difference = abs(ai_u - current)
            if difference > MAX_AI_PD_DIFF:
                reason = 'AI-PD divergence'
                break
            if difference > max_ai_diff:
                max_ai_diff = difference
            sum_ai_diff += difference
            if abs(current) > max_command:
                max_command = abs(current)

            blended = int(ai_u)  # 100% AI command
            if blended > MAX_CURRENT:
                blended = MAX_CURRENT
            elif blended < -MAX_CURRENT:
                blended = -MAX_CURRENT
            if abs(blended) > max_blended_command:
                max_blended_command = abs(blended)

            motor1.set_motor_max_current(blended)
            motor2.set_motor_max_current(-blended)

            count += 1
            next_us = time.ticks_add(next_us, PERIOD_US)
    finally:
        stop_motors()
        gc.enable()

    display(BLUE if reason in ('Button A', '60 seconds completed') else RED)
    print('Stopped:', reason)
    print('Cycles:', count)
    print('Elapsed target (s):', MAX_SAMPLES * PERIOD_US / 1000000)
    print('Max PD command:', max_command)
    print('Max AI motor command:', max_blended_command)
    print('Max angle:', max_theta)
    print('Max dt (ms):', max_dt_us / 1000)
    print('AI MAE:', sum_ai_diff / count if count else None)
    print('AI max difference:', max_ai_diff)



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

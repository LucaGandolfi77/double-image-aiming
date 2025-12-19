import ctypes
import cv2
import numpy as np
import time
import os
import math

# --- Configuration ---
LIB_PATH = "./libstereo.so"
WIDTH = 640
HEIGHT = 480
BASELINE = 2.0  # Meters
FOCAL_LENGTH = 800.0 # Pixels (hypothetical value, needs calibration in reality)
THRESHOLD = 100 # Brightness threshold (0-255). Below = object, Above = sky
MIN_PIXELS = 10 # Minimum number of pixels to consider an object

# --- C Structure Definition ---
# AERONAUTICAL STANDARD: Rigorous Types (Matching C int32_t and float64_t)
class StereoResult(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_double),
        ("disparity", ctypes.c_double),
        ("object_found", ctypes.c_int32),
        ("left_x", ctypes.c_int32),
        ("left_y", ctypes.c_int32),
        ("right_x", ctypes.c_int32),
        ("right_y", ctypes.c_int32)
    ]

class PIDState(ctypes.Structure):
    _fields_ = [
        ("Kp", ctypes.c_double),
        ("Ki", ctypes.c_double),
        ("Kd", ctypes.c_double),
        ("prev_error", ctypes.c_double),
        ("integral", ctypes.c_double)
    ]

class KalmanState(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("vx", ctypes.c_double),
        ("vy", ctypes.c_double),
        ("P", ctypes.c_double * 4 * 4),
        ("Q", ctypes.c_double * 4 * 4),
        ("R", ctypes.c_double * 2 * 2)
    ]

# --- AI Supervisor (High-Level Control) ---
class AISupervisor:
    def __init__(self, laser_controller):
        self.laser = laser_controller
        self.error_history = []
        self.history_len = 20
        self.maneuver_threshold = 50.0 # Pixel deviation
        
    def analyze(self, aim_x, aim_y, raw_x, raw_y, dt):
        # 1. Anomaly Detection (Maneuver Detection)
        # Compare Kalman prediction (aim) with raw measurement
        # If difference is huge, object is maneuvering faster than model
        residual = math.sqrt((aim_x - raw_x)**2 + (aim_y - raw_y)**2)
        
        if residual > self.maneuver_threshold:
            print(f"[AI] MANEUVER DETECTED! Residual: {residual:.1f}. Boosting Kalman Q.")
            # Increase Process Noise (Q) to trust measurement more
            # Accessing C struct directly
            self.laser.kf.Q[0][0] = 100.0
            self.laser.kf.Q[1][1] = 100.0
            self.laser.kf.Q[2][2] = 100.0
            self.laser.kf.Q[3][3] = 100.0
        else:
            # Decay Q back to normal (10.0)
            if self.laser.kf.Q[0][0] > 10.0:
                self.laser.kf.Q[0][0] *= 0.95
                self.laser.kf.Q[1][1] *= 0.95
                self.laser.kf.Q[2][2] *= 0.95
                self.laser.kf.Q[3][3] *= 0.95

        # 2. Adaptive PID Tuning
        # Monitor tracking error (difference between center and aim)
        # Ideally, we want aim to be center (WIDTH/2, HEIGHT/2) eventually? 
        # No, aim is where we point. The error is (Target - Current_Laser_Pos).
        # But we don't have feedback of actual laser pos here, only the command.
        # We can monitor the stability of the command.
        
        self.error_history.append(residual)
        if len(self.error_history) > self.history_len:
            self.error_history.pop(0)
            
        avg_error = sum(self.error_history) / len(self.error_history)
        
        # Simple Heuristic: If error is consistently low, increase Kp for faster response
        # If error is high/oscillating, decrease Kp
        if avg_error < 5.0:
            self.laser.pid_yaw.Kp = min(self.laser.pid_yaw.Kp * 1.01, 5.0) # Cap at 5.0
            self.laser.pid_pitch.Kp = min(self.laser.pid_pitch.Kp * 1.01, 5.0)
        elif avg_error > 20.0:
            self.laser.pid_yaw.Kp = max(self.laser.pid_yaw.Kp * 0.99, 0.5) # Floor at 0.5
            self.laser.pid_pitch.Kp = max(self.laser.pid_pitch.Kp * 0.99, 0.5)

# --- Watchdog Timer ---
class Watchdog:
    def __init__(self, timeout_sec):
        self.timeout = timeout_sec
        self.last_kick = time.time()
    
    def kick(self):
        self.last_kick = time.time()
        
    def check(self):
        if (time.time() - self.last_kick) > self.timeout:
            return False # Watchdog expired
        return True

# --- Load C Library ---
def load_c_lib():
    if not os.path.exists(LIB_PATH):
        print(f"Error: {LIB_PATH} not found. Run 'make' first.")
        exit(1)
    
    lib = ctypes.CDLL(LIB_PATH)
    
    # Define function arguments
    lib.process_stereo_frame.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte), # img_left
        ctypes.POINTER(ctypes.c_ubyte), # img_right
        ctypes.c_int32,                 # width
        ctypes.c_int32,                 # height
        ctypes.c_double,                # baseline
        ctypes.c_double,                # focal_length
        ctypes.c_int32,                 # threshold
        ctypes.c_int32,                 # min_pixels
        ctypes.c_int32,                 # prev_lx
        ctypes.c_int32,                 # prev_ly
        ctypes.c_int32,                 # prev_rx
        ctypes.c_int32,                 # prev_ry
        ctypes.POINTER(StereoResult)    # result struct
    ]
    
    # Define arguments for laser control function
    lib.set_laser_angles.argtypes = [ctypes.c_double, ctypes.c_double]
    
    # Define arguments for PID functions
    lib.pid_init.argtypes = [ctypes.POINTER(PIDState), ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.pid_compute.argtypes = [ctypes.POINTER(PIDState), ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.pid_compute.restype = ctypes.c_double

    # Define arguments for Kalman Filter functions
    lib.kalman_init.argtypes = [ctypes.POINTER(KalmanState), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.kalman_predict.argtypes = [ctypes.POINTER(KalmanState), ctypes.c_double]
    lib.kalman_update.argtypes = [ctypes.POINTER(KalmanState), ctypes.c_double, ctypes.c_double]
    lib.kalman_get_prediction.argtypes = [ctypes.POINTER(KalmanState), ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

    return lib

# --- Physics Class ---
class PhysicsTracker:
    def __init__(self):
        self.prev_distance = None
        self.prev_velocity = 0.0
        self.prev_time = None
        
        self.velocity = 0.0      # m/s
        self.acceleration = 0.0  # m/s^2

    def update(self, current_distance):
        current_time = time.time()
        
        if self.prev_distance is None:
            self.prev_distance = current_distance
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        if dt <= 0: return

        # Calculate Velocity: v = (d2 - d1) / t
        # Note: If object approaches, distance decreases, negative velocity.
        # Here we calculate the scalar velocity of approach/recession
        new_velocity = (current_distance - self.prev_distance) / dt
        
        # Calculate Acceleration: a = (v2 - v1) / t
        self.acceleration = (new_velocity - self.prev_velocity) / dt
        self.velocity = new_velocity

        # Update previous state
        self.prev_distance = current_distance
        self.prev_velocity = new_velocity
        self.prev_time = current_time

# --- Laser Pointer Controller ---
class LaserController:
    def __init__(self, width, height, focal_length, lib):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.lib = lib
        self.yaw = 0.0   # Left/Right angle in degrees
        self.pitch = 0.0 # Up/Down angle in degrees
        
        # Initialize PID controllers
        self.pid_yaw = PIDState()
        self.pid_pitch = PIDState()
        
        # Tune PID values (Kp, Ki, Kd)
        # Kp: Proportional gain (speed of approach)
        # Ki: Integral gain (corrects steady-state error)
        # Kd: Derivative gain (dampens overshoot)
        # We treat the PID output as angular velocity (deg/s)
        self.lib.pid_init(ctypes.byref(self.pid_yaw), 2.0, 0.5, 0.1)
        self.lib.pid_init(ctypes.byref(self.pid_pitch), 2.0, 0.5, 0.1)
        
        # Initialize Kalman Filter for aiming (Center X, Center Y)
        self.kf = KalmanState()
        # Init with center of screen, Q=10 (process noise), R=5 (measurement noise)
        self.lib.kalman_init(ctypes.byref(self.kf), width/2.0, height/2.0, 10.0, 5.0)
        
        self.last_time = time.time()

    def update(self, target_x, target_y, found=True):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.001 # Avoid div by zero
        self.last_time = current_time

        # 1. Kalman Predict
        self.lib.kalman_predict(ctypes.byref(self.kf), dt)

        # 2. Kalman Update (only if object found)
        if found:
            self.lib.kalman_update(ctypes.byref(self.kf), target_x, target_y)
        
        # 3. Get Prediction (Lookahead)
        # Predict where object will be in 0.1s (e.g. to account for lag)
        pred_x = ctypes.c_double()
        pred_y = ctypes.c_double()
        lookahead = 0.1 
        self.lib.kalman_get_prediction(ctypes.byref(self.kf), lookahead, ctypes.byref(pred_x), ctypes.byref(pred_y))
        
        # Use predicted coordinates for aiming
        aim_x = pred_x.value
        aim_y = pred_y.value

        # Calculate deviation from the image center (optical axis)
        dx = aim_x - (self.width / 2.0)
        dy = aim_y - (self.height / 2.0)

        # Calculate angles using trigonometry
        # tan(theta) = opposite / adjacent = dx / focal_length
        yaw_rad = math.atan(dx / self.focal_length)
        pitch_rad = math.atan(dy / self.focal_length)

        target_yaw = math.degrees(yaw_rad)
        target_pitch = -math.degrees(pitch_rad) # Invert Y for standard pitch

        # Apply PID Control
        # The PID computes the angular velocity needed to reach the target
        yaw_velocity = self.lib.pid_compute(ctypes.byref(self.pid_yaw), target_yaw, self.yaw, dt)
        pitch_velocity = self.lib.pid_compute(ctypes.byref(self.pid_pitch), target_pitch, self.pitch, dt)

        # Update position: pos += vel * dt
        self.yaw += yaw_velocity * dt
        self.pitch += pitch_velocity * dt

        return self.yaw, self.pitch, aim_x, aim_y

# --- Telemetry Logger (Black Box) ---
class TelemetryLogger:
    def __init__(self, filename="flight_data.csv"):
        self.filename = filename
        self.start_time = time.time()
        # Write Header
        with open(self.filename, "w") as f:
            f.write("Time,Dist,Vel,RawX,RawY,PredX,PredY,Yaw,Pitch,Found\n")
            
    def log(self, dist, vel, raw_x, raw_y, pred_x, pred_y, yaw, pitch, found):
        t = time.time() - self.start_time
        with open(self.filename, "a") as f:
            f.write(f"{t:.3f},{dist:.2f},{vel:.2f},{raw_x},{raw_y},{pred_x:.1f},{pred_y:.1f},{yaw:.2f},{pitch:.2f},{int(found)}\n")

# --- Simulation Generator (Mock) with Fault Injection ---
# Creates two images with a black dot moving to simulate approach
class MockCamera:
    def __init__(self):
        self.frame_count = 0
        self.fault_noise = False
        self.fault_occlusion = False
        
    def inject_fault(self, fault_type):
        if fault_type == 'noise': self.fault_noise = True
        elif fault_type == 'occlusion': self.fault_occlusion = True
        elif fault_type == 'clear': 
            self.fault_noise = False
            self.fault_occlusion = False
        
    def get_frames(self):
        # Create white background (sky)
        img_l = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255
        img_r = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255
        
        # FAULT: Occlusion (Return blank frames)
        if self.fault_occlusion:
            self.frame_count += 1
            return img_l, img_r

        # Simulate approaching object (disparity increases)
        # Frame 0: Disparity 10px -> Far
        # Frame 100: Disparity 100px -> Near
        
        # Simulate sinusoidal movement for distance (disparity)
        shift = abs(np.sin(self.frame_count * 0.05)) * 50 + 10 
        
        # Simulate movement in X and Y (aiming)
        # Move center X back and forth
        offset_x = np.sin(self.frame_count * 0.1) * 100 
        # Move Y up and down
        offset_y = np.cos(self.frame_count * 0.1) * 50
        
        base_x = (WIDTH // 2) + offset_x
        y = int((HEIGHT // 2) + offset_y)
        
        # Disparity = shift * 2 (a bit left, a bit right)
        lx = int(base_x + shift)
        rx = int(base_x - shift)
        
        # Draw object (dark circle)
        cv2.circle(img_l, (lx, y), 20, (50), -1) # Dark gray
        cv2.circle(img_r, (rx, y), 20, (50), -1)
        
        # FAULT: Noise Burst (Add random noise)
        if self.fault_noise:
            noise = np.random.randint(0, 100, (HEIGHT, WIDTH), dtype=np.uint8)
            img_l = cv2.add(img_l, noise)
            img_r = cv2.add(img_r, noise)
        
        self.frame_count += 1
        time.sleep(0.05) # Simulate 20fps
        
        return img_l, img_r

# --- Main ---
def main():
    lib = load_c_lib()
    tracker = PhysicsTracker()
    laser = LaserController(WIDTH, HEIGHT, FOCAL_LENGTH, lib)
    
    # Use MockCamera to test without hardware. 
    # To use real webcams: capL = cv2.VideoCapture(0), capR = cv2.VideoCapture(1)
    camera = MockCamera() 
    
    print("Starting stereo tracking...")
    print(f"Baseline: {BASELINE}m")
    
    # Initialize previous coordinates for tracking optimization
    prev_lx, prev_ly = -1, -1
    prev_rx, prev_ry = -1, -1

    # Initialize Watchdog (100ms timeout)
    wd = Watchdog(0.1)
    
    # Initialize Telemetry Logger
    logger = TelemetryLogger()
    
    # Initialize AI Supervisor
    ai = AISupervisor(laser)
    
    # Fault Injection Instructions
    print("\n--- CONTROLS ---")
    print("Press 'n' to inject NOISE")
    print("Press 'o' to inject OCCLUSION")
    print("Press 'c' to CLEAR faults")
    print("Press 'q' to QUIT")
    print("----------------\n")

    try:
        while True:
            # Kick the watchdog at start of loop
            wd.kick()

            # 1. Acquisition
            frame_l, frame_r = camera.get_frames()
            
            # Ensure data is C-contiguous (important for ctypes)
            if not frame_l.flags['C_CONTIGUOUS']: frame_l = np.ascontiguousarray(frame_l)
            if not frame_r.flags['C_CONTIGUOUS']: frame_r = np.ascontiguousarray(frame_r)

            # 2. Prepare data for C
            c_res = StereoResult()
            p_frame_l = frame_l.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            p_frame_r = frame_r.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            
            # 3. Call C library (Heavy lifting)
            lib.process_stereo_frame(
                p_frame_l, p_frame_r, 
                WIDTH, HEIGHT, 
                BASELINE, FOCAL_LENGTH, 
                THRESHOLD, 
                MIN_PIXELS,
                prev_lx, prev_ly,
                prev_rx, prev_ry,
                ctypes.byref(c_res)
            )
            
            # 4. Physics Processing (Python)
            if c_res.object_found:
                # Update previous coordinates for next frame
                prev_lx, prev_ly = c_res.left_x, c_res.left_y
                prev_rx, prev_ry = c_res.right_x, c_res.right_y

                tracker.update(c_res.distance)
                
                # Calculate center for Kalman Filter
                center_x = (c_res.left_x + c_res.right_x) / 2.0
                center_y = (c_res.left_y + c_res.right_y) / 2.0
                
                yaw, pitch, aim_x, aim_y = laser.update(center_x, center_y, found=True)
                
                # AI Analysis & Control
                # Calculate dt (approx 0.05s from sleep)
                ai.analyze(aim_x, aim_y, center_x, center_y, 0.05)
                
                # Call C function to control hardware
                lib.set_laser_angles(yaw, pitch)
                
                # Log Data
                logger.log(c_res.distance, tracker.velocity, center_x, center_y, aim_x, aim_y, yaw, pitch, True)
                
                # Console Output
                print(f"Dist: {c_res.distance:.2f}m | "
                      f"Vel: {tracker.velocity:.2f}m/s | "
                      f"Laser -> Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}° | "
                      f"Aim: ({aim_x:.0f}, {aim_y:.0f}) | "
                      f"Kp: {laser.pid_yaw.Kp:.2f}")
                
                # Visualization (Draw info on left frame)
                display_img = cv2.cvtColor(frame_l, cv2.COLOR_GRAY2BGR)
                cv2.putText(display_img, f"Dist: {c_res.distance:.2f}m", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(display_img, (c_res.left_x, c_res.left_y), 5, (0, 255, 0), 2)
                # Draw predicted aim point
                cv2.circle(display_img, (int(aim_x), int(aim_y)), 5, (255, 0, 0), 2)
                
                # Note: In a headless environment (no monitor), cv2.imshow might fail.
                # Comment out lines below if running on remote server.
                # cv2.imshow("Stereo Tracker (Left)", display_img)
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q'): break
                # elif key == ord('n'): camera.inject_fault('noise')
                # elif key == ord('o'): camera.inject_fault('occlusion')
                # elif key == ord('c'): camera.inject_fault('clear')

            else:
                print("Object not found. Predicting...")
                # Reset tracking if object is lost
                prev_lx, prev_ly = -1, -1
                
                # Still update laser with prediction
                yaw, pitch, aim_x, aim_y = laser.update(0, 0, found=False)
                lib.set_laser_angles(yaw, pitch)
                
                # Log Data (with 0 for raw values)
                logger.log(0, 0, 0, 0, aim_x, aim_y, yaw, pitch, False)

            # Check Watchdog
            if not wd.check():
                print("[CRITICAL] WATCHDOG TIMEOUT! Resetting system...")
                # In a real system, this would reboot hardware.
                # Here we just reset the loop state.
                prev_lx, prev_ly = -1, -1
                wd.kick() # Reset watchdog
                prev_rx, prev_ry = -1, -1
                
                # Update Laser with prediction only (found=False)
                yaw, pitch, aim_x, aim_y = laser.update(0, 0, found=False)
                lib.set_laser_angles(yaw, pitch)
                print(f"Laser -> Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}° (Predicted)")

    except KeyboardInterrupt:
        print("\nStopping...")
    
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

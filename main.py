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
class StereoResult(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_double),
        ("disparity", ctypes.c_double),
        ("object_found", ctypes.c_int),
        ("left_x", ctypes.c_int),
        ("left_y", ctypes.c_int),
        ("right_x", ctypes.c_int),
        ("right_y", ctypes.c_int)
    ]

class PIDState(ctypes.Structure):
    _fields_ = [
        ("Kp", ctypes.c_double),
        ("Ki", ctypes.c_double),
        ("Kd", ctypes.c_double),
        ("prev_error", ctypes.c_double),
        ("integral", ctypes.c_double)
    ]

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
        ctypes.c_int,                   # width
        ctypes.c_int,                   # height
        ctypes.c_double,                # baseline
        ctypes.c_double,                # focal_length
        ctypes.c_int,                   # threshold
        ctypes.c_int,                   # min_pixels
        ctypes.c_int,                   # prev_lx
        ctypes.c_int,                   # prev_ly
        ctypes.c_int,                   # prev_rx
        ctypes.c_int,                   # prev_ry
        ctypes.POINTER(StereoResult)    # result struct
    ]
    
    # Define arguments for laser control function
    lib.set_laser_angles.argtypes = [ctypes.c_double, ctypes.c_double]
    
    # Define arguments for PID functions
    lib.pid_init.argtypes = [ctypes.POINTER(PIDState), ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.pid_compute.argtypes = [ctypes.POINTER(PIDState), ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.pid_compute.restype = ctypes.c_double

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
        
        self.last_time = time.time()

    def update(self, left_x, left_y, right_x, right_y):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.001 # Avoid div by zero
        self.last_time = current_time

        # Calculate the center of the object in the "virtual center camera"
        # We approximate this as the midpoint between left and right detections
        center_x = (left_x + right_x) / 2.0
        center_y = (left_y + right_y) / 2.0

        # Calculate deviation from the image center (optical axis)
        dx = center_x - (self.width / 2.0)
        dy = center_y - (self.height / 2.0)

        # Calculate angles using trigonometry
        # tan(theta) = opposite / adjacent = dx / focal_length
        # Note: In computer vision, Y increases downwards. 
        # For a laser pointer, usually "up" means positive pitch.
        # So we might need to invert dy depending on the servo setup.
        # Here we assume positive dy (down in image) -> negative pitch (downwards)
        
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

        return self.yaw, self.pitch

# --- Simulation Generator (Mock) ---
# Creates two images with a black dot moving to simulate approach
class MockCamera:
    def __init__(self):
        self.frame_count = 0
        
    def get_frames(self):
        # Create white background (sky)
        img_l = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255
        img_r = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255
        
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

    try:
        while True:
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
                yaw, pitch = laser.update(c_res.left_x, c_res.left_y, c_res.right_x, c_res.right_y)
                
                # Call C function to control hardware
                lib.set_laser_angles(yaw, pitch)
                
                # Console Output
                print(f"Dist: {c_res.distance:.2f}m | "
                      f"Vel: {tracker.velocity:.2f}m/s | "
                      f"Acc: {tracker.acceleration:.2f}m/s^2 | "
                      f"Laser -> Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°")
                
                # Visualization (Draw info on left frame)
                display_img = cv2.cvtColor(frame_l, cv2.COLOR_GRAY2BGR)
                cv2.putText(display_img, f"Dist: {c_res.distance:.2f}m", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(display_img, (c_res.left_x, c_res.left_y), 5, (0, 255, 0), 2)
                
                # Note: In a headless environment (no monitor), cv2.imshow might fail.
                # Comment out lines below if running on remote server.
                # cv2.imshow("Stereo Tracker (Left)", display_img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                print("Object not found.")
                # Reset tracking if object is lost
                prev_lx, prev_ly = -1, -1
                prev_rx, prev_ry = -1, -1

    except KeyboardInterrupt:
        print("\nStopping...")
    
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

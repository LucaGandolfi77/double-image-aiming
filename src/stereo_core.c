#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Structure to return results to the wrapper
typedef struct {
    double distance;      // Distance in meters
    double disparity;     // Disparity in pixels
    int object_found;     // 1 if object found in both frames, 0 otherwise
    int left_x;           // Left centroid X coordinate
    int left_y;           // Left centroid Y coordinate
    int right_x;          // Right centroid X coordinate
    int right_y;          // Right centroid Y coordinate
} StereoResult;

// PID Controller Structure
typedef struct {
    double Kp;
    double Ki;
    double Kd;
    double prev_error;
    double integral;
} PIDState;

// Initialize PID Controller
void pid_init(PIDState* pid, double Kp, double Ki, double Kd) {
    pid->Kp = Kp;
    pid->Ki = Ki;
    pid->Kd = Kd;
    pid->prev_error = 0.0;
    pid->integral = 0.0;
}

// Compute PID Output
double pid_compute(PIDState* pid, double setpoint, double measured, double dt) {
    if (dt <= 0.0) return 0.0;

    double error = setpoint - measured;
    
    // Integral term
    pid->integral += error * dt;
    
    // Derivative term
    double derivative = (error - pid->prev_error) / dt;
    
    // PID Output
    double output = (pid->Kp * error) + (pid->Ki * pid->integral) + (pid->Kd * derivative);
    
    pid->prev_error = error;
    
    return output;
}

// Kalman Filter Structure (Constant Velocity Model)
// State: [x, y, vx, vy]
typedef struct {
    double x, y;      // Position
    double vx, vy;    // Velocity
    double P[4][4];   // Error Covariance Matrix
    double Q[4][4];   // Process Noise Covariance
    double R[2][2];   // Measurement Noise Covariance
} KalmanState;

// Initialize Kalman Filter
void kalman_init(KalmanState* kf, double init_x, double init_y, double q_noise, double r_noise) {
    kf->x = init_x;
    kf->y = init_y;
    kf->vx = 0.0;
    kf->vy = 0.0;

    // Initialize P (Identity * large value for high uncertainty initially)
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) kf->P[i][j] = (i==j) ? 1000.0 : 0.0;

    // Initialize Q (Process Noise)
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) kf->Q[i][j] = 0.0;
    kf->Q[0][0] = q_noise * 0.1; kf->Q[1][1] = q_noise * 0.1; // Position noise
    kf->Q[2][2] = q_noise;       kf->Q[3][3] = q_noise;       // Velocity noise

    // Initialize R (Measurement Noise)
    kf->R[0][0] = r_noise; kf->R[0][1] = 0.0;
    kf->R[1][0] = 0.0;     kf->R[1][1] = r_noise;
}

// Predict Step
void kalman_predict(KalmanState* kf, double dt) {
    // 1. Predict State: x = F * x
    kf->x += kf->vx * dt;
    kf->y += kf->vy * dt;

    // 2. Predict Covariance: P = F * P * F^T + Q
    double P_new[4][4];
    double F[4][4] = {
        {1, 0, dt, 0},
        {0, 1, 0, dt},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    
    // Temp = P * F^T
    double Temp[4][4];
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            Temp[i][j] = 0;
            for(int k=0; k<4; k++) {
                Temp[i][j] += kf->P[i][k] * F[j][k]; 
            }
        }
    }
    
    // P_new = F * Temp + Q
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            P_new[i][j] = 0;
            for(int k=0; k<4; k++) {
                P_new[i][j] += F[i][k] * Temp[k][j];
            }
            P_new[i][j] += kf->Q[i][j];
        }
    }
    
    // Copy back
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) kf->P[i][j] = P_new[i][j];
}

// Update Step
void kalman_update(KalmanState* kf, double meas_x, double meas_y) {
    // y = z - H * x
    double y_res[2];
    y_res[0] = meas_x - kf->x;
    y_res[1] = meas_y - kf->y;

    // S = H * P * H^T + R
    double S[2][2];
    S[0][0] = kf->P[0][0] + kf->R[0][0];
    S[0][1] = kf->P[0][1] + kf->R[0][1];
    S[1][0] = kf->P[1][0] + kf->R[1][0];
    S[1][1] = kf->P[1][1] + kf->R[1][1];

    // Inverse of S
    double det = S[0][0]*S[1][1] - S[0][1]*S[1][0];
    if(fabs(det) < 1e-6) return; 
    double S_inv[2][2];
    S_inv[0][0] =  S[1][1] / det;
    S_inv[0][1] = -S[0][1] / det;
    S_inv[1][0] = -S[1][0] / det;
    S_inv[1][1] =  S[0][0] / det;

    // K = P * H^T * S_inv
    double K[4][2];
    for(int i=0; i<4; i++) {
        K[i][0] = kf->P[i][0] * S_inv[0][0] + kf->P[i][1] * S_inv[1][0];
        K[i][1] = kf->P[i][0] * S_inv[0][1] + kf->P[i][1] * S_inv[1][1];
    }

    // Update State: x = x + K * y
    kf->x  += K[0][0] * y_res[0] + K[0][1] * y_res[1];
    kf->y  += K[1][0] * y_res[0] + K[1][1] * y_res[1];
    kf->vx += K[2][0] * y_res[0] + K[2][1] * y_res[1];
    kf->vy += K[3][0] * y_res[0] + K[3][1] * y_res[1];

    // Update Covariance: P = (I - K * H) * P
    double P_new[4][4];
    double I_KH[4][4];
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) I_KH[i][j] = (i==j) ? 1.0 : 0.0;
    
    for(int i=0; i<4; i++) {
        I_KH[i][0] -= K[i][0];
        I_KH[i][1] -= K[i][1];
    }

    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            P_new[i][j] = 0;
            for(int k=0; k<4; k++) {
                P_new[i][j] += I_KH[i][k] * kf->P[k][j];
            }
        }
    }
    
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) kf->P[i][j] = P_new[i][j];
}

// Get predicted position at future time
void kalman_get_prediction(KalmanState* kf, double dt_ahead, double* out_x, double* out_y) {
    *out_x = kf->x + kf->vx * dt_ahead;
    *out_y = kf->y + kf->vy * dt_ahead;
}

// Helper function to find the centroid with ROI and Noise Filtering
int find_centroid(const unsigned char* img, int width, int height, int threshold, int min_pixels, int start_x, int start_y, int search_w, int search_h, int* out_x, int* out_y) {
    long sum_x = 0;
    long sum_y = 0;
    long count = 0;

    // Clamp ROI to image boundaries
    int end_x = start_x + search_w;
    int end_y = start_y + search_h;
    if (start_x < 0) start_x = 0;
    if (start_y < 0) start_y = 0;
    if (end_x > width) end_x = width;
    if (end_y > height) end_y = height;

    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            // Image is a 1D array. Index = y * width + x
            unsigned char pixel_val = img[y * width + x];

            if (pixel_val < threshold) {
                sum_x += x;
                sum_y += y;
                count++;
            }
        }
    }

    // Filter noise: object must be bigger than min_pixels
    if (count > min_pixels) {
        *out_x = (int)(sum_x / count);
        *out_y = (int)(sum_y / count);
        return 1; // Found
    }

    return 0; // Not found
}

// Function to control the laser pointer hardware
// In a real scenario, this would write to a serial port or GPIO
void set_laser_angles(double yaw, double pitch) {
    // Simulate hardware control
    // For example, mapping degrees to servo PWM values (0-255) or similar
    // Here we just print to stdout to verify the C function is called
    printf("[C-CORE] HARDWARE CONTROL: Setting Laser -> Yaw: %.2f, Pitch: %.2f\n", yaw, pitch);
}

// Main exported function
// prev_lx, prev_ly, etc: coordinates from previous frame for ROI (-1 if not found previously)
void process_stereo_frame(
    const unsigned char* img_left, 
    const unsigned char* img_right, 
    int width, int height, 
    double baseline, 
    double focal_length, 
    int threshold, 
    int min_pixels,
    int prev_lx, int prev_ly,
    int prev_rx, int prev_ry,
    StereoResult* result
) {
    int lx, ly, rx, ry;
    int found_l = 0;
    int found_r = 0;
    
    // ROI Configuration
    int roi_size = 100; // Search window size (100x100)
    
    // Search Left
    if (prev_lx != -1 && prev_ly != -1) {
        // Fast search in ROI
        found_l = find_centroid(img_left, width, height, threshold, min_pixels, 
                                prev_lx - roi_size/2, prev_ly - roi_size/2, roi_size, roi_size, &lx, &ly);
    }
    // If not found in ROI (or no prev pos), search full image
    if (!found_l) {
        found_l = find_centroid(img_left, width, height, threshold, min_pixels, 
                                0, 0, width, height, &lx, &ly);
    }

    // Search Right
    if (prev_rx != -1 && prev_ry != -1) {
        found_r = find_centroid(img_right, width, height, threshold, min_pixels, 
                                prev_rx - roi_size/2, prev_ry - roi_size/2, roi_size, roi_size, &rx, &ry);
    }
    if (!found_r) {
        found_r = find_centroid(img_right, width, height, threshold, min_pixels, 
                                0, 0, width, height, &rx, &ry);
    }

    if (found_l && found_r) {
        result->object_found = 1;
        result->left_x = lx;
        result->left_y = ly;
        result->right_x = rx;
        result->right_y = ry;

        // Calculate Disparity: d = x_left - x_right
        // (Assuming rectified images where y is same, but we use centroids so y might differ slightly)
        double disparity = (double)(lx - rx);
        
        // Avoid division by zero
        if (disparity <= 0.1) disparity = 0.1;

        result->disparity = disparity;

        // Triangulation Formula: Z = (f * b) / d
        result->distance = (focal_length * baseline) / disparity;
    } else {
        result->object_found = 0;
        result->distance = -1.0;
        result->disparity = 0.0;
    }
}

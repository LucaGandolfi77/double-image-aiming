#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../src/stereo_core.c"

// Simple Test Framework
#define ASSERT_NEAR(a, b, tol) if(fabs(a-b) > tol) { printf("FAIL: %f != %f\n", a, b); return 0; }
#define TEST_PASS() printf("PASS\n"); return 1;

int test_pid_safety() {
    printf("Test: PID Safety (Clamping & Anti-Windup)... ");
    PIDState pid;
    // Kp=1, Ki=100, Kd=0
    pid_init(&pid, 1.0, 100.0, 0.0); 
    
    // Step 1: Large error (100) for 1 second
    // Expected Integral: 100 * 1 = 100 -> Clamped to 50
    // Expected Output: 1*100 + 100*50 + 0 = 5100 -> Clamped to 300
    double out = pid_compute(&pid, 100.0, 0.0, 1.0);
    
    ASSERT_NEAR(pid.integral, 50.0, 0.1);
    ASSERT_NEAR(out, 300.0, 0.1);
    
    TEST_PASS();
}

int test_kalman_prediction() {
    printf("Test: Kalman Prediction (Constant Velocity)... ");
    KalmanState kf;
    kalman_init(&kf, 0.0, 0.0, 1.0, 1.0);
    
    // Simulate moving object: x = 10*t, y = 0
    // t=0: x=0
    // t=1: x=10 (Update)
    kalman_predict(&kf, 1.0);
    kalman_update(&kf, 10.0, 0.0);
    
    // t=2: x=20 (Update)
    kalman_predict(&kf, 1.0);
    kalman_update(&kf, 20.0, 0.0);
    
    // Now velocity should be approx 10 m/s
    // Predict t=3 (Lookahead 1.0s) -> Should be 30.0
    double pred_x, pred_y;
    kalman_get_prediction(&kf, 1.0, &pred_x, &pred_y);
    
    // Allow some convergence error
    if (fabs(pred_x - 30.0) > 5.0) {
        printf("FAIL: PredX %f != 30.0\n", pred_x);
        return 0;
    }
    
    TEST_PASS();
}

int main() {
    printf("=== Running Aeronautical Standard Unit Tests ===\n");
    int passed = 0;
    passed += test_pid_safety();
    passed += test_kalman_prediction();
    
    printf("Tests Passed: %d/2\n", passed);
    return 0;
}

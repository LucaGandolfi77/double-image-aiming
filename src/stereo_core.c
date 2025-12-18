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

// Helper function to find the centroid of a dark object on a light background
// Assumes grayscale input (1 byte per pixel) for simplicity and speed
int find_centroid(const unsigned char* img, int width, int height, int threshold, int* out_x, int* out_y) {
    long sum_x = 0;
    long sum_y = 0;
    long count = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Image is a 1D array. Index = y * width + x
            unsigned char pixel_val = img[y * width + x];

            // Threshold logic:
            // If pixel is DARKER than threshold, it is part of the object (sky is light)
            if (pixel_val < threshold) {
                sum_x += x;
                sum_y += y;
                count++;
            }
        }
    }

    if (count > 0) {
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
// img_left, img_right: pointers to raw image buffers (grayscale)
// width, height: image dimensions
// baseline: distance between cameras in meters (e.g., 2.0)
// focal_length: focal length in pixels (depends on camera and resolution)
// threshold: value 0-255 to distinguish object from sky
void process_stereo_frame(
    const unsigned char* img_left, 
    const unsigned char* img_right, 
    int width, 
    int height, 
    double baseline, 
    double focal_length, 
    int threshold,
    StereoResult* result
) {
    int lx, ly, rx, ry;
    
    // 1. Find object in left image
    int found_l = find_centroid(img_left, width, height, threshold, &lx, &ly);
    
    // 2. Find object in right image
    int found_r = find_centroid(img_right, width, height, threshold, &rx, &ry);

    if (found_l && found_r) {
        result->object_found = 1;
        result->left_x = lx;
        result->left_y = ly;
        result->right_x = rx;
        result->right_y = ry;

        // 3. Calculate Disparity
        // Disparity = (X_left - X_right)
        // Note: We assume cameras are rectified.
        // If object is at infinity, disparity is 0. The closer it is, the higher the disparity.
        double disparity = (double)(lx - rx);

        // Handle edge cases (negative or zero disparity)
        if (disparity <= 0.1) {
            disparity = 0.1; // Avoid division by zero, object very far away
        }
        
        result->disparity = disparity;

        // 4. Calculate Distance (Triangulation)
        // Z = (f * b) / d
        result->distance = (focal_length * baseline) / disparity;

    } else {
        result->object_found = 0;
        result->distance = -1.0;
        result->disparity = 0.0;
    }
}

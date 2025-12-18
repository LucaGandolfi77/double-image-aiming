# Stereo Aiming System (C + Python)

This project implements a hybrid stereo vision system. The computational "core" for image processing is written in **C** to maximize performance, while the interface, video acquisition, physics calculations (velocity, acceleration), and laser pointer control are managed in **Python**.

## Structure

- `src/stereo_core.c`: C library that analyzes pixels, finds the centroid of a dark object on a light background, and calculates disparity.
- `main.py`: Python script that loads the C library, simulates (or acquires) video streams, calculates movement physics, and controls the laser pointer.
- `Makefile`: Script to compile the shared library.

## Prerequisites

- GCC
- Python 3
- OpenCV for Python (`pip install opencv-python numpy`)

## Compilation

Before running the program, you must compile the C code into a shared library (`.so`).

```bash
make
```

This will generate the `libstereo.so` file.

## Execution

Run the Python script:

```bash
python3 main.py
```

## How it Works

1.  **Simulation**: Currently, the code uses a `MockCamera` class that generates two white images with a moving black dot, simulating an object approaching and receding.
2.  **C Processing**: Python passes pointers to the raw image buffers to the C function `process_stereo_frame`.
3.  **Triangulation**: C calculates the distance $Z$ using the formula:
    $$ Z = \frac{f \cdot b}{d} $$
    Where $b = 2.0m$ (baseline) and $d$ is the disparity in pixels.
4.  **Physics**: Python derives velocity and acceleration based on the change in distance over time.
5.  **Laser Control**: Python calculates the Yaw (left/right) and Pitch (up/down) angles required to point a laser at the target.

## Adapting to Real Cameras

To use real cameras, modify `main.py`:

1.  Replace `MockCamera` with `cv2.VideoCapture(0)` and `cv2.VideoCapture(1)`.
2.  Ensure you calibrate `FOCAL_LENGTH` based on your specific lenses.
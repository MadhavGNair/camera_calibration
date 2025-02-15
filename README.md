# Camera Calibration

A Python module to calibrate camera using chessboard images and visualize 3D axes and cube on detected chessboards. Supports both image files and live webcam input.

Author: Madhav Girish Nair ([madhavgirish02@gmail.com](mailto:madhavgirish02@gmail.com))

## Directory Structure

 camera_calibration/
├── images/
│ ├── run_1/ # Calibration images for first run
│ ├── run_2/ # Calibration images for second run
│ ├── run_3/ # Calibration images for third run
│ └── test/ # Test images
├── output/ # Calibration data and output images
│ ├── run_1.json
│ ├── run_2.json
│ └── run_3.json
├── camera_calibrator.py # Main implementation
├── requirements.txt # Python dependencies
└── README.md # This file

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Features

- Camera calibration using chessboard pattern
- Real-time visualization of 3D axes and cube on chessboard using webcam
- Support for both image files and live webcam input
- 3D visualization of camera positions relative to chessboard
- Automatic corner detection with manual fallback
- Save/load calibration parameters

## Usage

### 1. Camera Calibration (Offline Step)

To calibrate the camera using chessboard images:

1. Place your chessboard images in the `images/run_X` directory
2. Uncomment the OFFLINE STEP section in `camera_calibrator.py`
3. Run:

```bash
python camera_calibrator.py
```

### 2. Visualize Axes and Cube (Online Step)

#### Using Webcam

1. Set `USE_WEBCAM = True` in the main section of `camera_calibrator.py`
2. Run:

```bash
python camera_calibrator.py
```

3. Show a chessboard pattern to your webcam
4. Press 'q' to quit the webcam feed

#### Using Images

1. Set `USE_WEBCAM = False` in the main section of `camera_calibrator.py`
2. Place test images in `images/test/`
3. Run:

```bash
python camera_calibrator.py
```

### 3. Plot Camera Locations

To visualize the 3D positions of cameras relative to the chessboard:

1. Uncomment the `calibrator.plot_camera_locations(params_path)` line
2. Run the script


## Notes

- note the 'save' parameter in all functions before running the script
- the 'save' parameter is set to False by default
- change the square size in the main section of the script to the actual size of the chessboard squares
- plotting defaults to running for each run, if you want to plot for a single run, set the loop to run for a single run
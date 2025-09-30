# Thermal Camera Fusion for Raspberry Pi 5

Real-time fusion of thermal camera data (MLX90640) with Pi Camera video on Raspberry Pi 5.

## Hardware Requirements

- Raspberry Pi 5
- Adafruit MLX90640 Thermal Camera (https://www.adafruit.com/product/4469)
- Pi Camera (any compatible camera)
- Display connected to Pi 5

## Installation

1. Clone the repository:
```bash
git clone git@github.com:kdesch5000/thermal-camera-fusion.git
cd thermal-camera-fusion
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Enable I2C on your Raspberry Pi:
```bash
sudo raspi-config
# Navigate to Interface Options -> I2C -> Enable
```

4. Connect the MLX90640 to your Pi:
- VIN to 3.3V
- GND to GND
- SCL to SCL (GPIO 3)
- SDA to SDA (GPIO 2)

## Usage

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

2. Run the application:

**With real thermal camera:**
```bash
python3 thermal_camera_fusion.py
```

**Demo mode (without thermal camera):**
```bash
python3 thermal_camera_fusion.py --demo
```

Demo mode generates simulated thermal data with moving hot spots so you can test all features and calibration controls without the MLX90640 hardware.

**Additional options:**
```bash
python3 thermal_camera_fusion.py --help
# Options:
#   --demo          Run in demo mode with simulated thermal data
#   --width WIDTH   Display width (default: 1920)
#   --height HEIGHT Display height (default: 1080)
#   --alpha ALPHA   Initial fusion transparency (default: 0.5)
```

### Controls

**Basic Controls:**
- `q` or `ESC`: Quit the application
- `t`: Toggle thermal-only mode
- `+`/`-`: Adjust thermal overlay intensity (fusion alpha)

**Calibration Controls:**
- **Arrow keys**: Move thermal overlay (position adjustment)
- `[`/`]`: Scale thermal overlay smaller/larger
- `,`/`.`: Rotate thermal overlay left/right (5° increments)
- `c`: Toggle calibration grid (shows crosshairs and current settings)
- `s`: Save calibration to file

### Calibration Process

Since the thermal camera and regular camera are physically offset and have different fields of view, you'll need to calibrate the overlay alignment:

1. Press `c` to enable the calibration grid
2. Use **arrow keys** to move the thermal overlay to align with the camera view
3. Use `[`/`]` to adjust the scale if the thermal field of view doesn't match
4. Use `,`/`.` to rotate if the cameras are mounted at different angles
5. Press `s` to save your calibration settings
6. Press `c` again to hide the calibration grid

Calibration settings are saved to `thermal_calibration.json` and loaded automatically on startup.

## Features

- Real-time thermal camera overlay on regular camera feed
- Manual alignment controls for position, scale, and rotation
- Adjustable fusion alpha (transparency)
- Temperature scale with min/max display
- FPS counter
- Thermal-only mode
- Calibration grid with live parameter display
- Persistent calibration settings
- Color-coded temperature visualization using JET colormap

## Display Settings

Default resolution is 1920x1080. To change, modify the `ThermalCameraFusion` initialization:

```python
fusion = ThermalCameraFusion(display_width=1280, display_height=720)
```

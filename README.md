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

Run the application:
```bash
python3 thermal_camera_fusion.py
```

### Controls

- `q` or `ESC`: Quit the application
- `t`: Toggle thermal-only mode
- `+`: Increase thermal overlay intensity
- `-`: Decrease thermal overlay intensity

## Features

- Real-time thermal camera overlay on regular camera feed
- Adjustable fusion alpha (transparency)
- Temperature scale with min/max display
- FPS counter
- Thermal-only mode
- Color-coded temperature visualization using JET colormap

## Display Settings

Default resolution is 1920x1080. To change, modify the `ThermalCameraFusion` initialization:

```python
fusion = ThermalCameraFusion(display_width=1280, display_height=720)
```

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

**Wiring Connections:**
```
MLX90640        Raspberry Pi 5 GPIO
--------        -------------------
VIN      →      Pin 1  (3.3V)
SDA      →      Pin 3  (GPIO 2 / SDA)
SCL      →      Pin 5  (GPIO 3 / SCL)
GND      →      Pin 6  (GND)
```

**Important Notes:**
- Use 3.3V power (NOT 5V) - Pin 1 or Pin 17
- SDA connects to GPIO 2 (Pin 3)
- SCL connects to GPIO 3 (Pin 5)
- Any ground pin works (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
- No external pull-up resistors needed (Pi has built-in I2C pull-ups)

See [wiring_diagram.txt](wiring_diagram.txt) for detailed pinout diagram.

5. Verify the thermal camera is detected:
```bash
i2cdetect -y 1
```
You should see device at address `0x33`

## Usage

1. Create and activate a virtual environment:
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip3 install -r requirements.txt
```

**Note:** The script will auto-activate the virtual environment if it exists, so you can also run it directly:
```bash
python3 thermal_camera_fusion.py --demo --width 800 --height 480
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

**Touchscreen Controls:**

The application includes an on-screen button interface for touchscreen displays:

**Top Row:**
- **Fusion** button: Switch to fusion mode (camera + thermal overlay)
- **Thermal** button: Switch to thermal-only mode
- **Video** button: Switch to video-only mode (no thermal)
- **Snapshot** button: Take a JPEG snapshot with timestamp
- **Record/Stop** button: Start/stop MP4 video recording
- **Gallery** button: Browse saved snapshots and videos
- **Settings** button: Toggle settings panel
- **Calibrate** button: Toggle calibration grid and controls

**Bottom Row (visible when Calibrate is active):**
- **← → ↑ ↓** buttons: Move thermal overlay position
- **[ ]** buttons: Scale thermal overlay smaller/larger
- **, .** buttons: Rotate thermal overlay left/right
- **Save** button: Save calibration to file

**Keyboard Controls (alternative):**
- `q` or `ESC`: Quit the application
- `t`: Toggle thermal-only mode
- `v`: Toggle video-only mode
- `+`/`-`: Adjust thermal overlay intensity (fusion alpha)
- **Arrow keys**: Move thermal overlay (position adjustment)
- `[`/`]`: Scale thermal overlay smaller/larger
- `,`/`.`: Rotate thermal overlay left/right (5° increments)
- `c`: Toggle calibration grid (shows crosshairs and current settings)
- `s`: Save calibration to file
- `p`: Take snapshot (saves JPEG with timestamp)
- `r`: Start/stop recording (saves MP4 with timestamp)
- `m`: Toggle settings menu
- `h`: Toggle temperature histogram
- `k`: Cycle through colormaps (JET, HOT, COOL, RAINBOW, BONE, TURBO)
- `i`: Toggle frame interpolation (smooths thermal updates)
- `g`: Open/close gallery
- **Gallery navigation**: Arrow keys (left/right), Delete key to remove files

### Calibration Process

Since the thermal camera and regular camera are physically offset and have different fields of view, you'll need to calibrate the overlay alignment:

**Using Touchscreen:**
1. Tap the **Calibrate** button to show the calibration grid and controls
2. Use the **← → ↑ ↓** buttons to move the thermal overlay to align with the camera view
3. Use **[ ]** buttons to adjust the scale if the thermal field of view doesn't match
4. Use **, .** buttons to rotate if the cameras are mounted at different angles
5. Tap **Save** to save your calibration settings
6. Tap **Calibrate** again to hide the calibration controls

**Using Keyboard:**
1. Press `c` to enable the calibration grid
2. Use **arrow keys** to move the thermal overlay to align with the camera view
3. Use `[`/`]` to adjust the scale if the thermal field of view doesn't match
4. Use `,`/`.` to rotate if the cameras are mounted at different angles
5. Press `s` to save your calibration settings
6. Press `c` again to hide the calibration grid

Calibration settings are saved to `thermal_calibration.json` and loaded automatically on startup.

### Recording and Snapshots

**Taking Snapshots:**
- Tap the **Snapshot** button or press `p` to capture the current display
- Snapshots are saved as JPEG files with timestamp filenames
- Format: `thermal_snapshot_YYYYMMDD_HHMMSS.jpg`
- Files are saved in the current working directory

**Recording Videos:**
- Tap the **Record** button or press `r` to start recording
- Recording indicator appears in top-right corner with elapsed time
- Tap **Stop** button or press `r` again to stop recording
- Videos are saved as MP4 files for cross-platform compatibility
- Format: `thermal_video_YYYYMMDD_HHMMSS.mp4`
- Compatible with iPhone, Windows 11, macOS, and other platforms
- Records the full thermal overlay as displayed (including temperature scale and calibration grid if visible)
- Files are saved in the current working directory

**Note:** Recorded videos and snapshots do NOT include the on-screen buttons, but they do include the temperature overlay, calibration grid (if visible), and recording indicator.

### Image Gallery

Browse, view, and manage your saved thermal snapshots and videos without leaving the application:

**Features:**
- View all saved snapshots (`.jpg`) and video thumbnails (`.mp4`)
- Files sorted by date (newest first)
- Navigate with arrow keys or swipe gestures
- Delete unwanted files instantly
- Shows file information (name, size, date)
- Automatically rotates snapshots for correct orientation

**Usage:**
- Tap **Gallery** button or press `g` to open
- Use **Left/Right arrow keys** to navigate between files
- Press **Delete** key to remove the current file
- Press `g` again to close gallery
- Gallery displays the first frame of videos as thumbnails

**Display:**
- Images are scaled to fit 80% of screen while maintaining aspect ratio
- Semi-transparent dark background for better visibility
- White border around images
- Info bar shows: file index, filename, size, and timestamp

### Settings Menu

Access the settings panel by tapping the **Settings** button or pressing `m`:

**Available Settings:**
- **Colormap Selection**: Cycle through different thermal colormaps
  - JET (blue-red, default)
  - HOT (black-red-yellow-white)
  - COOL (cyan-magenta)
  - RAINBOW (full spectrum)
  - BONE (grayscale-blue)
  - TURBO (perceptually uniform)
- **Recording FPS**: Current video recording frame rate
- **Histogram Toggle**: Enable/disable temperature distribution display
- **Frame Interpolation**: Smooth thermal updates between 8Hz sensor readings

Press `k` to quickly cycle through colormaps, `h` to toggle the histogram, or `i` to toggle interpolation without opening settings.

### Temperature Histogram

Enable the histogram (press `h` or via Settings menu) to see:
- Live temperature distribution across the thermal image
- Color-coded bars matching your selected colormap
- Useful for identifying temperature anomalies and patterns
- Automatically hidden in video-only mode

### Frame Interpolation

The MLX90640 thermal sensor updates at 8Hz (~125ms between frames), while the display may run at higher frame rates. Frame interpolation smooths thermal data between sensor readings:

**Benefits:**
- Eliminates jerky thermal overlay motion
- Creates smoother temperature transitions
- Better visual experience for moving heat sources
- No impact on actual temperature accuracy

**How it works:**
- Linearly interpolates between consecutive thermal frames based on elapsed time
- Automatically syncs with the 8Hz sensor refresh rate
- Toggle on/off with `i` key if you prefer raw sensor updates
- Enabled by default for smoother visualization

## Features

- Real-time thermal camera overlay on regular camera feed
- Three display modes: Fusion, Thermal-only, and Video-only
- Touchscreen button interface for easy control
- **Video recording and snapshot capture** with timestamped filenames
- **Image gallery** to browse and manage saved files
- **Cross-platform MP4 video format** compatible with all devices
- **Settings menu** with configurable options
- **6 thermal colormaps** (JET, HOT, COOL, RAINBOW, BONE, TURBO)
- **Temperature histogram** for distribution analysis
- **Frame interpolation** for smooth 8Hz to display fps conversion
- Manual alignment controls for position, scale, and rotation
- Adjustable fusion alpha (transparency)
- Temperature scale with min/max display
- FPS counter
- Calibration grid with live parameter display
- Persistent calibration settings
- Demo mode for testing without hardware

## Display Settings

Default resolution is 1920x1080. To change, modify the `ThermalCameraFusion` initialization:

```python
fusion = ThermalCameraFusion(display_width=1280, display_height=720)
```

#!/usr/bin/env python3
"""
Thermal Camera Fusion for Raspberry Pi 5
Fuses MLX90640 thermal camera data with Pi Camera video
"""

import sys
import os

# Auto-activate virtual environment if not already activated
if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    if os.path.exists(venv_path):
        activate_script = os.path.join(venv_path, 'bin', 'activate_this.py')
        # For newer Python versions without activate_this.py, we need to adjust sys.path
        if not os.path.exists(activate_script):
            site_packages = os.path.join(venv_path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
            if os.path.exists(site_packages):
                sys.path.insert(0, site_packages)
        else:
            exec(open(activate_script).read(), {'__file__': activate_script})

import time
import numpy as np
import cv2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import pygame
from PIL import Image
import json
import argparse
from datetime import datetime
import glob
import os

# Force SDL to position window at specific location if needed
# This helps with multi-monitor setups
if 'SDL_VIDEO_WINDOW_POS' not in os.environ:
    os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'


class ThermalCameraFusion:
    def __init__(self, display_width=1920, display_height=1080, config_file="thermal_calibration.json", demo_mode=False, windowed=False):
        """Initialize thermal camera, regular camera, and display"""
        self.display_width = display_width
        self.display_height = display_height
        self.config_file = config_file
        self.demo_mode = demo_mode
        self.windowed = windowed

        # Thermal overlay calibration parameters
        self.thermal_offset_x = 0
        self.thermal_offset_y = 0
        self.thermal_scale = 1.0
        self.thermal_rotation = 0.0

        # Load calibration if exists
        self.load_calibration()

        # Initialize thermal camera (MLX90640) or demo mode
        if demo_mode:
            print("Running in DEMO mode with simulated thermal data...")
            self.mlx = None
            self.thermal_frame = np.zeros((24 * 32,))
            self.demo_time = 0
        else:
            print("Initializing MLX90640 thermal camera...")
            import board
            import busio
            import adafruit_mlx90640
            i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.mlx = adafruit_mlx90640.MLX90640(i2c)
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
            # Thermal image is 32x24 pixels
            self.thermal_frame = np.zeros((24 * 32,))

        # Initialize Pi Camera
        print("Initializing Pi Camera...")
        self.picam = Picamera2()
        config = self.picam.create_preview_configuration(
            main={"size": (display_width, display_height), "format": "RGB888"}
        )
        self.picam.configure(config)
        self.picam.start()

        # Give camera time to warm up
        time.sleep(2)

        # Initialize pygame for display
        print("Initializing display...")
        pygame.init()
        if windowed:
            self.screen = pygame.display.set_mode((display_width, display_height))
        else:
            self.screen = pygame.display.set_mode((display_width, display_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Thermal Camera Fusion")

        # Color map for thermal data (iron colormap approximation)
        self.colormap = cv2.COLORMAP_JET
        self.colormap_name = "JET"
        self.available_colormaps = {
            "JET": cv2.COLORMAP_JET,
            "HOT": cv2.COLORMAP_HOT,
            "COOL": cv2.COLORMAP_COOL,
            "RAINBOW": cv2.COLORMAP_RAINBOW,
            "BONE": cv2.COLORMAP_BONE,
            "TURBO": cv2.COLORMAP_TURBO,
        }

        # Recording state
        self.recording = False
        self.video_writer = None
        self.recording_filename = None
        self.recording_start_time = None
        self.recording_fps = 10.0

        # Settings state
        self.show_histogram = False
        self.show_settings = False
        self.show_gallery = False

        # Gallery state
        self.gallery_files = []
        self.gallery_index = 0
        self.gallery_image = None

        # Frame interpolation state
        self.enable_interpolation = True
        self.previous_thermal_frame = None
        self.last_thermal_capture_time = time.time()
        self.interpolation_alpha = 0.0

    def read_thermal_frame(self):
        """Read frame from thermal camera or generate demo data"""
        if self.demo_mode:
            # Generate simulated thermal data with moving hot spots
            thermal_image = np.zeros((24, 32))

            # Create base temperature (room temp around 20-25C)
            thermal_image += 22 + np.random.normal(0, 0.5, (24, 32))

            # Add moving hot spot 1 (simulates person or heat source)
            x1 = int(16 + 10 * np.sin(self.demo_time * 0.5))
            y1 = int(12 + 6 * np.cos(self.demo_time * 0.3))
            for i in range(24):
                for j in range(32):
                    dist = np.sqrt((i - y1)**2 + (j - x1)**2)
                    if dist < 8:
                        thermal_image[i, j] += 10 * np.exp(-dist / 3)

            # Add moving hot spot 2
            x2 = int(16 + 8 * np.cos(self.demo_time * 0.4 + 1))
            y2 = int(12 + 5 * np.sin(self.demo_time * 0.6 + 2))
            for i in range(24):
                for j in range(32):
                    dist = np.sqrt((i - y2)**2 + (j - x2)**2)
                    if dist < 5:
                        thermal_image[i, j] += 6 * np.exp(-dist / 2)

            self.demo_time += 0.1
            return thermal_image
        else:
            # Real thermal camera
            try:
                self.mlx.getFrame(self.thermal_frame)
                # Reshape to 24x32 array
                thermal_image = np.reshape(self.thermal_frame, (24, 32))
                return thermal_image
            except ValueError:
                # Frame not ready, return previous frame
                return np.reshape(self.thermal_frame, (24, 32))

    def interpolate_thermal_frame(self, current_frame):
        """Interpolate between thermal frames for smoother animation"""
        if not self.enable_interpolation:
            return current_frame

        # Check if we have a previous frame to interpolate with
        if self.previous_thermal_frame is None:
            self.previous_thermal_frame = current_frame.copy()
            self.last_thermal_capture_time = time.time()
            return current_frame

        # Calculate time-based interpolation factor
        # MLX90640 updates at ~8Hz (125ms between frames)
        current_time = time.time()
        time_since_capture = current_time - self.last_thermal_capture_time
        expected_frame_time = 0.125  # 8Hz = 125ms

        # Check if we should get a new thermal frame
        if time_since_capture >= expected_frame_time:
            # New frame available, update tracking
            self.previous_thermal_frame = current_frame.copy()
            self.last_thermal_capture_time = current_time
            self.interpolation_alpha = 0.0
            return current_frame
        else:
            # Interpolate between previous and current frame
            self.interpolation_alpha = time_since_capture / expected_frame_time
            # Smooth interpolation using linear blend
            interpolated = (1 - self.interpolation_alpha) * self.previous_thermal_frame + \
                          self.interpolation_alpha * current_frame
            return interpolated

    def process_thermal_image(self, thermal_data):
        """Convert thermal data to colored image with calibration transforms"""
        # Normalize to 0-255 range
        min_temp = np.min(thermal_data)
        max_temp = np.max(thermal_data)

        if max_temp - min_temp > 0:
            normalized = ((thermal_data - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(thermal_data, dtype=np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(normalized, self.colormap)

        # Calculate scaled size
        scaled_width = int(self.display_width * self.thermal_scale)
        scaled_height = int(self.display_height * self.thermal_scale)

        # Resize to scaled size
        resized = cv2.resize(colored, (scaled_width, scaled_height),
                           interpolation=cv2.INTER_CUBIC)

        # Apply rotation if needed
        if self.thermal_rotation != 0:
            center = (scaled_width // 2, scaled_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.thermal_rotation, 1.0)
            resized = cv2.warpAffine(resized, rotation_matrix, (scaled_width, scaled_height))

        return resized, min_temp, max_temp

    def fuse_images(self, camera_frame, thermal_frame, alpha=0.5):
        """Fuse camera and thermal images with transparency and position offset"""
        # Ensure both images are RGB
        if len(camera_frame.shape) == 2:
            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_GRAY2RGB)
        if len(thermal_frame.shape) == 2:
            thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2RGB)

        # Create a blank canvas the size of the camera frame
        thermal_canvas = np.zeros_like(camera_frame)

        # Calculate position with offset
        th, tw = thermal_frame.shape[:2]
        ch, cw = camera_frame.shape[:2]

        # Center position with offset
        x_pos = (cw - tw) // 2 + self.thermal_offset_x
        y_pos = (ch - th) // 2 + self.thermal_offset_y

        # Calculate valid region to paste (handle overflow)
        src_x1 = max(0, -x_pos)
        src_y1 = max(0, -y_pos)
        src_x2 = tw - max(0, (x_pos + tw) - cw)
        src_y2 = th - max(0, (y_pos + th) - ch)

        dst_x1 = max(0, x_pos)
        dst_y1 = max(0, y_pos)
        dst_x2 = min(cw, x_pos + tw)
        dst_y2 = min(ch, y_pos + th)

        # Place thermal frame on canvas at offset position
        if src_x2 > src_x1 and src_y2 > src_y1:
            thermal_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = thermal_frame[src_y1:src_y2, src_x1:src_x2]

        # Blend images
        fused = cv2.addWeighted(camera_frame, 1 - alpha, thermal_canvas, alpha, 0)
        return fused

    def draw_temperature_overlay(self, frame, min_temp, max_temp):
        """Draw temperature scale and info on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw temperature range
        text = f"Min: {min_temp:.1f}C  Max: {max_temp:.1f}C"
        cv2.putText(frame, text, (20, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw color scale bar
        scale_height = 300
        scale_width = 30
        scale_x = self.display_width - 60
        scale_y = 100

        # Create gradient bar
        gradient = np.linspace(255, 0, scale_height, dtype=np.uint8).reshape(-1, 1)
        gradient = np.repeat(gradient, scale_width, axis=1)
        gradient_colored = cv2.applyColorMap(gradient, self.colormap)

        # Place gradient on frame
        frame[scale_y:scale_y+scale_height, scale_x:scale_x+scale_width] = gradient_colored

        # Draw border around scale
        cv2.rectangle(frame, (scale_x, scale_y), (scale_x+scale_width, scale_y+scale_height),
                     (255, 255, 255), 2)

        # Add temperature labels
        cv2.putText(frame, f"{max_temp:.1f}C", (scale_x-80, scale_y+20),
                   font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{min_temp:.1f}C", (scale_x-80, scale_y+scale_height),
                   font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def draw_calibration_grid(self, frame):
        """Draw calibration grid and crosshairs"""
        h, w = frame.shape[:2]
        color = (0, 255, 0)

        # Draw center crosshairs
        cv2.line(frame, (w//2, 0), (w//2, h), color, 2)
        cv2.line(frame, (0, h//2), (w, h//2), color, 2)

        # Draw grid
        for i in range(1, 4):
            x = w * i // 4
            y = h * i // 4
            cv2.line(frame, (x, 0), (x, h), color, 1)
            cv2.line(frame, (0, y), (w, y), color, 1)

        # Draw current calibration values
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = [
            f"Offset X: {self.thermal_offset_x}px",
            f"Offset Y: {self.thermal_offset_y}px",
            f"Scale: {self.thermal_scale:.2f}x",
            f"Rotation: {self.thermal_rotation:.1f}deg",
        ]

        y_pos = h - 150
        for text in info_text:
            cv2.putText(frame, text, (20, y_pos), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_pos += 30

        return frame

    def take_snapshot(self, frame):
        """Save current frame as JPEG with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"thermal_snapshot_{timestamp}.jpg"
        # Rotate 90 degrees clockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(filename, rotated_frame)
        print(f"Snapshot saved: {filename}")
        return filename

    def cycle_colormap(self):
        """Cycle to next colormap"""
        colormap_names = list(self.available_colormaps.keys())
        current_index = colormap_names.index(self.colormap_name)
        next_index = (current_index + 1) % len(colormap_names)
        self.colormap_name = colormap_names[next_index]
        self.colormap = self.available_colormaps[self.colormap_name]
        print(f"Colormap: {self.colormap_name}")

    def start_recording(self, frame_width, frame_height):
        """Start video recording with software encoding (thermal overlay)"""
        # Note: We use software encoding to capture the thermal overlay
        # Hardware encoding would only record the raw camera feed without overlay
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"thermal_video_{timestamp}.mp4"

        # Try different codecs in order of preference
        codecs_to_try = [
            ('mp4v', 'MPEG-4'),
            ('XVID', 'Xvid'),
            ('MJPG', 'Motion JPEG'),
        ]

        for fourcc_str, codec_name in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            self.video_writer = cv2.VideoWriter(self.recording_filename, fourcc,
                                               self.recording_fps, (frame_width, frame_height))

            if self.video_writer.isOpened():
                self.recording = True
                self.recording_start_time = time.time()
                print(f"Recording started with {codec_name} encoding: {self.recording_filename}")
                return self.recording_filename
            else:
                self.video_writer = None

        print("Failed to start recording - no compatible codec found")
        return None

    def stop_recording(self):
        """Stop video recording"""
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0

        # Stop OpenCV writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"Recording stopped. Duration: {duration:.1f}s")

        self.recording = False

    def write_video_frame(self, frame):
        """Write frame to video file if recording"""
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def draw_histogram(self, frame, thermal_data, min_temp, max_temp):
        """Draw temperature histogram"""
        h, w = frame.shape[:2]

        # Histogram parameters
        hist_width = min(300, int(w * 0.25))
        hist_height = min(200, int(h * 0.20))
        hist_x = 20
        hist_y = h - hist_height - 80

        # Create histogram
        bins = 50
        hist, bin_edges = np.histogram(thermal_data.flatten(), bins=bins, range=(min_temp, max_temp))

        # Normalize histogram
        if hist.max() > 0:
            hist_normalized = (hist / hist.max() * (hist_height - 20)).astype(int)
        else:
            hist_normalized = np.zeros_like(hist, dtype=int)

        # Draw histogram background
        cv2.rectangle(frame, (hist_x, hist_y), (hist_x + hist_width, hist_y + hist_height),
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (hist_x, hist_y), (hist_x + hist_width, hist_y + hist_height),
                     (200, 200, 200), 2)

        # Draw bars
        bar_width = hist_width / bins
        for i, count in enumerate(hist_normalized):
            if count > 0:
                x1 = int(hist_x + i * bar_width)
                x2 = int(hist_x + (i + 1) * bar_width)
                y1 = hist_y + hist_height - 10
                y2 = y1 - count

                # Color bars based on temperature range
                temp_ratio = i / bins
                color_value = int(temp_ratio * 255)
                bar_color = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), self.colormap)[0][0]
                color = (int(bar_color[0]), int(bar_color[1]), int(bar_color[2]))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Draw title
        cv2.putText(frame, "Temperature Distribution", (hist_x + 5, hist_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def load_gallery_files(self):
        """Load list of saved snapshots and videos"""
        # Find all thermal snapshots and videos in current directory
        snapshots = glob.glob("thermal_snapshot_*.jpg")
        videos = glob.glob("thermal_video_*.mp4")

        # Combine and sort by modification time (newest first)
        all_files = [(f, os.path.getmtime(f)) for f in snapshots + videos]
        all_files.sort(key=lambda x: x[1], reverse=True)

        self.gallery_files = [f[0] for f in all_files]
        self.gallery_index = 0

        if self.gallery_files:
            self.load_gallery_image(self.gallery_files[self.gallery_index])

        return len(self.gallery_files)

    def load_gallery_image(self, filename):
        """Load an image for gallery display"""
        try:
            if filename.endswith('.jpg'):
                # Load snapshot (already rotated 90 degrees clockwise when saved)
                self.gallery_image = cv2.imread(filename)
                if self.gallery_image is not None:
                    # Flip horizontally (mirror image)
                    self.gallery_image = cv2.flip(self.gallery_image, 1)
            elif filename.endswith('.mp4'):
                # Load first frame of video as thumbnail
                cap = cv2.VideoCapture(filename)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.gallery_image = frame
                else:
                    self.gallery_image = None
        except Exception as e:
            print(f"Error loading gallery file {filename}: {e}")
            self.gallery_image = None

    def gallery_next(self):
        """Navigate to next image in gallery"""
        if self.gallery_files and self.gallery_index < len(self.gallery_files) - 1:
            self.gallery_index += 1
            self.load_gallery_image(self.gallery_files[self.gallery_index])

    def gallery_previous(self):
        """Navigate to previous image in gallery"""
        if self.gallery_files and self.gallery_index > 0:
            self.gallery_index -= 1
            self.load_gallery_image(self.gallery_files[self.gallery_index])

    def delete_current_gallery_file(self):
        """Delete currently displayed gallery file"""
        if self.gallery_files and 0 <= self.gallery_index < len(self.gallery_files):
            filename = self.gallery_files[self.gallery_index]
            try:
                os.remove(filename)
                print(f"Deleted: {filename}")

                # Remove from list
                del self.gallery_files[self.gallery_index]

                # Adjust index and load next/previous image
                if self.gallery_files:
                    if self.gallery_index >= len(self.gallery_files):
                        self.gallery_index = len(self.gallery_files) - 1
                    self.load_gallery_image(self.gallery_files[self.gallery_index])
                else:
                    self.gallery_image = None
                    self.show_gallery = False
                    print("Gallery is empty")
            except Exception as e:
                print(f"Error deleting file: {e}")

    def draw_gallery(self, frame):
        """Draw gallery overlay"""
        if not self.gallery_image is None:
            h, w = frame.shape[:2]

            # Scale gallery image to fit display while maintaining aspect ratio
            gh, gw = self.gallery_image.shape[:2]
            scale = min(w / gw, h / gh) * 0.8  # Use 80% of screen

            new_w = int(gw * scale)
            new_h = int(gh * scale)

            gallery_resized = cv2.resize(self.gallery_image, (new_w, new_h))

            # Center the image
            x_offset = (w - new_w) // 2
            y_offset = (h - new_h) // 2

            # Draw semi-transparent black background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            # Place gallery image
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = gallery_resized

            # Draw border around image
            cv2.rectangle(frame, (x_offset-2, y_offset-2),
                         (x_offset+new_w+2, y_offset+new_h+2), (255, 255, 255), 2)

        # Draw filename and navigation info
        if self.gallery_files:
            filename = self.gallery_files[self.gallery_index]
            info_text = f"{self.gallery_index + 1}/{len(self.gallery_files)}: {filename}"

            # Get file size and date
            file_size = os.path.getsize(filename) / 1024  # KB
            file_time = datetime.fromtimestamp(os.path.getmtime(filename)).strftime("%Y-%m-%d %H:%M:%S")
            size_text = f"{file_size:.1f} KB - {file_time}"

            # Draw info bar at bottom
            cv2.rectangle(frame, (0, h-80), (w, h), (40, 40, 40), -1)
            cv2.putText(frame, info_text, (20, h-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, size_text, (20, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw controls on right side
            controls_text = [
                "Arrow Left/Right: Navigate",
                "Delete: Remove file",
                "G: Close gallery"
            ]
            y_pos = h - 70
            for text in controls_text:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(frame, text, (w - text_size[0] - 20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 20
        else:
            # No files in gallery
            text = "No saved images or videos found"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(frame, text, ((w - text_size[0]) // 2, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def draw_settings_panel(self, frame):
        """Draw settings panel overlay"""
        h, w = frame.shape[:2]

        # Settings text to display
        settings_text = [
            "SETTINGS",
            f"Colormap: {self.colormap_name}",
            f"Recording FPS: {self.recording_fps:.0f}",
            f"Histogram: {'ON' if self.show_histogram else 'OFF'}",
            f"Interpolation: {'ON' if self.enable_interpolation else 'OFF'}",
            "",
            "Press 'h' to toggle histogram",
            "Press 'k' to cycle colormap",
            "Press 'i' to toggle interpolation",
            "Press 'm' to close settings",
        ]

        # Calculate panel size based on content
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 22  # Reduced from 35 for tighter spacing
        padding = 15

        # Calculate required width (find longest line)
        max_width = 0
        for text in settings_text:
            text_size = cv2.getTextSize(text, font, font_scale if text != "SETTINGS" else 0.7,
                                       font_thickness if text != "SETTINGS" else 2)[0]
            max_width = max(max_width, text_size[0])

        # Panel dimensions with padding
        panel_width = max_width + (padding * 2)
        panel_height = len(settings_text) * line_height + padding * 2
        panel_x = w - panel_width - 20
        panel_y = 100

        # Draw panel background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (200, 200, 200), 2)

        # Draw text
        y_pos = panel_y + padding + 15
        for i, text in enumerate(settings_text):
            if text == "SETTINGS":
                # Title with larger font
                cv2.putText(frame, text, (panel_x + padding, y_pos),
                           font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            elif text:  # Skip empty lines for spacing
                cv2.putText(frame, text, (panel_x + padding, y_pos),
                           font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y_pos += line_height

        return frame

    def draw_touch_buttons(self, frame, show_thermal_only, show_video_only, show_calibration):
        """Draw touchscreen buttons overlay"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Scale button sizes based on display size
        btn_height = max(40, int(h * 0.08))
        btn_width = max(80, int(w * 0.08))
        margin = max(5, int(w * 0.01))

        # Calculate button width for calibration buttons
        cal_btn_width = max(50, int(w * 0.06))

        # Define buttons (x, y, width, height, label, active)
        buttons = [
            # Mode buttons (top row)
            (margin, margin, btn_width, btn_height, "Fusion", not show_thermal_only and not show_video_only),
            (margin + btn_width + margin, margin, btn_width, btn_height, "Thermal", show_thermal_only),
            (margin + (btn_width + margin) * 2, margin, btn_width, btn_height, "Video", show_video_only),

            # Recording buttons
            (margin + (btn_width + margin) * 3, margin, btn_width, btn_height, "Snapshot", False),
            (margin + (btn_width + margin) * 4, margin, btn_width, btn_height, "Record" if not self.recording else "Stop", self.recording),

            # Gallery, Settings and Calibration buttons
            (w - (btn_width + margin) * 3, margin, btn_width, btn_height, "Gallery", self.show_gallery),
            (w - (btn_width + margin) * 2, margin, btn_width, btn_height, "Settings", self.show_settings),
            (w - btn_width - margin, margin, btn_width, btn_height, "Calibrate", show_calibration),

            # Control buttons (bottom row) - only show in calibration mode
        ]

        # Add calibration control buttons if in calibration mode
        if show_calibration:
            # Calculate positions for calibration buttons
            btn_spacing = cal_btn_width + margin
            cal_buttons = [
                (margin, h - btn_height - margin, cal_btn_width, btn_height, "LEFT", False),
                (margin + btn_spacing, h - btn_height - margin, cal_btn_width, btn_height, "RIGHT", False),
                (margin + btn_spacing * 2, h - btn_height - margin, cal_btn_width, btn_height, "UP", False),
                (margin + btn_spacing * 3, h - btn_height - margin, cal_btn_width, btn_height, "DOWN", False),
                (margin + btn_spacing * 4, h - btn_height - margin, cal_btn_width, btn_height, "SCALE-", False),
                (margin + btn_spacing * 5, h - btn_height - margin, cal_btn_width, btn_height, "SCALE+", False),
                (margin + btn_spacing * 6, h - btn_height - margin, cal_btn_width, btn_height, "ROT-", False),
                (margin + btn_spacing * 7, h - btn_height - margin, cal_btn_width, btn_height, "ROT+", False),
                (w - btn_width - margin, h - btn_height - margin, btn_width, btn_height, "Save", False),
            ]
            buttons.extend(cal_buttons)

        # Draw buttons
        for btn in buttons:
            x, y, bw, bh, label, active = btn
            # Button background
            color = (0, 180, 0) if active else (60, 60, 60)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, -1)
            # Button border
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 2)
            # Button text
            text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
            text_x = x + (bw - text_size[0]) // 2
            text_y = y + (bh + text_size[1]) // 2
            cv2.putText(frame, label, (text_x, text_y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return frame, buttons

    def save_calibration(self):
        """Save calibration parameters to file"""
        calibration = {
            "thermal_offset_x": self.thermal_offset_x,
            "thermal_offset_y": self.thermal_offset_y,
            "thermal_scale": self.thermal_scale,
            "thermal_rotation": self.thermal_rotation,
        }
        with open(self.config_file, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"Calibration saved to {self.config_file}")

    def load_calibration(self):
        """Load calibration parameters from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    calibration = json.load(f)
                self.thermal_offset_x = calibration.get("thermal_offset_x", 0)
                self.thermal_offset_y = calibration.get("thermal_offset_y", 0)
                self.thermal_scale = calibration.get("thermal_scale", 1.0)
                self.thermal_rotation = calibration.get("thermal_rotation", 0.0)
                print(f"Calibration loaded from {self.config_file}")
            except Exception as e:
                print(f"Error loading calibration: {e}")

    def run(self, fusion_alpha=0.5, show_thermal_only=False):
        """Main loop to capture and display fused images"""
        print("Starting thermal camera fusion...")
        print("Controls:")
        print("  q/ESC: Quit")
        print("  t: Toggle thermal-only mode")
        print("  v: Toggle video-only mode")
        print("  +/-: Adjust fusion alpha")
        print("  Arrow keys: Move thermal overlay")
        print("  [/]: Scale thermal overlay")
        print("  ,/.: Rotate thermal overlay")
        print("  c: Toggle calibration grid")
        print("  s: Save calibration")
        print("  p: Take snapshot")
        print("  r: Toggle recording")
        print("  m: Toggle settings menu")
        print("  h: Toggle histogram")
        print("  k: Cycle colormap")
        print("  i: Toggle frame interpolation")
        print("  g: Toggle gallery")
        print("  In gallery: Arrow keys navigate, Delete removes file")

        running = True
        fps_time = time.time()
        frame_count = 0
        show_calibration = False
        show_video_only = False
        move_step = 10
        buttons = []
        last_touch_time = 0
        touch_debounce = 0.3  # 300ms debounce

        try:
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.FINGERDOWN:
                        # Debounce touch events to prevent double-triggering
                        current_time = time.time()
                        if current_time - last_touch_time < touch_debounce:
                            continue
                        last_touch_time = current_time

                        # Handle touch/click events
                        if event.type == pygame.FINGERDOWN:
                            # Convert normalized touch coordinates (0-1) to pixel coordinates
                            # In fullscreen, get the actual window size which may differ from display_width/height
                            window_size = self.screen.get_size()
                            mouse_x = int(event.x * window_size[0])
                            mouse_y = int(event.y * window_size[1])
                        else:
                            mouse_x, mouse_y = event.pos

                        for btn in buttons:
                            x, y, bw, bh, label, active = btn
                            if x <= mouse_x <= x + bw and y <= mouse_y <= y + bh:
                                # Button clicked
                                if label == "Fusion":
                                    show_thermal_only = False
                                    show_video_only = False
                                    print("Fusion mode")
                                elif label == "Thermal":
                                    show_thermal_only = True
                                    show_video_only = False
                                    print("Thermal-only mode")
                                elif label == "Video":
                                    show_video_only = True
                                    show_thermal_only = False
                                    print("Video-only mode")
                                elif label == "Snapshot":
                                    # Take snapshot of current display
                                    # We'll capture the frame before buttons are drawn
                                    snapshot_frame = display_frame.copy()
                                    self.take_snapshot(snapshot_frame)
                                elif label == "Record":
                                    # Start recording
                                    if not self.recording:
                                        self.start_recording(self.display_width, self.display_height)
                                elif label == "Stop":
                                    # Stop recording
                                    if self.recording:
                                        self.stop_recording()
                                elif label == "Gallery":
                                    self.show_gallery = not self.show_gallery
                                    if self.show_gallery:
                                        count = self.load_gallery_files()
                                        print(f"Gallery opened: {count} files")
                                    else:
                                        print("Gallery closed")
                                elif label == "Settings":
                                    self.show_settings = not self.show_settings
                                    print(f"Settings panel: {self.show_settings}")
                                elif label == "Calibrate":
                                    show_calibration = not show_calibration
                                    print(f"Calibration grid: {show_calibration}")
                                elif label == "LEFT":
                                    self.thermal_offset_x -= move_step
                                    print(f"Offset X: {self.thermal_offset_x}")
                                elif label == "RIGHT":
                                    self.thermal_offset_x += move_step
                                    print(f"Offset X: {self.thermal_offset_x}")
                                elif label == "UP":
                                    self.thermal_offset_y -= move_step
                                    print(f"Offset Y: {self.thermal_offset_y}")
                                elif label == "DOWN":
                                    self.thermal_offset_y += move_step
                                    print(f"Offset Y: {self.thermal_offset_y}")
                                elif label == "SCALE-":
                                    self.thermal_scale = max(0.1, self.thermal_scale - 0.1)
                                    print(f"Scale: {self.thermal_scale:.2f}x")
                                elif label == "SCALE+":
                                    self.thermal_scale = min(3.0, self.thermal_scale + 0.1)
                                    print(f"Scale: {self.thermal_scale:.2f}x")
                                elif label == "ROT-":
                                    self.thermal_rotation -= 5.0
                                    print(f"Rotation: {self.thermal_rotation:.1f}deg")
                                elif label == "ROT+":
                                    self.thermal_rotation += 5.0
                                    print(f"Rotation: {self.thermal_rotation:.1f}deg")
                                elif label == "Save":
                                    self.save_calibration()
                                break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_t:
                            show_thermal_only = not show_thermal_only
                            show_video_only = False
                            print(f"Thermal-only mode: {show_thermal_only}")
                        elif event.key == pygame.K_v:
                            show_video_only = not show_video_only
                            show_thermal_only = False
                            print(f"Video-only mode: {show_video_only}")
                        elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                            fusion_alpha = min(1.0, fusion_alpha + 0.1)
                            print(f"Fusion alpha: {fusion_alpha:.1f}")
                        elif event.key == pygame.K_MINUS:
                            fusion_alpha = max(0.0, fusion_alpha - 0.1)
                            print(f"Fusion alpha: {fusion_alpha:.1f}")

                        # Gallery navigation (handle BEFORE arrow keys for thermal adjustment)
                        elif self.show_gallery and event.key == pygame.K_LEFT:
                            self.gallery_previous()
                        elif self.show_gallery and event.key == pygame.K_RIGHT:
                            self.gallery_next()
                        elif self.show_gallery and event.key == pygame.K_DELETE:
                            self.delete_current_gallery_file()

                        # Arrow keys for position (only when gallery is NOT open)
                        elif event.key == pygame.K_LEFT:
                            self.thermal_offset_x -= move_step
                            print(f"Offset X: {self.thermal_offset_x}")
                        elif event.key == pygame.K_RIGHT:
                            self.thermal_offset_x += move_step
                            print(f"Offset X: {self.thermal_offset_x}")
                        elif event.key == pygame.K_UP:
                            self.thermal_offset_y -= move_step
                            print(f"Offset Y: {self.thermal_offset_y}")
                        elif event.key == pygame.K_DOWN:
                            self.thermal_offset_y += move_step
                            print(f"Offset Y: {self.thermal_offset_y}")

                        # Scale controls
                        elif event.key == pygame.K_LEFTBRACKET:
                            self.thermal_scale = max(0.1, self.thermal_scale - 0.1)
                            print(f"Scale: {self.thermal_scale:.2f}x")
                        elif event.key == pygame.K_RIGHTBRACKET:
                            self.thermal_scale = min(3.0, self.thermal_scale + 0.1)
                            print(f"Scale: {self.thermal_scale:.2f}x")

                        # Rotation controls
                        elif event.key == pygame.K_COMMA:
                            self.thermal_rotation -= 5.0
                            print(f"Rotation: {self.thermal_rotation:.1f}deg")
                        elif event.key == pygame.K_PERIOD:
                            self.thermal_rotation += 5.0
                            print(f"Rotation: {self.thermal_rotation:.1f}deg")

                        # Calibration grid toggle
                        elif event.key == pygame.K_c:
                            show_calibration = not show_calibration
                            print(f"Calibration grid: {show_calibration}")

                        # Save calibration
                        elif event.key == pygame.K_s:
                            self.save_calibration()

                        # Snapshot
                        elif event.key == pygame.K_p:
                            snapshot_frame = display_frame.copy()
                            self.take_snapshot(snapshot_frame)

                        # Recording toggle
                        elif event.key == pygame.K_r:
                            if not self.recording:
                                self.start_recording(self.display_width, self.display_height)
                            else:
                                self.stop_recording()

                        # Settings menu
                        elif event.key == pygame.K_m:
                            self.show_settings = not self.show_settings
                            print(f"Settings panel: {self.show_settings}")

                        # Toggle histogram
                        elif event.key == pygame.K_h:
                            self.show_histogram = not self.show_histogram
                            print(f"Histogram: {self.show_histogram}")

                        # Cycle colormap
                        elif event.key == pygame.K_k:
                            self.cycle_colormap()

                        # Toggle interpolation
                        elif event.key == pygame.K_i:
                            self.enable_interpolation = not self.enable_interpolation
                            print(f"Frame interpolation: {self.enable_interpolation}")

                        # Gallery toggle
                        elif event.key == pygame.K_g:
                            self.show_gallery = not self.show_gallery
                            if self.show_gallery:
                                count = self.load_gallery_files()
                                print(f"Gallery opened: {count} files")
                            else:
                                print("Gallery closed")

                # Capture camera frame
                camera_frame = self.picam.capture_array()
                camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)

                # Read thermal frame
                thermal_data = self.read_thermal_frame()

                # Apply interpolation for smoother thermal updates
                thermal_data = self.interpolate_thermal_frame(thermal_data)

                thermal_image, min_temp, max_temp = self.process_thermal_image(thermal_data)

                # Determine display mode
                if show_video_only:
                    display_frame = camera_frame
                elif show_thermal_only:
                    display_frame = thermal_image.copy()
                    # Expand to full size if smaller
                    if thermal_image.shape[:2] != camera_frame.shape[:2]:
                        display_frame = np.zeros_like(camera_frame)
                        th, tw = thermal_image.shape[:2]
                        ch, cw = camera_frame.shape[:2]
                        x_pos = (cw - tw) // 2 + self.thermal_offset_x
                        y_pos = (ch - th) // 2 + self.thermal_offset_y
                        # Calculate valid region
                        src_x1 = max(0, -x_pos)
                        src_y1 = max(0, -y_pos)
                        src_x2 = tw - max(0, (x_pos + tw) - cw)
                        src_y2 = th - max(0, (y_pos + th) - ch)
                        dst_x1 = max(0, x_pos)
                        dst_y1 = max(0, y_pos)
                        dst_x2 = min(cw, x_pos + tw)
                        dst_y2 = min(ch, y_pos + th)
                        if src_x2 > src_x1 and src_y2 > src_y1:
                            display_frame[dst_y1:dst_y2, dst_x1:dst_x2] = thermal_image[src_y1:src_y2, src_x1:src_x2]
                else:
                    display_frame = self.fuse_images(camera_frame, thermal_image, fusion_alpha)

                # Add temperature overlay (skip in video-only mode)
                if not show_video_only:
                    display_frame = self.draw_temperature_overlay(display_frame, min_temp, max_temp)

                # Add calibration grid if enabled
                if show_calibration:
                    display_frame = self.draw_calibration_grid(display_frame)

                # Add histogram if enabled
                if self.show_histogram and not show_video_only:
                    display_frame = self.draw_histogram(display_frame, thermal_data, min_temp, max_temp)

                # Save frame for recording BEFORE adding buttons and overlays
                if self.recording:
                    recording_frame = display_frame.copy()
                    self.write_video_frame(recording_frame)

                # Draw touchscreen buttons
                display_frame, buttons = self.draw_touch_buttons(display_frame, show_thermal_only, show_video_only, show_calibration)

                # Add settings panel if enabled
                if self.show_settings:
                    display_frame = self.draw_settings_panel(display_frame)

                # Add gallery overlay if enabled (drawn on top of everything)
                if self.show_gallery:
                    display_frame = self.draw_gallery(display_frame)

                # Add recording indicator
                if self.recording:
                    # Red recording dot
                    cv2.circle(display_frame, (self.display_width - 50, 30), 15, (0, 0, 255), -1)
                    # Recording time
                    duration = time.time() - self.recording_start_time
                    mins = int(duration // 60)
                    secs = int(duration % 60)
                    rec_text = f"REC {mins:02d}:{secs:02d}"
                    cv2.putText(display_frame, rec_text, (self.display_width - 150, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_time)
                    fps_time = time.time()
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, self.display_height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Convert to pygame surface and display
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                display_frame = np.rot90(display_frame)
                display_frame = np.flipud(display_frame)
                surface = pygame.surfarray.make_surface(display_frame)
                self.screen.blit(surface, (0, 0))
                pygame.display.flip()

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        # Stop recording if active
        if self.recording:
            self.stop_recording()
        self.picam.stop()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Thermal Camera Fusion for Raspberry Pi 5')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode with simulated thermal data (no MLX90640 required)')
    parser.add_argument('--windowed', action='store_true',
                        help='Run in windowed mode instead of fullscreen')
    parser.add_argument('--width', type=int, default=1920,
                        help='Display width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Display height (default: 1080)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Initial fusion alpha/transparency (default: 0.5)')

    args = parser.parse_args()

    # Create fusion object
    fusion = ThermalCameraFusion(
        display_width=args.width,
        display_height=args.height,
        demo_mode=args.demo,
        windowed=args.windowed
    )
    fusion.run(fusion_alpha=args.alpha)


if __name__ == "__main__":
    main()

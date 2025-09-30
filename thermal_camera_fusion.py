#!/usr/bin/env python3
"""
Thermal Camera Fusion for Raspberry Pi 5
Fuses MLX90640 thermal camera data with Pi Camera video
"""

import time
import numpy as np
import cv2
from picamera2 import Picamera2
import pygame
from PIL import Image
import json
import os
import argparse


class ThermalCameraFusion:
    def __init__(self, display_width=1920, display_height=1080, config_file="thermal_calibration.json", demo_mode=False):
        """Initialize thermal camera, regular camera, and display"""
        self.display_width = display_width
        self.display_height = display_height
        self.config_file = config_file
        self.demo_mode = demo_mode

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
        self.screen = pygame.display.set_mode((display_width, display_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Thermal Camera Fusion")

        # Color map for thermal data (iron colormap approximation)
        self.colormap = cv2.COLORMAP_JET

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
        print("  +/-: Adjust fusion alpha")
        print("  Arrow keys: Move thermal overlay")
        print("  [/]: Scale thermal overlay")
        print("  ,/.: Rotate thermal overlay")
        print("  c: Toggle calibration grid")
        print("  s: Save calibration")

        running = True
        fps_time = time.time()
        frame_count = 0
        show_calibration = False
        move_step = 10

        try:
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_t:
                            show_thermal_only = not show_thermal_only
                            print(f"Thermal-only mode: {show_thermal_only}")
                        elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                            fusion_alpha = min(1.0, fusion_alpha + 0.1)
                            print(f"Fusion alpha: {fusion_alpha:.1f}")
                        elif event.key == pygame.K_MINUS:
                            fusion_alpha = max(0.0, fusion_alpha - 0.1)
                            print(f"Fusion alpha: {fusion_alpha:.1f}")

                        # Arrow keys for position
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

                # Capture camera frame
                camera_frame = self.picam.capture_array()
                camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)

                # Read thermal frame
                thermal_data = self.read_thermal_frame()
                thermal_image, min_temp, max_temp = self.process_thermal_image(thermal_data)

                # Fuse or show thermal only
                if show_thermal_only:
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

                # Add temperature overlay
                display_frame = self.draw_temperature_overlay(display_frame, min_temp, max_temp)

                # Add calibration grid if enabled
                if show_calibration:
                    display_frame = self.draw_calibration_grid(display_frame)

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
        self.picam.stop()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Thermal Camera Fusion for Raspberry Pi 5')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode with simulated thermal data (no MLX90640 required)')
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
        demo_mode=args.demo
    )
    fusion.run(fusion_alpha=args.alpha)


if __name__ == "__main__":
    main()

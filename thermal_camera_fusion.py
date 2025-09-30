#!/usr/bin/env python3
"""
Thermal Camera Fusion for Raspberry Pi 5
Fuses MLX90640 thermal camera data with Pi Camera video
"""

import time
import board
import busio
import numpy as np
import cv2
from picamera2 import Picamera2
import adafruit_mlx90640
import pygame
from PIL import Image


class ThermalCameraFusion:
    def __init__(self, display_width=1920, display_height=1080):
        """Initialize thermal camera, regular camera, and display"""
        self.display_width = display_width
        self.display_height = display_height

        # Initialize thermal camera (MLX90640)
        print("Initializing MLX90640 thermal camera...")
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
        """Read frame from thermal camera"""
        try:
            self.mlx.getFrame(self.thermal_frame)
            # Reshape to 24x32 array
            thermal_image = np.reshape(self.thermal_frame, (24, 32))
            return thermal_image
        except ValueError:
            # Frame not ready, return previous frame
            return np.reshape(self.thermal_frame, (24, 32))

    def process_thermal_image(self, thermal_data):
        """Convert thermal data to colored image"""
        # Normalize to 0-255 range
        min_temp = np.min(thermal_data)
        max_temp = np.max(thermal_data)

        if max_temp - min_temp > 0:
            normalized = ((thermal_data - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(thermal_data, dtype=np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(normalized, self.colormap)

        # Resize to display size
        resized = cv2.resize(colored, (self.display_width, self.display_height),
                           interpolation=cv2.INTER_CUBIC)

        return resized, min_temp, max_temp

    def fuse_images(self, camera_frame, thermal_frame, alpha=0.5):
        """Fuse camera and thermal images with transparency"""
        # Ensure both images are RGB
        if len(camera_frame.shape) == 2:
            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_GRAY2RGB)
        if len(thermal_frame.shape) == 2:
            thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2RGB)

        # Blend images
        fused = cv2.addWeighted(camera_frame, 1 - alpha, thermal_frame, alpha, 0)
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

    def run(self, fusion_alpha=0.5, show_thermal_only=False):
        """Main loop to capture and display fused images"""
        print("Starting thermal camera fusion...")
        print("Press 'q' to quit, 't' to toggle thermal-only mode, '+/-' to adjust fusion")

        running = True
        fps_time = time.time()
        frame_count = 0

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

                # Capture camera frame
                camera_frame = self.picam.capture_array()
                camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)

                # Read thermal frame
                thermal_data = self.read_thermal_frame()
                thermal_image, min_temp, max_temp = self.process_thermal_image(thermal_data)

                # Fuse or show thermal only
                if show_thermal_only:
                    display_frame = thermal_image
                else:
                    display_frame = self.fuse_images(camera_frame, thermal_image, fusion_alpha)

                # Add temperature overlay
                display_frame = self.draw_temperature_overlay(display_frame, min_temp, max_temp)

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
    # Adjust display resolution as needed for your display
    fusion = ThermalCameraFusion(display_width=1920, display_height=1080)
    fusion.run(fusion_alpha=0.5)


if __name__ == "__main__":
    main()

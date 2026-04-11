"""Drowsiness Detection with ESP32 Integration

This script runs YOLO-based drowsiness detection on Raspberry Pi and sends
control commands to an ESP32 device via WiFi to activate buzzers and vibrators.

The Raspberry Pi connects to the ESP32's WiFi AP (ESP32-Drowsiness-AP) and
sends HTTP POST requests with drowsiness level commands.

Drowsiness Levels:
    0: ALERT FULLY AWAKE      -> No alert
    1: EARLY DROWSINESS        -> Low intensity (20%)
    2: MODERATE DROWSINESS     -> Medium intensity (50%)
    3: MICROSLEEP              -> High intensity (80%)
    4: REM SLEEP               -> Maximum intensity (100%)
    5: STAGE N1 N2 N3          -> Maximum intensity (100%)

Usage:
    python scripts/drowsiness_detection_esp32.py --model yolo/outputs/20260302-011953_best.pt
    python scripts/drowsiness_detection_esp32.py --camera 0 --esp32-ip 192.168.4.1
"""

import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from collections import deque
import cv2
import numpy as np


# ESP32 Configuration
DEFAULT_ESP32_IP = "192.168.4.1"  # Default IP for ESP32 AP mode
DEFAULT_ESP32_PORT = 80
ESP32_COMMAND_ENDPOINT = "/command"

# Drowsiness level mapping to intensity
DROWSINESS_LEVELS = {
    "ALERT  FULLY AWAKE": {"level": 0, "description": "Fully awake"},
    "EARLY DROWSINESS": {"level": 1, "description": "Early signs"},
    "MODERATE DROWSINESS": {"level": 2, "description": "Moderate"},
    "MICROSLEEP": {"level": 3, "description": "Microsleep detected"},
    "REM SLEEP": {"level": 4, "description": "REM sleep"},
    "STAGE N1 N2 N3": {"level": 5, "description": "Deep sleep"}
}

# Alert thresholds
ALERT_THRESHOLD = 1  # Start alerting at EARLY DROWSINESS
BUZZER_THRESHOLD = 2  # Use buzzer at MODERATE and above
VIBRATOR_THRESHOLD = 1  # Use vibrator at EARLY and above
DETECTION_HOLD_SECONDS = 3.0  # Require 3 seconds of continuous detection before alerting

# Smoothing settings
SMOOTHING_WINDOW = 5  # Average predictions over last N frames for stability


class ESP32Controller:
    """Handles HTTP communication with ESP32"""
    
    def __init__(self, ip_address, port=80, timeout=5):
        self.ip = ip_address
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{ip_address}:{port}"
        self.last_command_time = 0
        self.command_cooldown = 2.0  # Minimum seconds between commands
        self.connected = False
        
    def test_connection(self):
        """Test if ESP32 is reachable"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            self.connected = True
            return True
        except Exception as e:
            self.connected = False
            return False
    
    def send_command(self, command):
        """Send command to ESP32"""
        current_time = time.time()
        
        # Enforce cooldown to prevent command spam
        if current_time - self.last_command_time < self.command_cooldown:
            return False
        
        try:
            url = f"{self.base_url}{ESP32_COMMAND_ENDPOINT}"
            payload = {"command": command}
            response = requests.post(url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                self.last_command_time = current_time
                self.connected = True
                return True
            else:
                print(f"ESP32 returned status code: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"Connection error: Cannot reach ESP32 at {self.ip}")
            self.connected = False
            return False
        except requests.exceptions.Timeout:
            print(f"Timeout: ESP32 at {self.ip} did not respond")
            self.connected = False
            return False
        except Exception as e:
            print(f"Error sending command to ESP32: {e}")
            self.connected = False
            return False
    
    def send_alert_signal(self):
        """Send a single generic alert signal to the ESP32."""
        return self.send_command("ALERT")


class DrowsinessDetector:
    """Drowsiness detection using YOLO model"""
    
    def __init__(self, model_path, confidence=0.5, imgsz=640, device='auto'):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.model = None
        self.class_names = None
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Get class names from model
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
            print(f"Model loaded with {len(self.class_names)} classes:")
            for idx, name in self.class_names.items():
                print(f"  {idx}: {name}")
        else:
            print("Warning: Could not extract class names from model")
            self.class_names = {}
        
    def predict(self, frame):
        """Run inference on a frame"""
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )
        return results[0]
    
    def get_drowsiness_level(self, result):
        """Extract highest drowsiness level from detection result"""
        if result.boxes is None or len(result.boxes) == 0:
            return None, 0, None
        
        # Get the detection with highest confidence
        best_idx = result.boxes.conf.argmax()
        class_id = int(result.boxes.cls[best_idx])
        confidence = float(result.boxes.conf[best_idx])
        
        # Get class name
        class_name = self.class_names.get(class_id, f"Unknown ({class_id})")
        
        # Find matching drowsiness level
        drowsiness_info = DROWSINESS_LEVELS.get(class_name, None)
        
        return class_name, confidence, drowsiness_info
    
    def get_smoothed_level(self, current_level):
        """Apply temporal smoothing to reduce false alerts"""
        self.prediction_history.append(current_level)
        
        if len(self.prediction_history) == 0:
            return 0
        
        # Return average level
        return sum(self.prediction_history) / len(self.prediction_history)


def find_default_model():
    """Find the most recent trained model"""
    root = Path(__file__).resolve().parents[1]
    outputs_dir = root / 'yolo' / 'outputs'
    
    # Look for timestamped models
    models = sorted(outputs_dir.glob('*_best.pt'), reverse=True)
    if models:
        return models[0]
    
    # Fallback to standard paths
    standard_paths = [
        outputs_dir / 'yolo_training' / 'weights' / 'best.pt',
        outputs_dir / 'best.pt'
    ]
    
    for path in standard_paths:
        if path.exists():
            return path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Drowsiness Detection with ESP32 Integration')
    parser.add_argument('--model', type=str, default=None, help='Path to YOLO model weights')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--esp32-ip', type=str, default=DEFAULT_ESP32_IP, help='ESP32 IP address')
    parser.add_argument('--esp32-port', type=int, default=DEFAULT_ESP32_PORT, help='ESP32 port')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, or cuda')
    parser.add_argument('--display', action='store_true', help='Show video display (if available)')
    parser.add_argument('--test-esp32', action='store_true', help='Test ESP32 connection only')
    args = parser.parse_args()
    
    # Initialize ESP32 controller
    print("\n" + "="*60)
    print("Drowsiness Detection System - ESP32 Integration")
    print("="*60)
    
    esp32 = ESP32Controller(args.esp32_ip, args.esp32_port)
    
    print(f"\nConnecting to ESP32 at {args.esp32_ip}:{args.esp32_port}...")
    if esp32.test_connection():
        print("✓ ESP32 connected successfully!")
    else:
        print("✗ Cannot connect to ESP32")
        print("  Make sure:")
        print("  1. Raspberry Pi is connected to ESP32-Drowsiness-AP WiFi")
        print("  2. ESP32 is powered on and running")
        print("  3. IP address is correct (default: 192.168.4.1)")
        if not args.test_esp32:
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    # If test mode, just test and exit
    if args.test_esp32:
        print("\nTesting ESP32 devices...")
        print("Testing buzzer at 50% intensity...")
        esp32.send_command("TEST:buzzer:50")
        time.sleep(4)
        print("Testing vibrator at 50% intensity...")
        esp32.send_command("TEST:vibrator:50")
        print("\nTest complete!")
        return
    
    # Find and load model
    model_path = Path(args.model) if args.model else find_default_model()
    
    if model_path is None or not model_path.exists():
        print("\nError: No model found!")
        print("Provide --model or train a model first")
        return
    
    # Initialize detector
    detector = DrowsinessDetector(
        model_path=model_path,
        confidence=args.conf,
        imgsz=args.imgsz,
        device=args.device
    )
    
    # Open camera
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return
    
    # Camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✓ Camera opened: {frame_width}x{frame_height} @ {fps}fps")
    print("\n" + "="*60)
    print("Starting detection... Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    frame_count = 0
    last_alert_level = 0
    detection_hold_start = None
    detection_hold_class = None
    alert_sent_for_hold = False
    display_supported = args.display
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            frame_count += 1
            
            # Run detection
            result = detector.predict(frame)
            class_name, confidence, drowsiness_info = detector.get_drowsiness_level(result)
            
            # Get current level
            current_level = drowsiness_info["level"] if drowsiness_info else 0
            smoothed_level = detector.get_smoothed_level(current_level)
            
            # Determine if alert is needed
            if drowsiness_info and current_level >= ALERT_THRESHOLD:
                description = drowsiness_info["description"]
                current_time = time.monotonic()

                if detection_hold_class != class_name:
                    detection_hold_class = class_name
                    detection_hold_start = current_time
                    alert_sent_for_hold = False
                elif detection_hold_start is None:
                    detection_hold_start = current_time
                    alert_sent_for_hold = False

                hold_duration = current_time - detection_hold_start
                
                print(f"\n[Frame {frame_count}] {class_name} ({confidence:.2f})")
                print(f"  Level: {current_level} | Smoothed: {smoothed_level:.1f} | Held: {hold_duration:.1f}s")
                print(f"  Status: {description}")
                
                # Only send the alert signal after the detection has been stable for 3 seconds
                if hold_duration >= DETECTION_HOLD_SECONDS and not alert_sent_for_hold:
                    esp32.send_alert_signal()
                    alert_sent_for_hold = True
                    last_alert_level = current_level
            else:
                # Print status every 30 frames when alert
                if frame_count % 30 == 0:
                    status = class_name if class_name else "No detection"
                    print(f"[Frame {frame_count}] {status} - Normal (No alert)")
                
                # Reset alert level when returning to normal
                if last_alert_level > 0:
                    last_alert_level = 0
                detection_hold_start = None
                detection_hold_class = None
                alert_sent_for_hold = False
            
            # Display frame if requested
            if display_supported:
                try:
                    annotated_frame = result.plot()
                    
                    # Add status overlay
                    status_text = f"ESP32: {'Connected' if esp32.connected else 'Disconnected'}"
                    cv2.putText(annotated_frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0) if esp32.connected else (0, 0, 255), 2)
                    
                    if drowsiness_info:
                        level_text = f"Level: {current_level} ({drowsiness_info['description']})"
                        cv2.putText(annotated_frame, level_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Drowsiness Detection', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nExiting...")
                        break
                        
                except cv2.error:
                    print("Warning: Display not supported, disabling...")
                    display_supported = False
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        cap.release()
        if display_supported:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print("\n" + "="*60)
        print(f"Session complete - {frame_count} frames processed")
        print("="*60)


if __name__ == '__main__':
    main()

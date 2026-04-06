# Drowsiness Detection System - ESP32 Integration

## System Overview

This system uses a Raspberry Pi to run YOLO-based drowsiness detection and sends a generic alert signal to an ESP32 microcontroller after the same drowsiness stage is detected continuously for 3 seconds.

### Components

- **Raspberry Pi**: Runs the machine learning model, processes camera feed
- **ESP32**: Creates WiFi AP, controls buzzers and vibrators
- **Camera**: USB or CSI camera connected to Raspberry Pi
- **Buzzers (2x)**: Audio alert devices connected to ESP32
- **Vibrator Motors (6x)**: Haptic feedback devices connected to ESP32

### Drowsiness Levels

| Level | Class Name | Purpose |
|-------|------------|---------|
| 0 | ALERT  FULLY AWAKE | No signal |
| 1 | EARLY DROWSINESS | Trigger alert signal |
| 2 | MODERATE DROWSINESS | Trigger alert signal |
| 3 | MICROSLEEP | Trigger alert signal |
| 4 | REM SLEEP | Trigger alert signal |
| 5 | STAGE N1 N2 N3 | Trigger alert signal |

---

## Setup Instructions

### 1. ESP32 Setup

#### Hardware Connections
- **Buzzer Pins**: GPIO 25, 33
- **Vibrator Pins**: GPIO 26, 27, 14, 12, 13, 15
- **Status LEDs**:
  - Green (Connected): GPIO 2
  - Red (Disconnected): GPIO 4

#### Upload Firmware
1. Open `drowsiness_controllerWIFI.ino` in Arduino IDE
2. Select board: ESP32 Dev Module
3. Select correct COM port
4. Upload the sketch

#### WiFi Access Point
- **SSID**: `ESP32-Drowsiness-AP`
- **Password**: `drowsy123`
- **IP Address**: `192.168.4.1`
- **Port**: `80`
- **Endpoint**: `http://192.168.4.1/command`

### 2. Raspberry Pi Setup

#### Prerequisites
- Raspberry Pi 3/4 or newer
- Raspbian OS (Bullseye or later recommended)
- Python 3.8 or newer
- USB or CSI camera
- WiFi adapter (built-in on most models)

#### Installation Steps

```bash
# Clone the repository (if not already done)
cd /home/admin
git clone https://github.com/yourusername/NutriBin-MachineLearning.git
cd NutriBin-MachineLearning

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r scripts/requirements.txt

# Verify installation
python -c "import cv2, ultralytics, requests; print('All packages installed successfully')"
```

#### Trained Model
Ensure you have a trained YOLO model at:
- `yolo/outputs/20260302-011953_best.pt` (or similar timestamp)
- Or specify with `--model` parameter

---

## Running the System

### Method 1: Manual Run (Testing)

#### Step 1: Connect to ESP32 WiFi

```bash
# Option A: Using the helper script (requires sudo)
sudo ./scripts/connect_esp32_wifi.sh

# Option B: Manual connection
sudo nmcli dev wifi connect ESP32-Drowsiness-AP password drowsy123

# Verify connection
ping -c 3 192.168.4.1
```

#### Step 2: Test ESP32 Connection

```bash
# Activate virtual environment
source venv/bin/activate

# Test ESP32 devices
python scripts/drowsiness_detection_esp32.py --test-esp32
```

This will test the buzzer and vibrator for 3 seconds each.

#### Step 3: Run Drowsiness Detection

```bash
# Basic usage
python scripts/drowsiness_detection_esp32.py

# With display (if X11/display available)
python scripts/drowsiness_detection_esp32.py --display

# Custom camera
python scripts/drowsiness_detection_esp32.py --camera 1

# Custom model
python scripts/drowsiness_detection_esp32.py --model path/to/your/model.pt

# Full options
python scripts/drowsiness_detection_esp32.py \
    --model yolo/outputs/20260302-011953_best.pt \
    --camera 0 \
    --esp32-ip 192.168.4.1 \
    --conf 0.5 \
    --display
```

#### Stop the Script
Press `Ctrl+C` to stop

---

### Method 2: Auto-Start on Boot (Production)

#### Install Systemd Service

```bash
# Copy service file
sudo cp drowsiness-detection.service /etc/systemd/system/

# Configure auto-connect to ESP32 WiFi
sudo nmcli connection modify ESP32-Drowsiness-AP connection.autoconnect yes

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable drowsiness-detection.service

# Start service now
sudo systemctl start drowsiness-detection.service

# Check status
sudo systemctl status drowsiness-detection.service

# View logs
sudo journalctl -u drowsiness-detection.service -f
```

#### Service Management

```bash
# Stop service
sudo systemctl stop drowsiness-detection.service

# Restart service
sudo systemctl restart drowsiness-detection.service

# Disable auto-start
sudo systemctl disable drowsiness-detection.service

# View recent logs
sudo journalctl -u drowsiness-detection.service -n 100
```

---

## Command Line Options

```
usage: drowsiness_detection_esp32.py [-h] [--model MODEL] [--camera CAMERA]
                                      [--esp32-ip ESP32_IP] [--esp32-port ESP32_PORT]
                                      [--conf CONF] [--imgsz IMGSZ] [--device DEVICE]
                                      [--display] [--test-esp32]

Options:
  --model MODEL         Path to YOLO model weights (auto-detects if not specified)
  --camera CAMERA       Camera device ID (default: 0)
  --esp32-ip ESP32_IP   ESP32 IP address (default: 192.168.4.1)
  --esp32-port ESP32_PORT  ESP32 port (default: 80)
  --conf CONF           Detection confidence threshold (default: 0.5)
  --imgsz IMGSZ         Input image size (default: 640)
  --device DEVICE       Device: auto, cpu, or cuda (default: auto)
  --display             Show video display if available
  --test-esp32          Test ESP32 connection and devices only
```

---

## Troubleshooting

### Cannot Connect to ESP32 WiFi

**Problem**: Raspberry Pi can't see or connect to ESP32-Drowsiness-AP

**Solutions**:
1. Verify ESP32 is powered on and running
2. Check ESP32 serial monitor for "Access Point started successfully"
3. Scan for available networks:
   ```bash
   sudo nmcli dev wifi list | grep ESP32
   ```
4. Check WiFi password is correct: `drowsy123`
5. Restart ESP32 and try again

### ESP32 Not Responding to Commands

**Problem**: Connected to WiFi but cannot reach 192.168.4.1

**Solutions**:
1. Ping the ESP32:
   ```bash
   ping 192.168.4.1
   ```
2. Check ESP32 serial monitor for errors
3. Test manual HTTP command:
   ```bash
   curl -X POST http://192.168.4.1/command \
        -H "Content-Type: application/json" \
        -d '{"command":"TEST:buzzer:50"}'
   ```
4. Verify ESP32 is in AP mode (not trying to connect to another WiFi)

### Camera Not Working

**Problem**: Cannot open camera or poor detection

**Solutions**:
1. List available cameras:
   ```bash
   ls /dev/video*
   ```
2. Try different camera IDs: `--camera 1`, `--camera 2`, etc.
3. Test camera with:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
   ```
4. Ensure camera permissions:
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

### Low Detection Accuracy

**Problem**: False alerts or missed drowsiness

**Solutions**:
1. Adjust confidence threshold: `--conf 0.3` (lower) or `--conf 0.7` (higher)
2. Ensure good lighting conditions
3. Position camera to clearly capture face/eyes
4. Retrain model with more data
5. Check model is loaded correctly in output messages

### Buzzer/Vibrator Not Activating

**Problem**: ESP32 receives commands but devices don't activate

**Solutions**:
1. Check ESP32 wiring connections
2. Verify GPIO pins match code configuration
3. Test devices manually using test mode
4. Check power supply can handle all devices
5. Monitor ESP32 serial output for device activation messages

### Service Won't Start on Boot

**Problem**: Systemd service fails to start

**Solutions**:
1. Check service status:
   ```bash
   sudo systemctl status drowsiness-detection.service
   ```
2. View detailed logs:
   ```bash
   sudo journalctl -u drowsiness-detection.service -n 50
   ```
3. Verify Python path in service file
4. Ensure model file exists
5. Check WiFi connects automatically:
   ```bash
   nmcli connection show
   ```

---

## API Reference

### ESP32 HTTP API

**Endpoint**: `POST http://192.168.4.1/command`

**Request Body** (JSON):
```json
{
  "command": "TEST:buzzer:50"
}
```

**Command Format**:
- Test buzzer: `TEST:buzzer:<intensity>`
- Test vibrator: `TEST:vibrator:<intensity>`
- Intensity range: 0-100 (percentage)

**Response** (JSON):
```json
{
  "status": "success"
}
```

**Example using curl**:
```bash
# Test buzzer at 75%
curl -X POST http://192.168.4.1/command \
     -H "Content-Type: application/json" \
     -d '{"command":"TEST:buzzer:75"}'

# Test vibrator at 30%
curl -X POST http://192.168.4.1/command \
     -H "Content-Type: application/json" \
     -d '{"command":"TEST:vibrator:30"}'
```

---

## Performance Optimization

### Raspberry Pi Optimization

```bash
# Enable camera hardware acceleration
echo "start_x=1" | sudo tee -a /boot/config.txt
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Disable desktop to save resources
sudo systemctl set-default multi-user.target

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Model Optimization

For faster inference on Raspberry Pi:

```bash
# Export to TensorFlow Lite (coming soon)
# Or use smaller YOLO variant (YOLOv8n instead of YOLOv8m)
```

---

## Development & Training

### Train New Model

See `yolo/scripts/train_model.py` for training new drowsiness detection models.

### Collect Training Data

Images should be organized in:
```
yolo/data/images/
├── ALERT  FULLY AWAKE/
├── EARLY DROWSINESS/
├── MODERATE DROWSINESS/
├── MICROSLEEP/
├── REM SLEEP/
└── STAGE N1 N2 N3/
```

---

## Safety Considerations

⚠️ **Important Safety Notes**:

1. **Not a Safety Device**: This is an assistive system, not a certified safety device
2. **Driver Responsibility**: Driver is always responsible for vehicle control
3. **Secondary Measure**: Should complement, not replace, adequate rest
4. **Testing Required**: Thoroughly test before real-world use
5. **Environmental Factors**: Lighting, camera position, and conditions affect accuracy
6. **Regular Calibration**: Periodically verify system performance

---

## License

See LICENSE file for details.

---

## Support

For issues and questions:
1. Check this documentation
2. Review ESP32 serial monitor output
3. Check Raspberry Pi logs: `sudo journalctl -u drowsiness-detection`
4. Open an issue on GitHub repository

---

## Version History

- **v1.0** (2026-03-03): Initial release with ESP32 integration
  - YOLO-based drowsiness detection
  - WiFi AP communication
  - Progressive intensity alerts
  - Systemd service support

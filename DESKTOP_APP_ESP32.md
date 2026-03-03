# Desktop App with ESP32 Integration - Quick Guide

## What Was Modified

The `yolo/scripts/desktop_app.py` now includes **ESP32 drowsiness detection integration** in the "Live Detection" feature.

## Features Added

### 1. **ESP32 Controller Class**
- Handles HTTP communication with ESP32
- Tests connection on startup
- Sends alert commands based on drowsiness level
- 2-second cooldown between commands to prevent spam

### 2. **Drowsiness Level Detection**
When the YOLO model detects drowsiness classes:
- **Level 0** (ALERT  FULLY AWAKE) → No alert
- **Level 1** (EARLY DROWSINESS) → Vibrator at 20%
- **Level 2** (MODERATE DROWSINESS) → Buzzer + Vibrator at 50%
- **Level 3** (MICROSLEEP) → Buzzer + Vibrator at 80%
- **Level 4-5** (REM SLEEP / STAGE N1 N2 N3) → Maximum at 100%

### 3. **Live Status Display**
The camera preview shows:
- 📹 Camera status
- 🎯 Detection count
- 🟢/🔴 ESP32 connection status
- 😊😐😪😴💤 Drowsiness level emoji
- Current drowsiness class

### 4. **Enhanced Parameters Dialog**
New options for "Live Detection":
- ✅ **Enable ESP32** checkbox
- 🌐 **ESP32 IP** field (default: 192.168.4.1)

## How to Use

### Step 1: Prepare ESP32
```bash
# 1. Upload drowsiness_controllerWIFI.ino to ESP32
# 2. Power on ESP32 - it creates WiFi AP: ESP32-Drowsiness-AP
```

### Step 2: Connect Raspberry Pi to ESP32 WiFi
```bash
# Connect to ESP32 WiFi
sudo nmcli dev wifi connect ESP32-Drowsiness-AP password drowsy123

# Verify connection
ping -c 3 192.168.4.1
```

### Step 3: Run Desktop App
```bash
cd /home/admin/NutriBin-MachineLearning/yolo/scripts
python desktop_app.py
```

### Step 4: Configure Live Detection
1. Click **"5. Live Detection"** button
2. In the parameters dialog:
   - Set **Model Path** (or leave as best.pt for auto-detect)
   - Set **Camera ID** (usually 0)
   - Set **Confidence** threshold (0.25-0.5 recommended)
   - ✅ Check **"Enable ESP32"**
   - Enter **ESP32 IP**: `192.168.4.1`
3. Click **"✓ Run"**

### Step 5: Monitor Detection
- The camera preview will show live detection
- Status bar shows:
  - Camera status
  - Detection count
  - 🟢 ESP32 connected (or 🔴 if disconnected)
  - 😊😐😪😴💤 Current drowsiness level
- Console log shows alert activations:
  ```
  🚨 Alert: MODERATE DROWSINESS (0.92) - 50%
  ```

### Step 6: Stop Detection
- Click **"Stop Process"** button
- Or close the desktop app

## Troubleshooting

### ESP32 Shows Red 🔴
**Problem**: ESP32 not reachable

**Solutions**:
1. Check WiFi connection: `iwconfig` or `nmcli con show`
2. Ping ESP32: `ping 192.168.4.1`
3. Verify ESP32 is powered and running
4. Check ESP32 serial monitor for errors

### No Alerts Sent
**Problem**: Drowsiness detected but no ESP32 activation

**Solutions**:
1. Make sure **"Enable ESP32"** checkbox is checked
2. Verify model is trained with drowsiness classes:
   - ALERT  FULLY AWAKE
   - EARLY DROWSINESS
   - MODERATE DROWSINESS
   - MICROSLEEP
   - REM SLEEP
   - STAGE N1 N2 N3
3. Check confidence threshold isn't too high
4. Look for drowsiness emojis in status bar

### Camera Not Opening
**Problem**: Cannot open camera device

**Solutions**:
1. List cameras: `ls /dev/video*`
2. Try different Camera ID: 1, 2, etc.
3. Close other apps using camera
4. Check permissions: `sudo usermod -a -G video $USER`

### Model Not Found
**Problem**: "No model found" error

**Solutions**:
1. Check model exists: `ls yolo/outputs/*_best.pt`
2. Specify full path in "Model Path" field
3. Train a model first if none exists

## Comparison: Desktop App vs Standalone Script

| Feature | Desktop App | Standalone Script |
|---------|-------------|-------------------|
| GUI | ✅ Yes | ❌ No |
| Camera Preview | ✅ Yes | ⚠️ Optional |
| ESP32 Integration | ✅ Yes | ✅ Yes |
| Configuration | ✅ Dialog | ⚙️ Command line |
| Multiple Tools | ✅ 7+ tools | ❌ Single purpose |
| Training | ✅ Yes | ❌ No |
| Logs | ✅ Console panel | 📝 Terminal |
| Best For | Desktop/development | Headless/automated |

## Advanced Usage

### Custom Alert Thresholds
Edit these constants in `desktop_app.py`:
```python
ALERT_THRESHOLD = 1      # Start alerting at level 1
BUZZER_THRESHOLD = 2     # Use buzzer at level 2+
VIBRATOR_THRESHOLD = 1   # Use vibrator at level 1+
```

### Smoothing Window
Adjust temporal smoothing:
```python
SMOOTHING_WINDOW = 5  # Average over 5 frames
```

### Command Cooldown
Change minimum time between ESP32 commands:
```python
self.command_cooldown = 2.0  # 2 seconds
```

## What Happens Behind the Scenes

```
┌──────────────────────────────────────────────────────┐
│  1. Camera captures frame                            │
│  2. YOLO model detects drowsiness class              │
│  3. Lookup class in DROWSINESS_LEVELS                │
│  4. Add to smoothing history (5 frames)              │
│  5. If level >= threshold:                           │
│     a. Determine buzzer/vibrator usage               │
│     b. Send HTTP POST to ESP32                       │
│     c. ESP32 activates devices at intensity%         │
│     d. Update UI with emoji & status                 │
│     e. Log alert to console                          │
└──────────────────────────────────────────────────────┘
```

## Benefits of Desktop App Integration

1. **Visual Feedback**: See detections and drowsiness level in real-time
2. **Easy Configuration**: No command-line arguments needed
3. **Integrated Workflow**: Train, test, and deploy in one app
4. **ESP32 Status**: Visual indicator of connection health
5. **Console Logs**: See alert activations and errors
6. **Quick Testing**: Enable/disable ESP32 with checkbox

## Next Steps

1. **Test the system**: Run desktop app with ESP32 enabled
2. **Adjust thresholds**: Tune confidence and alert levels
3. **Train better model**: Collect more drowsiness data
4. **Add automation**: Use systemd to run on startup (use standalone script for this)

---

**Note**: For production/always-on deployment, use the standalone script `scripts/drowsiness_detection_esp32.py` with systemd service for better reliability and lower resource usage.

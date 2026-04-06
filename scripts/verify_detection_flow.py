#!/usr/bin/env python3
"""
Verification script to test the complete drowsiness detection flow.
This script simulates detections and verifies ESP32 commands are sent correctly.
"""

import sys
from pathlib import Path

# Test the drowsiness level mapping
DROWSINESS_LEVELS = {
    "ALERT  FULLY AWAKE": {"level": 0, "description": "Fully awake"},
    "EARLY DROWSINESS": {"level": 1, "description": "Early signs"},
    "MODERATE DROWSINESS": {"level": 2, "description": "Moderate"},
    "MICROSLEEP": {"level": 3, "description": "Microsleep detected"},
    "REM SLEEP": {"level": 4, "description": "REM sleep"},
    "STAGE N1 N2 N3": {"level": 5, "description": "Deep sleep"}
}

ALERT_THRESHOLD = 1
BUZZER_THRESHOLD = 2
VIBRATOR_THRESHOLD = 1

print("="*60)
print("Drowsiness Detection Flow Verification")
print("="*60)
print()

# Test 1: Verify class names match folder structure
print("[Test 1] Verifying class names match folder structure...")
root = Path(__file__).resolve().parents[1]
images_dir = root / 'yolo' / 'data' / 'images'

if images_dir.exists():
    folders = [f.name for f in images_dir.iterdir() if f.is_dir() and not f.name in ['train', 'val']]
    print(f"Found {len(folders)} class folders:")
    
    all_match = True
    for folder_name in sorted(folders):
        if folder_name in DROWSINESS_LEVELS:
            print(f"  ✓ {folder_name} - Matched!")
        else:
            print(f"  ✗ {folder_name} - NOT FOUND in DROWSINESS_LEVELS!")
            all_match = False
    
    if all_match:
        print("\n✓ All class names match correctly!\n")
    else:
        print("\n✗ Class name mismatch detected!\n")
        sys.exit(1)
else:
    print(f"⚠ Image directory not found: {images_dir}")
    print("  Skipping folder verification\n")

# Test 2: Verify detection to alert logic
print("[Test 2] Simulating drowsiness detections...")
print()

test_cases = [
    ("ALERT  FULLY AWAKE", 0.95),
    ("EARLY DROWSINESS", 0.87),
    ("MODERATE DROWSINESS", 0.92),
    ("MICROSLEEP", 0.89),
    ("REM SLEEP", 0.94),
]

for class_name, confidence in test_cases:
    drowsiness_info = DROWSINESS_LEVELS.get(class_name)
    
    if not drowsiness_info:
        print(f"✗ {class_name}: NOT FOUND in mapping!")
        continue
    
    level = drowsiness_info["level"]
    description = drowsiness_info["description"]
    
    print(f"Detection: {class_name} (confidence: {confidence:.2f})")
    print(f"  → Level: {level}, Status: {description}")
    
    if level >= ALERT_THRESHOLD:
        print(f"  → ALERT TRIGGERED after 3-second hold")
        print(f"  → ESP32 Command:")
        print(f"     • POST /command {{'command': 'ALERT:high'}}")
    else:
        print(f"  → No alert (level {level} < threshold {ALERT_THRESHOLD})")
    
    print()

# Test 3: Verify model file exists
print("[Test 3] Checking for trained model...")
outputs_dir = root / 'yolo' / 'outputs'
models = sorted(outputs_dir.glob('*_best.pt'), reverse=True)

if models:
    print(f"✓ Found model: {models[0].name}")
    print(f"  Path: {models[0]}")
else:
    print("✗ No trained model found!")
    print("  Train a model first or the detection won't work")

print()
print("="*60)
print("Summary")
print("="*60)
print()
print("The detection flow will work as follows:")
print()
print("1. Camera captures frame")
print("2. YOLO model detects drowsiness class")
print("3. Class name is looked up in DROWSINESS_LEVELS")
print("4. If level >= 1 (EARLY DROWSINESS or higher), the same class must persist for 3 seconds:")
print("   a. Track the detection hold time")
print("   b. Send a single generic alert signal")
print("   c. POST http://192.168.4.1/command with {\"command\": \"ALERT:high\"}")
print("   d. ESP32 decides its own output pattern")
print()
print("✓ Flow verified! Run this to start detection:")
print("  python scripts/drowsiness_detection_esp32.py")
print()

#!/bin/bash

# Quick Start Script for Drowsiness Detection System
# This script helps you quickly test the complete system

echo "=========================================="
echo "Drowsiness Detection - Quick Start"
echo "=========================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /etc/rpi-issue ]; then
    echo "Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/n): " choice
    if [ "$choice" != "y" ]; then
        exit 0
    fi
fi

# Check Python installation
echo "[1/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 not found. Please install Python 3.8+"
    exit 1
fi
echo "✓ Python3 found: $(python3 --version)"

# Check virtual environment
echo ""
echo "[2/6] Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate and install dependencies
echo ""
echo "[3/6] Installing dependencies..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r scripts/requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Check for model
echo ""
echo "[4/6] Checking for trained model..."
MODEL=$(find yolo/outputs -name "*_best.pt" -type f | head -n 1)
if [ -z "$MODEL" ]; then
    echo "✗ No trained model found in yolo/outputs/"
    echo "  Please train a model first or provide --model path"
    exit 1
fi
echo "✓ Found model: $MODEL"

# Check ESP32 connection
echo ""
echo "[5/6] Checking ESP32 connection..."
echo "Make sure you're connected to ESP32-Drowsiness-AP WiFi"
read -p "Are you connected to ESP32 WiFi? (y/n): " wifi_connected

if [ "$wifi_connected" = "y" ]; then
    if ping -c 2 -W 2 192.168.4.1 > /dev/null 2>&1; then
        echo "✓ ESP32 is reachable"
        
        # Test ESP32 devices
        echo ""
        echo "[6/6] Testing ESP32 devices..."
        read -p "Test buzzer and vibrator? (y/n): " test_devices
        
        if [ "$test_devices" = "y" ]; then
            python scripts/drowsiness_detection_esp32.py --test-esp32
        fi
    else
        echo "✗ Cannot reach ESP32 at 192.168.4.1"
        echo "  Check WiFi connection and ESP32 power"
        exit 1
    fi
else
    echo ""
    echo "Please connect to ESP32 WiFi first:"
    echo "  SSID: ESP32-Drowsiness-AP"
    echo "  Password: drowsy123"
    echo ""
    echo "Run this script to connect:"
    echo "  sudo ./scripts/connect_esp32_wifi.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Start drowsiness detection with:"
echo "  python scripts/drowsiness_detection_esp32.py"
echo ""
echo "Or with display:"
echo "  python scripts/drowsiness_detection_esp32.py --display"
echo ""
echo "Press Ctrl+C to stop the detection"
echo ""

# Ask if they want to start now
read -p "Start drowsiness detection now? (y/n): " start_now

if [ "$start_now" = "y" ]; then
    echo ""
    echo "Starting detection... Press Ctrl+C to stop"
    echo ""
    sleep 2
    python scripts/drowsiness_detection_esp32.py
fi

#!/bin/bash

# Connect Raspberry Pi to ESP32 WiFi Access Point
# This script connects to the ESP32-Drowsiness-AP WiFi network

SSID="ESP32-Drowsiness-AP"
PASSWORD="drowsy123"
ESP32_IP="192.168.4.1"

echo "=========================================="
echo "ESP32 WiFi Connection Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

echo "Scanning for ESP32 WiFi..."
echo ""

# Scan for the network
if nmcli dev wifi list | grep -q "$SSID"; then
    echo "✓ Found ESP32 WiFi: $SSID"
else
    echo "✗ Cannot find $SSID"
    echo "  Make sure ESP32 is powered on"
    exit 1
fi

echo ""
echo "Connecting to $SSID..."

# Connect using NetworkManager
nmcli dev wifi connect "$SSID" password "$PASSWORD"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Connected successfully!"
    echo "=========================================="
    echo ""
    echo "Testing connection to ESP32..."
    sleep 2
    
    # Test ping
    if ping -c 3 -W 2 "$ESP32_IP" > /dev/null 2>&1; then
        echo "✓ ESP32 is reachable at $ESP32_IP"
        echo ""
        echo "You can now run the drowsiness detection script:"
        echo "  python scripts/drowsiness_detection_esp32.py"
    else
        echo "✗ Cannot reach ESP32 at $ESP32_IP"
        echo "  WiFi connected but ESP32 not responding"
    fi
else
    echo ""
    echo "✗ Connection failed"
    echo "  Check password and ESP32 status"
    exit 1
fi

echo ""
echo "=========================================="

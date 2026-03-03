#!/usr/bin/env python3
"""
Simple test script to verify ESP32 connection and send test commands
without needing the full drowsiness detection system.
"""

import sys
import time
import requests
import argparse


def test_esp32_connection(ip_address="192.168.4.1", port=80):
    """Test basic connectivity to ESP32"""
    print(f"\n{'='*50}")
    print(f"Testing ESP32 Connection")
    print(f"{'='*50}\n")
    
    base_url = f"http://{ip_address}:{port}"
    
    print(f"Target: {base_url}")
    print(f"Testing connectivity...")
    
    try:
        response = requests.get(base_url, timeout=3)
        print(f"✓ ESP32 is reachable (Status: {response.status_code})")
        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection failed - Cannot reach ESP32")
        print(f"\nTroubleshooting:")
        print(f"  1. Check ESP32 is powered on")
        print(f"  2. Verify WiFi connection to ESP32-Drowsiness-AP")
        print(f"  3. Ping test: ping {ip_address}")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Timeout - ESP32 not responding")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def send_test_command(ip_address, port, command):
    """Send a test command to ESP32"""
    url = f"http://{ip_address}:{port}/command"
    payload = {"command": command}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✓ Command sent: {command}")
            return True
        else:
            print(f"✗ Command failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ Command error: {e}")
        return False


def run_hardware_tests(ip_address, port):
    """Run comprehensive hardware tests"""
    print(f"\n{'='*50}")
    print(f"ESP32 Hardware Test Suite")
    print(f"{'='*50}\n")
    
    tests = [
        ("Buzzer - Low (30%)", "TEST:buzzer:30", 3),
        ("Buzzer - Medium (60%)", "TEST:buzzer:60", 3),
        ("Buzzer - High (90%)", "TEST:buzzer:90", 3),
        ("Vibrator - Low (30%)", "TEST:vibrator:30", 3),
        ("Vibrator - Medium (60%)", "TEST:vibrator:60", 3),
        ("Vibrator - High (90%)", "TEST:vibrator:90", 3),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, command, wait_time in tests:
        print(f"\n[Test] {test_name}")
        print(f"  Command: {command}")
        
        if send_test_command(ip_address, port, command):
            passed += 1
            print(f"  Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            failed += 1
            print(f"  Skipping wait due to error")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='Test ESP32 connection and hardware'
    )
    parser.add_argument(
        '--ip', 
        type=str, 
        default='192.168.4.1',
        help='ESP32 IP address (default: 192.168.4.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=80,
        help='ESP32 port (default: 80)'
    )
    parser.add_argument(
        '--test-hardware',
        action='store_true',
        help='Run full hardware test suite'
    )
    parser.add_argument(
        '--buzzer',
        type=int,
        metavar='INTENSITY',
        help='Test buzzer at specified intensity (0-100)'
    )
    parser.add_argument(
        '--vibrator',
        type=int,
        metavar='INTENSITY',
        help='Test vibrator at specified intensity (0-100)'
    )
    
    args = parser.parse_args()
    
    # Test connection first
    if not test_esp32_connection(args.ip, args.port):
        sys.exit(1)
    
    # Run requested tests
    if args.test_hardware:
        success = run_hardware_tests(args.ip, args.port)
        sys.exit(0 if success else 1)
    
    elif args.buzzer is not None:
        intensity = max(0, min(100, args.buzzer))
        print(f"\nTesting buzzer at {intensity}%...")
        send_test_command(args.ip, args.port, f"TEST:buzzer:{intensity}")
        print("Wait 3 seconds...")
        time.sleep(3)
        print("Done!")
    
    elif args.vibrator is not None:
        intensity = max(0, min(100, args.vibrator))
        print(f"\nTesting vibrator at {intensity}%...")
        send_test_command(args.ip, args.port, f"TEST:vibrator:{intensity}")
        print("Wait 3 seconds...")
        time.sleep(3)
        print("Done!")
    
    else:
        print("\n✓ Connection test passed!")
        print("\nNext steps:")
        print("  - Test hardware: python scripts/test_esp32.py --test-hardware")
        print("  - Test buzzer: python scripts/test_esp32.py --buzzer 50")
        print("  - Test vibrator: python scripts/test_esp32.py --vibrator 50")
        print("  - Run detection: python scripts/drowsiness_detection_esp32.py")


if __name__ == '__main__':
    main()

#include <WiFi.h>
#include <HTTPClient.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// WiFi Access Point credentials (for ESP32 to create its own network)
const char* ap_ssid = "ESP32-Drowsiness-AP";
const char* ap_password = "drowsy123";  // Minimum 8 characters

// Optional: WiFi Station credentials (to connect to existing network)
const char* sta_ssid = "";  // Leave empty to disable station mode
const char* sta_password = "";

// Backend server URL (optional - configure if needed)
const char* serverUrl = "http://192.168.4.2:3000/api/logs";  // Default AP IP is 192.168.4.1

// Pin definitions (adjust based on your wiring)
// 2 Buzzer pins (will scale like vibrators)
const int BUZZER_PINS[2] = {25, 33};  // GPIO pins for 2 buzzers

// 6 Vibrator motor pins
const int VIBRATOR_PINS[6] = {26, 27, 14, 12, 13, 15};  // GPIO pins for 6 vibrator motors

// LED Status Indicators
const int GREEN_LED_PIN = 2;    // GPIO pin for GREEN LED (connected)
const int RED_LED_PIN = 4;      // GPIO pin for RED LED (disconnected)

// PWM settings for intensity control
const int PWM_FREQUENCY = 5000;  // 5 KHz
const int PWM_RESOLUTION = 8;    // 8-bit resolution (0-255)

// BLE Service and Characteristic UUIDs
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define COMMAND_CHAR_UUID   "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define STATUS_CHAR_UUID    "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

// Device information
String deviceName = "ESP32-Drowsiness";
bool deviceConnected = false;
bool oldDeviceConnected = false;

// BLE objects
BLEServer* pServer = NULL;
BLECharacteristic* pCommandCharacteristic = NULL;
BLECharacteristic* pStatusCharacteristic = NULL;

// Control states
bool buzzerStates[2] = {false, false};
bool vibratorStates[6] = {false, false, false, false, false, false};

// Forward declarations
void handleCommand(String command);
void updateStatus();
void activateBuzzer(int intensity, int duration);
void activateVibrators(int intensity, int duration);
void setBuzzerIntensity(int intensity);
void setVibratorIntensity(int intensity);

// BLE Server Callbacks
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("\n=================================");
    Serial.println("[BLE] Client connected!");
    Serial.println("=================================\n");
    
    // Turn on GREEN LED, turn off RED LED
    digitalWrite(GREEN_LED_PIN, HIGH);
    digitalWrite(RED_LED_PIN, LOW);
    Serial.println("[LED] Status: GREEN ON (Connected)");
  };

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("\n=================================");
    Serial.println("[BLE] Client disconnected!");
    Serial.println("=================================\n");
    
    // Turn off GREEN LED, turn on RED LED
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(RED_LED_PIN, HIGH);
    Serial.println("[LED] Status: RED ON (Disconnected)");
    
    // Stop all outputs when disconnected
    for (int i = 0; i < 2; i++) {
      ledcWrite(BUZZER_PINS[i], 0);
      buzzerStates[i] = false;
    }
    for (int i = 0; i < 6; i++) {
      ledcWrite(VIBRATOR_PINS[i], 0);
      vibratorStates[i] = false;
    }
    
    // Restart advertising
    BLEDevice::startAdvertising();
    Serial.println("[BLE] Advertising restarted");
  }
};

// BLE Characteristic Callbacks - Handle incoming commands
class CommandCallbacks: public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String command = pCharacteristic->getValue().c_str();
    
    if (command.length() > 0) {
      Serial.println("\n=================================");
      Serial.println("[COMMAND RECEIVED]");
      Serial.println("=================================");
      Serial.printf("Raw command: %s\n", command.c_str());
      Serial.println("---------------------------------");
      
      handleCommand(command);
      
      Serial.println("=================================\n");
    }
  }
};

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n=================================");
  Serial.println("ESP32 Drowsiness Detection Controller");
  Serial.println("WiFi AP + BLE Mode");
  Serial.println("=================================\n");
  
  // Initialize WiFi Access Point
  Serial.println("Starting WiFi Access Point...");
  WiFi.mode(WIFI_AP);
  
  bool apStarted = WiFi.softAP(ap_ssid, ap_password);
  
  if (apStarted) {
    Serial.println("[WiFi AP] Access Point started successfully!");
    Serial.printf("Network Name (SSID): %s\n", ap_ssid);
    Serial.printf("Password: %s\n", ap_password);
    Serial.printf("IP Address: %s\n", WiFi.softAPIP().toString().c_str());
    Serial.printf("MAC Address: %s\n", WiFi.softAPmacAddress().c_str());
    Serial.println("Devices can now connect to this network!");
  } else {
    Serial.println("[WiFi AP] Failed to start Access Point!");
  }
  
  // Optional: Also connect to existing WiFi network (Station mode)
  if (strlen(sta_ssid) > 0) {
    Serial.println("\nAlso connecting to existing WiFi network...");
    WiFi.mode(WIFI_AP_STA);  // Both AP and Station
    WiFi.begin(sta_ssid, sta_password);
    
    int wifiAttempts = 0;
    while (WiFi.status() != WL_CONNECTED && wifiAttempts < 20) {
      delay(500);
      Serial.print(".");
      wifiAttempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n[WiFi STA] Connected to existing network!");
      Serial.printf("Station IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
      Serial.println("\n[WiFi STA] Connection failed, continuing with AP only...");
    }
  }
  
  Serial.println();
  
  // CRITICAL: Set pin modes FIRST before PWM
  for (int i = 0; i < 2; i++) {
    pinMode(BUZZER_PINS[i], OUTPUT);
    digitalWrite(BUZZER_PINS[i], LOW);
  }
  
  for (int i = 0; i < 6; i++) {
    pinMode(VIBRATOR_PINS[i], OUTPUT);
    digitalWrite(VIBRATOR_PINS[i], LOW);
  }
  
  // Initialize LED status pins
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  digitalWrite(GREEN_LED_PIN, LOW);  // Start with GREEN off
  digitalWrite(RED_LED_PIN, HIGH);   // Start with RED on (disconnected)
  
  Serial.println("GPIO pins set as OUTPUT");
  Serial.println("[LED] Initial state: RED ON (waiting for connection)");
  
  // Initialize PWM - ESP32 Arduino Core 3.0+ API
  for (int i = 0; i < 2; i++) {
    ledcAttach(BUZZER_PINS[i], PWM_FREQUENCY, PWM_RESOLUTION);
    ledcWrite(BUZZER_PINS[i], 0);
  }
  
  for (int i = 0; i < 6; i++) {
    ledcAttach(VIBRATOR_PINS[i], PWM_FREQUENCY, PWM_RESOLUTION);
    ledcWrite(VIBRATOR_PINS[i], 0);
  }
  
  Serial.println("GPIO pins initialized with PWM:");
  Serial.println("  - Buzzers:");
  for (int i = 0; i < 2; i++) {
    Serial.printf("    - Buzzer %d: GPIO %d\n", i + 1, BUZZER_PINS[i]);
  }
  Serial.println("  - Vibrators:");
  for (int i = 0; i < 6; i++) {
    Serial.printf("    - Vibrator %d: GPIO %d\n", i + 1, VIBRATOR_PINS[i]);
  }
  Serial.println("  - LED Status Indicators:");
  Serial.printf("    - GREEN LED (Connected): GPIO %d\n", GREEN_LED_PIN);
  Serial.printf("    - RED LED (Disconnected): GPIO %d\n", RED_LED_PIN);
  Serial.printf("  - PWM Frequency: %d Hz\n", PWM_FREQUENCY);
  Serial.printf("  - PWM Resolution: %d-bit (0-255)\n\n", PWM_RESOLUTION);
  
  // === IMMEDIATE HARDWARE TEST ===
  Serial.println("=== HARDWARE TEST ===");
  Serial.println("Testing all 2 buzzers sequentially...");
  for (int i = 0; i < 2; i++) {
    Serial.printf("Testing Buzzer %d on GPIO %d...\n", i + 1, BUZZER_PINS[i]);
    ledcWrite(BUZZER_PINS[i], 200);
    delay(500);
    ledcWrite(BUZZER_PINS[i], 0);
    delay(200);
  }
  
  Serial.println("Testing all 6 vibrators sequentially...");
  for (int i = 0; i < 6; i++) {
    Serial.printf("Testing Vibrator %d on GPIO %d...\n", i + 1, VIBRATOR_PINS[i]);
    ledcWrite(VIBRATOR_PINS[i], 200);
    delay(500);
    ledcWrite(VIBRATOR_PINS[i], 0);
    delay(200);
  }
  Serial.println("=== HARDWARE TEST COMPLETE ===\n");
  
  // Initialize BLE
  Serial.println("Initializing BLE...");
  BLEDevice::init(deviceName.c_str());
  
  // Create BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  
  // Create BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);
  
  // Create Command Characteristic (Write)
  pCommandCharacteristic = pService->createCharacteristic(
    COMMAND_CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  pCommandCharacteristic->setCallbacks(new CommandCallbacks());
  
  // Create Status Characteristic (Read & Notify)
  pStatusCharacteristic = pService->createCharacteristic(
    STATUS_CHAR_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pStatusCharacteristic->addDescriptor(new BLE2902());
  
  // Start the service
  pService->start();
  
  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMaxPreferred(0x12);
  BLEDevice::startAdvertising();
  
  Serial.println("\n=================================");
  Serial.println("BLE Server started!");
  Serial.printf("Device Name: %s\n", deviceName.c_str());
  Serial.println("Waiting for connections...");
  Serial.println("=================================\n");
}

void loop() {
  // Handle connection state changes
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
    // Send initial status when connected
    updateStatus();
  }
  
  if (!deviceConnected && oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
  
  delay(20);
}

// Command handler - parses and executes commands from BLE
void handleCommand(String command) {
  command.trim();
  
  // Command format: "TEST:buzzer:100" (intensity 0-100%) or "CONTROL:vibrator:on" or "ALERT:high"
  
  if (command.startsWith("TEST:")) {
    // Extract test parameters
    int firstColon = command.indexOf(':');
    int secondColon = command.indexOf(':', firstColon + 1);
    
    String testType = command.substring(firstColon + 1, secondColon);
    int intensity = command.substring(secondColon + 1).toInt();
    
    // Clamp intensity to 0-100%
    if (intensity < 0) intensity = 0;
    if (intensity > 100) intensity = 100;
    
    // Fixed duration of 3 seconds
    int duration = 3000;
    
    Serial.printf("Test Type: %s\n", testType.c_str());
    Serial.printf("Intensity: %d%% (PWM: %d/255)\n", intensity, map(intensity, 0, 100, 0, 255));
    Serial.printf("Duration: %d ms (fixed)\n", duration);
    Serial.println("---------------------------------");
    
    if (testType == "buzzer") {
      Serial.println("Testing BUZZER...");
      activateBuzzer(intensity, duration);
    }
    else if (testType == "vibrator") {
      Serial.println("Testing VIBRATORS...");
      activateVibrators(intensity, duration);
    }
    else if (testType == "both" || testType == "full") {
      Serial.println("Testing ALL devices...");
      activateBuzzer(intensity, duration);
      delay(100);
      activateVibrators(intensity, duration);
    }
    
    Serial.println("Test completed!");
    updateStatus();
  }
  else if (command.startsWith("CONTROL:")) {
    int firstColon = command.indexOf(':');
    int secondColon = command.indexOf(':', firstColon + 1);
    
    String device = command.substring(firstColon + 1, secondColon);
    String state = command.substring(secondColon + 1);
    
    bool turnOn = (state == "on" || state == "1" || state == "true");
    
    Serial.printf("Device: %s, State: %s\n", device.c_str(), turnOn ? "ON" : "OFF");
    
    if (device == "buzzer") {
      int intensity = turnOn ? 100 : 0;
      setBuzzerIntensity(intensity);
      Serial.printf("All buzzers turned %s\n", turnOn ? "ON" : "OFF");
    }
    else if (device == "vibrator") {
      int intensity = turnOn ? 100 : 0;
      setVibratorIntensity(intensity);
      Serial.printf("All vibrators turned %s\n", turnOn ? "ON" : "OFF");
    }
    
    updateStatus();
  }
  else if (command.startsWith("ALERT:")) {
    String level = command.substring(6);
    level.trim();
    
    Serial.printf("Alert Level: %s\n", level.c_str());
    Serial.println("---------------------------------");
    
    if (level == "low") {
      Serial.println("Low alert: Brief vibration");
      activateVibrators(50, 500);
    }
    else if (level == "medium") {
      Serial.println("Medium alert: Vibration + Buzzer");
      activateVibrators(75, 1000);
      delay(200);
      activateBuzzer(75, 500);
    }
    else if (level == "high") {
      Serial.println("High alert: Strong pattern");
      for (int i = 0; i < 3; i++) {
        activateVibrators(100, 500);
        activateBuzzer(100, 500);
        delay(200);
      }
    }
    
    Serial.println("Alert completed!");
    updateStatus();
  }
  else if (command == "STOP") {
    Serial.println("Stopping all devices...");
    for (int i = 0; i < 2; i++) {
      ledcWrite(BUZZER_PINS[i], 0);
      buzzerStates[i] = false;
    }
    for (int i = 0; i < 6; i++) {
      ledcWrite(VIBRATOR_PINS[i], 0);
      vibratorStates[i] = false;
    }
    Serial.println("All devices stopped");
    updateStatus();
  }
  else if (command == "STATUS") {
    Serial.println("Status requested");
    updateStatus();
  }
  else {
    Serial.printf("Unknown command: %s\n", command.c_str());
  }
}

// Update status characteristic
void updateStatus() {
  if (deviceConnected && pStatusCharacteristic != NULL) {
    bool anyBuzzerOn = false;
    for (int i = 0; i < 2; i++) {
      if (buzzerStates[i]) {
        anyBuzzerOn = true;
        break;
      }
    }
    bool anyVibratorOn = false;
    for (int i = 0; i < 6; i++) {
      if (vibratorStates[i]) {
        anyVibratorOn = true;
        break;
      }
    }
    String status = String(anyBuzzerOn ? "1" : "0") + "," + String(anyVibratorOn ? "1" : "0");
    pStatusCharacteristic->setValue(status.c_str());
    pStatusCharacteristic->notify();
    Serial.printf("[STATUS] Buzzers=%s, Vibrators=%s (notified)\n", 
                  anyBuzzerOn ? "ON" : "OFF", 
                  anyVibratorOn ? "ON" : "OFF");
  }
}

// Helper functions for device control with PWM intensity
void activateBuzzer(int intensity, int duration) {
  setBuzzerIntensity(intensity);
  Serial.printf("  > Buzzers activated for %d ms\n", duration);
  delay(duration);
  setBuzzerIntensity(0);
  Serial.println("  > Buzzers OFF");
}

void activateVibrators(int intensity, int duration) {
  setVibratorIntensity(intensity);
  Serial.printf("  > Vibrators activated for %d ms\n", duration);
  delay(duration);
  setVibratorIntensity(0);
  Serial.println("  > Vibrators OFF");
}

// Set buzzer intensity with progressive scaling (same as vibrators but for 2 buzzers)
void setBuzzerIntensity(int intensity) {
  // intensity: 0-100%
  // Determines how many buzzers are active and their PWM level
  
  int numActiveBuzzers = 0;
  int pwmValue = 0;
  
  if (intensity == 0) {
    numActiveBuzzers = 0;
    pwmValue = 0;
  } else if (intensity <= 50) {
    // 1-50%: Only 1 buzzer active with scaled PWM
    numActiveBuzzers = 1;
    pwmValue = map(intensity, 1, 50, 64, 200);  // Low to moderate
  } else {
    // 51-100%: Both buzzers active with high PWM
    numActiveBuzzers = 2;
    pwmValue = map(intensity, 51, 100, 200, 255);  // Moderate to maximum
  }
  
  Serial.printf("  > Setting %d/%d buzzers at intensity %d%% (PWM: %d/255)\n", 
                numActiveBuzzers, 2, intensity, pwmValue);
  
  // Activate the appropriate number of buzzers
  for (int i = 0; i < 2; i++) {
    if (i < numActiveBuzzers) {
      ledcWrite(BUZZER_PINS[i], pwmValue);
      buzzerStates[i] = true;
    } else {
      ledcWrite(BUZZER_PINS[i], 0);
      buzzerStates[i] = false;
    }
  }
}

// Set vibrator intensity with progressive scaling
void setVibratorIntensity(int intensity) {
  // intensity: 0-100%
  // Determines how many vibrators are active and their PWM level
  
  int numActiveVibrators = 0;
  int pwmValue = 0;
  
  if (intensity == 0) {
    numActiveVibrators = 0;
    pwmValue = 0;
  } else if (intensity <= 16) {
    numActiveVibrators = 1;
    pwmValue = map(intensity, 1, 16, 64, 150);  // Low-moderate
  } else if (intensity <= 33) {
    numActiveVibrators = 2;
    pwmValue = map(intensity, 17, 33, 64, 150);  // Low-moderate
  } else if (intensity <= 50) {
    numActiveVibrators = 3;
    pwmValue = map(intensity, 34, 50, 150, 200);  // Moderate
  } else if (intensity <= 66) {
    numActiveVibrators = 4;
    pwmValue = map(intensity, 51, 66, 200, 230);  // Moderate-high
  } else if (intensity <= 83) {
    numActiveVibrators = 5;
    pwmValue = map(intensity, 67, 83, 230, 255);  // High
  } else {
    numActiveVibrators = 6;
    pwmValue = 255;  // Maximum
  }
  
  Serial.printf("  > Setting %d/%d vibrators at intensity %d%% (PWM: %d/255)\n", 
                numActiveVibrators, 6, intensity, pwmValue);
  
  // Activate the appropriate number of vibrators
  for (int i = 0; i < 6; i++) {
    if (i < numActiveVibrators) {
      ledcWrite(VIBRATOR_PINS[i], pwmValue);
      vibratorStates[i] = true;
    } else {
      ledcWrite(VIBRATOR_PINS[i], 0);
      vibratorStates[i] = false;
    }
  }
}

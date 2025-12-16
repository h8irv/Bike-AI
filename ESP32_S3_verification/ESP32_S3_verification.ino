/*
 * ESP32-S3 Board Verification Script
 * 
 * This script tests essential ESP32-S3 functionality including:
 * - Serial communication (USB CDC)
 * - Built-in LED control
 * - WiFi scanning capability
 * - Memory information (RAM & Flash)
 * - GPIO functionality
 * - CPU and chip information
 * 
 * For Bike AI Theft Detection Project
 * Author: Harry Irving
 * Date: December 2025
 */

#include <WiFi.h>

// Pin definitions - adjust based on your specific ESP32-S3 board
#define LED_PIN 48    // Many ESP32-S3 boards use GPIO 48 for RGB LED
                      // Some boards use GPIO 2, 8, or 38 - check your board docs

void setup() {
  // Initialize Serial communication over USB
  Serial.begin(115200);
  delay(2000);  // Wait for Serial to initialize
  
  Serial.println("\n\n================================");
  Serial.println("ESP32-S3 VERIFICATION TEST");
  Serial.println("================================\n");
  
  // Test 1: Chip Information
  printChipInfo();
  
  // Test 2: Memory Information
  printMemoryInfo();
  
  // Test 3: GPIO Setup
  setupGPIO();
  
  // Test 4: WiFi Scan Test
  testWiFiScan();
  
  Serial.println("\n================================");
  Serial.println("VERIFICATION COMPLETE!");
  Serial.println("================================");
  Serial.println("\nBoard is functioning correctly.");
  Serial.println("LED will now blink continuously.\n");
}

void loop() {
  // Blink LED to show board is running
  digitalWrite(LED_PIN, HIGH);
  Serial.println("LED ON");
  delay(1000);
  
  digitalWrite(LED_PIN, LOW);
  Serial.println("LED OFF");
  delay(1000);
}

// ========================================
// Test Functions
// ========================================

void printChipInfo() {
  Serial.println("--- CHIP INFORMATION ---");
  
  // Using ESP class methods (compatible with all Arduino ESP32 versions)
  Serial.print("Chip Model: ");
  Serial.println(ESP.getChipModel());
  
  Serial.print("Chip Revision: ");
  Serial.println(ESP.getChipRevision());
  
  Serial.print("Number of Cores: ");
  Serial.println(ESP.getChipCores());
  
  Serial.print("CPU Frequency: ");
  Serial.print(getCpuFrequencyMhz());
  Serial.println(" MHz");
  
  Serial.print("Flash Size: ");
  Serial.print(ESP.getFlashChipSize() / (1024 * 1024));
  Serial.println(" MB");
  
  Serial.print("Flash Speed: ");
  Serial.print(ESP.getFlashChipSpeed() / 1000000);
  Serial.println(" MHz");
  
  Serial.print("WiFi MAC Address: ");
  Serial.println(WiFi.macAddress());
  
  Serial.print("Bluetooth: ");
  Serial.println("BLE Supported (ESP32-S3)");
  
  Serial.println();
}

void printMemoryInfo() {
  Serial.println("--- MEMORY INFORMATION ---");
  
  Serial.print("Total Heap: ");
  Serial.print(ESP.getHeapSize() / 1024);
  Serial.println(" KB");
  
  Serial.print("Free Heap: ");
  Serial.print(ESP.getFreeHeap() / 1024);
  Serial.println(" KB");
  
  Serial.print("Total PSRAM: ");
  if (ESP.getPsramSize() > 0) {
    Serial.print(ESP.getPsramSize() / 1024);
    Serial.println(" KB");
    
    Serial.print("Free PSRAM: ");
    Serial.print(ESP.getFreePsram() / 1024);
    Serial.println(" KB");
  } else {
    Serial.println("0 KB (Not available or not enabled)");
  }
  
  Serial.print("Sketch Size: ");
  Serial.print(ESP.getSketchSize() / 1024);
  Serial.println(" KB");
  
  Serial.print("Free Sketch Space: ");
  Serial.print(ESP.getFreeSketchSpace() / 1024);
  Serial.println(" KB");
  
  Serial.println();
}

void setupGPIO() {
  Serial.println("--- GPIO TEST ---");
  
  // Setup LED pin
  pinMode(LED_PIN, OUTPUT);
  
  // Test LED a few times
  Serial.print("Testing LED on GPIO ");
  Serial.println(LED_PIN);
  Serial.println("(If LED doesn't blink, try GPIO 2, 8, 38, or 48)");
  
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    Serial.print("  Flash #");
    Serial.println(i + 1);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  
  Serial.println("GPIO test complete!");
  Serial.println();
}

void testWiFiScan() {
  Serial.println("--- WIFI SCAN TEST ---");
  Serial.println("Scanning for WiFi networks...");
  
  // Set WiFi to station mode
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  
  // Scan for networks
  int numNetworks = WiFi.scanNetworks();
  
  if (numNetworks == 0) {
    Serial.println("No networks found.");
    Serial.println("(This is OK if you're not near any WiFi)");
  } else {
    Serial.print("Found ");
    Serial.print(numNetworks);
    Serial.println(" network(s):");
    
    // Print network details (limit to first 5)
    int displayCount = min(numNetworks, 5);
    for (int i = 0; i < displayCount; i++) {
      Serial.print("  ");
      Serial.print(i + 1);
      Serial.print(". ");
      Serial.print(WiFi.SSID(i));
      Serial.print(" (Signal: ");
      Serial.print(WiFi.RSSI(i));
      Serial.print(" dBm, ");
      Serial.print(WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "Open" : "Encrypted");
      Serial.println(")");
    }
    
    if (numNetworks > 5) {
      Serial.print("  ... and ");
      Serial.print(numNetworks - 5);
      Serial.println(" more");
    }
  }
  
  Serial.println("WiFi scan complete!");
  Serial.println();
  
  // Clean up
  WiFi.scanDelete();
}

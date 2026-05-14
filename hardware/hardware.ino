#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

const int         BUZZER_PIN      = 8;
const float       MOVE_THRESHOLD  = 0.08;
const unsigned long STOP_DELAY    = 2000;
const unsigned long STATUS_INTERVAL = 2000;  // resend START every 2 seconds

bool          isRunning     = false;
unsigned long lastMoveTime  = 0;
unsigned long buzzerEndTime = 0;
unsigned long lastStatusTime = 0;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    mpu.initialize();
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
}

void loop() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    float mag       = sqrt(pow(ax/16384.0,2) + pow(ay/16384.0,2) + pow(az/16384.0,2));
    float deviation = abs(mag - 1.0);

    if (deviation > MOVE_THRESHOLD) {
        lastMoveTime = millis();
        if (!isRunning) {
            Serial.println("START");
            isRunning    = true;
            lastStatusTime = millis();
        }
        // Keep resending START every 2 seconds so Python never misses it
        if (millis() - lastStatusTime > STATUS_INTERVAL) {
            Serial.println("START");
            lastStatusTime = millis();
        }
    } else if (isRunning && (millis() - lastMoveTime > STOP_DELAY)) {
        Serial.println("STOP");
        isRunning = false;
    }

    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "ALARM") {
            buzzerEndTime = millis() + 5000;
        }
    }

    digitalWrite(BUZZER_PIN, millis() < buzzerEndTime ? HIGH : LOW);

    delay(50);
}

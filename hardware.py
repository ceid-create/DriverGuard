# ============================================================
#  DriverGuard — Arduino Hardware Interface
#  Serial communication with Arduino:
#    - Receives START / STOP from MPU-6050 movement detection
#    - Sends ALARM to trigger the buzzer
#
#  Logic:
#    - Initially inactive (detection paused)
#    - START received  → detection starts immediately
#    - STOP received   → 3-second countdown begins
#                        if START arrives during countdown → cancelled
#                        if countdown expires → detection pauses
#    - No Arduino      → always active (software-only mode)
# ============================================================

import serial
import serial.tools.list_ports
import threading
import time

BAUD_RATE       = 9600
READ_TIMEOUT    = 1
STOP_GRACE      = 3.0   # seconds to wait after STOP before pausing detection


def _find_arduino_port():
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "cp210" in desc or "uart" in desc:
            return port.device
    return None


class ArduinoHardware:

    def __init__(self, port=None, baud=BAUD_RATE):
        self._active     = False   # starts inactive — waits for movement
        self._connected  = False
        self._ser        = None
        self._thread     = None
        self._stop_flag  = False

        target_port = port or _find_arduino_port()
        if target_port:
            try:
                self._ser = serial.Serial(target_port, baud, timeout=READ_TIMEOUT)
                time.sleep(2)
                self._ser.reset_input_buffer()
                self._connected = True
                print(f"[HW] Arduino connected on {target_port}")
                print(f"[HW] Waiting for vehicle movement...")
                self._thread = threading.Thread(target=self._listen, daemon=True)
                self._thread.start()
            except serial.SerialException as e:
                print(f"[HW] Could not open {target_port}: {e}")
                self._active = True
        else:
            print("[HW] No Arduino found — software-only mode (always active)")
            self._active = True

    def is_active(self):
        return self._active

    def is_connected(self):
        return self._connected

    def send_alarm(self):
        if self._connected and self._ser:
            try:
                self._ser.write(b"ALARM\n")
            except serial.SerialException:
                pass

    def disconnect(self):
        self._stop_flag = True
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _listen(self):
        stop_timer = None

        while not self._stop_flag:
            try:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode(errors="ignore").strip()

                    if line == "START":
                        self._active = True
                        stop_timer   = None   # cancel any pending stop countdown
                        print("[HW] Movement detected — detection started")

                    elif line == "STOP":
                        stop_timer = time.time()
                        print(f"[HW] No movement — pausing detection in {STOP_GRACE:.0f}s "
                              f"unless movement resumes")

                # Check if 3-second grace period has expired
                if stop_timer is not None and time.time() - stop_timer >= STOP_GRACE:
                    self._active = False
                    stop_timer   = None
                    print("[HW] Vehicle stopped — detection paused")

            except serial.SerialException:
                print("[HW] Serial connection lost")
                self._connected = False
                self._active    = True   # fail-safe: keep detection running
                break

            time.sleep(0.02)

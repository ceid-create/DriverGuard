# ============================================================
#  DriverGuard — Arduino Hardware Interface
#  Handles serial communication with the Arduino:
#    - Receives START / STOP from MPU-6050 movement detection
#    - Sends ALARM to trigger the buzzer
#
#  If no Arduino is detected, the system runs in software-only
#  mode (detection always active, no buzzer).
# ============================================================

import serial
import serial.tools.list_ports
import threading
import time

BAUD_RATE    = 9600
READ_TIMEOUT = 1   # seconds


def _find_arduino_port():
    """Auto-detect the Arduino's serial port."""
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "cp210" in desc or "uart" in desc:
            return port.device
    return None


class ArduinoHardware:
    """
    Manages the serial link to the Arduino.
    Call is_active() each frame to check if detection should run.
    Call send_alarm()  when an alert fires.
    Call disconnect()  on shutdown.
    """

    def __init__(self, port=None, baud=BAUD_RATE):
        self._active    = False   # True = car moving → detection runs
        self._connected = False
        self._ser       = None
        self._thread    = None
        self._stop_flag = False

        # Try to connect
        target_port = port or _find_arduino_port()
        if target_port:
            try:
                self._ser = serial.Serial(target_port, baud,
                                          timeout=READ_TIMEOUT)
                time.sleep(2)          # wait for Arduino reset after Serial open
                self._ser.reset_input_buffer()
                self._connected = True
                print(f"[HW] Arduino connected on {target_port}")
                self._thread = threading.Thread(target=self._listen,
                                                daemon=True)
                self._thread.start()
            except serial.SerialException as e:
                print(f"[HW] Could not open {target_port}: {e}")
        else:
            print("[HW] No Arduino found — running in software-only mode "
                  "(detection always active, no buzzer)")
            self._active = True     # always active when no hardware

    # ── Public API ────────────────────────────────────────────

    def is_active(self):
        """Returns True when the car is moving and detection should run."""
        return self._active

    def is_connected(self):
        return self._connected

    def send_alarm(self):
        """Tell the Arduino to activate the buzzer for 5 seconds."""
        if self._connected and self._ser:
            try:
                self._ser.write(b"ALARM\n")
            except serial.SerialException:
                pass

    def disconnect(self):
        self._stop_flag = True
        if self._ser and self._ser.is_open:
            self._ser.close()

    # ── Background serial reader ──────────────────────────────

    def _listen(self):
        while not self._stop_flag:
            try:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode(errors="ignore").strip()
                    if line == "START":
                        self._active = True
                        print("[HW] Vehicle moving — detection started")
                    elif line == "STOP":
                        self._active = False
                        print("[HW] Vehicle stopped — detection paused")
            except serial.SerialException:
                print("[HW] Serial connection lost")
                self._connected = False
                self._active    = True   # fail-safe: keep detection running
                break
            time.sleep(0.02)

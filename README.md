# DriverGuard

Real-time driver fatigue and distraction detection system built with Python and computer vision.
Detects drowsiness (eye closure), yawning, forward head drop, and phone usage while driving.
Optionally integrates with Arduino hardware (MPU-6050 + buzzer) for physical alerts and vehicle motion detection.

Made for our Semester 4 Computer Vision project at USJ (Spring 2026).

---

## Demo

https://drive.google.com/file/d/12vXjw39j9jb3cXyNDOe9Tk8lTDKoYL8N/view?usp=drive_link

---

## What it detects

| Alert | Method | Condition | Time gate |
|---|---|---|---|
| Eyes closed | EAR (Eye Aspect Ratio) | EAR < 0.25 sustained | 2 seconds |
| Yawning | MAR (Mouth Aspect Ratio) | MAR > 0.50 sustained | 1 second |
| Head drop forward | Nose displacement | Nose drop > 0.10 or face disappears | 2 seconds |
| Phone in hand | YOLOv8n + proximity filter | Detection near face | 3 seconds |

Each alert triggers: audio alarm (pygame), Gmail email notification, CSV log entry, and Arduino buzzer (if connected).

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Gmail credentials

Copy the template and fill in your details:

```bash
cp config.example.py config.py
```

Edit `config.py`:

```python
GMAIL_EXPEDITEUR   = "your_email@gmail.com"
GMAIL_MOT_PASSE    = "your_app_password"   # generate at myaccount.google.com/apppasswords
GMAIL_DESTINATAIRE = "recipient@gmail.com"
```

> `config.py` is gitignored — your credentials will never be pushed to the repository.

### 3. Download the YOLO weights

Download `best.pt` from the link below and place it at:

```
runs/detect/train3/weights/best.pt
```

**Weights download:** https://drive.google.com/file/d/1qrwW-OPN2WZ7KnbBJD9MT1T6RY_ZNChM/view?usp=drive_link

---

## How to run

### Software-only mode (no hardware required)

```bash
python detection_test.py
```

### Hardware mode (Arduino + MPU-6050 + buzzer)

Connect the Arduino before running (see wiring below):

```bash
python detection_hardware.py
```

In hardware mode, detection is **paused** until the MPU-6050 detects vehicle movement, and **resumes** automatically when the vehicle moves again.

Press `Q`, `ESC`, or click the **STOP** button to exit.

### Evaluation mode

Pass the `--eval` flag to either script to measure precision / recall / F1:

```bash
python detection_test.py --eval
python detection_hardware.py --eval
```

During the session, use the keyboard to mark ground-truth event windows:

| Key | Event |
|---|---|
| `E` | Eyes closed (open window, perform event, close window) |
| `Y` | Yawning |
| `F` | Forward head drop |
| `P` | Phone in hand |

Precision / Recall / F1 are printed at the end and saved to `logs/eval_*.csv`.

---

## Hardware (optional — Arduino + MPU-6050)

When the Arduino is connected, the system:
- **Waits** until the MPU-6050 detects vehicle movement before starting detection
- **Pauses** detection 3 seconds after movement stops (car parked)
- **Activates the buzzer** for 5 seconds on every alert

If no Arduino is connected, the system runs in software-only mode (always active, no buzzer).

### Wiring

```
MPU-6050 VCC  →  Arduino 3.3V
MPU-6050 GND  →  Arduino GND
MPU-6050 SDA  →  Arduino A4
MPU-6050 SCL  →  Arduino A5
Buzzer (+)    →  100Ω resistor  →  Arduino Pin 8
Buzzer (−)    →  Arduino GND
Arduino USB   →  Laptop/PC
```

Upload `hardware/hardware.ino` to the Arduino before running.

---

## YOLO Phone Detection Model

Fine-tuned YOLOv8n (Ultralytics) on a custom Roboflow phone detection dataset.

**Training configuration:**

| Parameter | Value |
|---|---|
| Base model | YOLOv8n (pretrained on COCO) |
| Epochs | 60 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Device | CPU |

**Best validation results (epoch 57):**

| Metric | Value |
|---|---|
| Precision | 88.0% |
| Recall | 83.3% |
| mAP@0.5 | **87.97%** |
| mAP@0.5–0.95 | 71.7% |

**Dataset:** phone-detection-1jjzq-npgus v1 — Roboflow workspace `joes-workspace-lgbro` (CC BY 4.0)
3,178 labelled images — Train / Val / Test: 2,934 / 122 / 122 (~92% / 4% / 4%)

**Weights download:** https://drive.google.com/file/d/1qrwW-OPN2WZ7KnbBJD9MT1T6RY_ZNChM/view?usp=drive_link

**To reproduce training from scratch:**

```bash
yolo detect train model=yolov8n.pt data=data/data.yaml epochs=60 imgsz=640 batch=16
```

---

## Live Evaluation Results

Evaluated using `python detection_test.py --eval` on the live running system.

| Alert type | Precision | Recall | F1 |
|---|---|---|---|
| YEUX_FERMES | 92% | 86% | 89% |
| BAILLEMENT | 86% | 80% | 83% |
| TETE_AVANT | 100% | 73% | 84% |
| TELEPHONE | 88% | 70% | 78% |
| **Macro avg** | **91%** | **77%** | **83%** |

---

## Project structure

```
DriverGuard/
├── detection_test.py       # main entry point — software-only (no hardware)
├── detection_hardware.py   # main entry point — with Arduino hardware
├── hardware.py             # Arduino serial interface (MPU-6050 + buzzer)
├── hardware/
│   └── hardware.ino        # Arduino sketch (upload to the board)
├── detection/
│   ├── fatigue.py          # EAR, MAR, head drop, blink counter, fatigue score
│   ├── phone.py            # YOLOv8 wrapper, proximity filter, temporal smoothing
│   └── landmarks.py        # MediaPipe landmark index constants
├── alerts/
│   ├── audio.py            # pygame alarm
│   ├── email_alert.py      # Gmail SMTP alerts
│   └── logger.py           # CSV session logger
├── ui/
│   └── dashboard.py        # OpenCV drawing functions
├── evaluate.py             # offline evaluation script (video + ground truth CSV)
├── evaluate_simple.py      # simplified frame-based evaluator (no YOLO)
├── record_eval.py          # evaluation video recorder with key-press ground truth
├── ablation.py             # threshold ablation study (sweeps EAR / MAR / angle)
├── config.py               # credentials (gitignored — never committed)
├── config.example.py       # credentials template
├── alarme.mp3              # alarm sound file
├── requirements.txt        # Python dependencies
└── logs/                   # session logs (auto-created, gitignored)
```

---

## Tech stack

| Library | Role |
|---|---|
| Python 3.11 | Language |
| OpenCV ≥ 4.13 | Video capture and UI rendering |
| MediaPipe 0.10.9 | 468-point face landmark detection |
| Ultralytics YOLOv8 ≥ 8.4 | Phone object detection |
| Pygame ≥ 2.6 | Alarm audio playback |
| pyserial ≥ 3.5 | Arduino serial communication |
| smtplib (stdlib) | Email alerts via Gmail SMTP |

---

## Notes

- `config.py` is gitignored — credentials are never pushed
- `logs/` is gitignored
- YOLO weights (`*.pt`) are gitignored — download via the link above
- Works best with good frontal lighting
- EAR detection may be affected by thick eyeglass frames
- Camera index 1 = external USB camera; change to 0 for the built-in webcam

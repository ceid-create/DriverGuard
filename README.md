# DriverGuard

Real-time driver fatigue and distraction detection system built with Python and computer vision. Detects drowsiness, yawning, forward head drop, and phone usage while driving. Includes Arduino hardware integration for physical buzzer alerts and vehicle motion detection.

Made for our Semester 4 Computer Vision project at USJ (Spring 2026).

---

## Demo

https://drive.google.com/file/d/12vXjw39j9jb3cXyNDOe9Tk8lTDKoYL8N/view?usp=drive_link

---

## What it detects

| Alert | Condition | Threshold |
|---|---|---|
| Eyes closed | EAR < 0.25 sustained | 2 seconds |
| Yawning | MAR > 0.50 sustained | 1 second |
| Head drop forward | Nose drop > 0.10 or face disappears | 2 seconds |
| Phone in hand | YOLOv8n detection near face | 3 seconds |

Each alert triggers: audio alarm, Gmail email notification, CSV log entry, and Arduino buzzer (if connected).

---

## How to run

Install dependencies:
```bash
pip install -r requirements.txt
```

Copy the config template and fill in your Gmail credentials:
```bash
cp config.example.py config.py
```

Edit `config.py`:
```python
GMAIL_EXPEDITEUR   = "your_email@gmail.com"
GMAIL_MOT_PASSE    = "your_app_password"   # generate at myaccount.google.com/apppasswords
GMAIL_DESTINATAIRE = "recipient@gmail.com"
```

Download the YOLO weights (see section below) and place at:
`runs/detect/train3/weights/best.pt`

Run:
```bash
python detection.py
```

Press `Q`, `ESC`, or click **STOP** to exit.

### Evaluation mode
```bash
python detection.py --eval
```
Press `E`/`Y`/`F`/`P` to mark event windows. Precision/recall/F1 printed at the end.

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
Buzzer (+)    →  100Ω  →  Arduino Pin 8
Buzzer (−)    →  Arduino GND
Arduino USB   →  Laptop/PC
```

---

## YOLO Phone Detection Model

Trained using YOLOv8n (Ultralytics) fine-tuned on a custom phone detection dataset.

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

To reproduce training:
```bash
yolo detect train model=yolov8n.pt data=data/data.yaml epochs=60 imgsz=640 batch=16
```

---

## Live Evaluation Results

Evaluated using `python detection.py --eval` on the live running system.

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
├── detection.py            # main entry point
├── hardware.py             # Arduino serial interface (optional)
├── detection/
│   ├── fatigue.py          # EAR, MAR, head drop, blink counter, fatigue score
│   ├── phone.py            # YOLOv8 wrapper, proximity filter, temporal smoothing
│   └── landmarks.py        # MediaPipe landmark indices
├── alerts/
│   ├── audio.py            # pygame alarm
│   ├── email_alert.py      # Gmail SMTP alerts
│   └── logger.py           # CSV session logger
├── ui/
│   └── dashboard.py        # OpenCV drawing functions
├── evaluate.py             # offline evaluation script
├── evaluate_simple.py      # simplified frame-based evaluator
├── record_eval.py          # evaluation video recorder
├── config.py               # credentials (gitignored)
├── config.example.py       # credentials template
├── alarme.mp3              # alarm sound
├── requirements.txt
└── logs/                   # session logs (auto-created, gitignored)
```

---

## Tech stack

| Library | Role |
|---|---|
| Python 3.11 | Language |
| OpenCV ≥ 4.13 | Video capture and UI |
| MediaPipe 0.10.9 | 468-point face landmark detection |
| Ultralytics YOLOv8 ≥ 8.4 | Phone object detection |
| Pygame ≥ 2.6 | Alarm audio |
| pyserial ≥ 3.5 | Arduino serial communication |
| smtplib (stdlib) | Email alerts |

---

## Notes

- `config.py` is gitignored — credentials are never pushed
- `logs/` is gitignored
- YOLO weights (`*.pt`) are gitignored — download via link above
- Works best with good frontal lighting
- EAR detection may be affected by thick eyeglass frames

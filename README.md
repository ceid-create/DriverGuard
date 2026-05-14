# DriverGuard

Real-time driver monitoring system built with Python and computer vision.
Detects drowsiness, yawning, head tilting, and phone usage while driving.

Made for our Semester 4 CV project at USJ (Spring 2026).

---

## What it does

The system uses the webcam to analyze the driver's face in real time:

- **Eyes closed too long** → alarm triggers after 2 seconds (EAR-based)
- **Yawning detected** → alarm after 1 second (MAR-based)
- **Head tilting sideways** → alert after 2 seconds (> 20 degrees)
- **Head dropping forward** → detects microsleep onset (nose drop > 0.08, 2.5s)
- **Phone in hand** → YOLOv8n detects it, alarm after 3 seconds
- **Fatigue score** → live 0–100% score combining all signals
- **Email alert** → sends a Gmail notification automatically
- **CSV log** → every alert is saved with timestamp and values

---

## How to run it

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

Copy the config template and add your Gmail credentials:

```bash
cp config.example.py config.py
```

Edit `config.py`:

```python
GMAIL_EXPEDITEUR   = "your_email@gmail.com"
GMAIL_MOT_PASSE    = "your_app_password"   # generate at myaccount.google.com/apppasswords
GMAIL_DESTINATAIRE = "recipient@gmail.com"
```

Download the YOLO weights (see section below) and place them at:
`runs/detect/train3/weights/best.pt`

Run:

```bash
python main.py
```

Press `Q` or click **STOP** on the sidebar to exit.

---

## Evaluation

To measure detection accuracy on a recorded video:

```bash
# 1. Generate a ground truth template
python evaluate.py --make-gt ground_truth.csv

# 2. Fill in ground_truth.csv with your event timestamps (open it in any text editor)

# 3. Run evaluation
python evaluate.py --video eval_video.mp4 --gt ground_truth.csv
```

To run the threshold ablation study:

```bash
python ablation.py --video eval_video.mp4 --gt ground_truth.csv
```

---

## YOLO Phone Detection Model

The model was trained using YOLOv8n (Ultralytics) on a custom phone detection dataset.

**Training configuration:**
| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8n (pretrained on COCO) |
| Epochs | 60 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | Auto (AdamW) |
| Device | CPU |

**Best validation results (epoch 57):**
| Metric | Value |
|--------|-------|
| Precision | 88.0% |
| Recall | 83.3% |
| mAP@0.5 | **87.97%** |
| mAP@0.5–0.95 | 71.7% |

**Dataset:** <!-- TODO: add dataset name, source URL, number of images, train/val/test split -->

**Weights download:** <!-- TODO: add Google Drive / HuggingFace link to best.pt -->

To reproduce training from scratch:

```bash
yolo detect train model=yolov8n.pt data=data/data.yaml epochs=60 imgsz=640 batch=16
```

---

## Project structure

```
DriverGuard/
├── main.py                 # entry point — camera loop and orchestration
├── detection/
│   ├── fatigue.py          # EAR, MAR, head angle, blink counter, fatigue score
│   ├── phone.py            # YOLOv8 wrapper, face-proximity filter, temporal smoothing
│   └── landmarks.py        # MediaPipe landmark indices and geometry helpers
├── alerts/
│   ├── audio.py            # pygame alarm
│   ├── email_alert.py      # Gmail SMTP alerts
│   └── logger.py           # CSV session logger
├── ui/
│   └── dashboard.py        # all OpenCV drawing functions
├── evaluate.py             # offline evaluation script (precision/recall/F1)
├── ablation.py             # threshold ablation study
├── config.py               # your credentials (gitignored)
├── config.example.py       # credentials template
├── alarme.mp3              # alarm sound
├── requirements.txt
└── logs/                   # session logs (auto-created, gitignored)
```

The YOLO model weights go in `runs/detect/train3/weights/best.pt` (not in git due to file size).

---

## Tech stack

| Library | Version | Role |
|---------|---------|------|
| Python | 3.11 | Language |
| OpenCV | ≥ 4.13 | Video capture and UI |
| MediaPipe | 0.10.9 | 468-point face landmark detection |
| Ultralytics YOLOv8 | ≥ 8.4 | Phone object detection |
| Pygame | ≥ 2.6 | Alarm audio |
| smtplib | stdlib | Email alerts |

---

## Notes

- `config.py` is gitignored so credentials are never pushed
- `logs/` is gitignored
- Tested on a laptop webcam; works best with decent lighting and a frontal face angle
- Known limitation: EAR-based detection breaks with sunglasses or heavy eyeglass frames

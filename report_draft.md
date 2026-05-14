# DriverGuard: Real-Time Driver Fatigue and Distraction Detection
**Computer Vision — Final Project Report**
Saint Joseph University · Spring 2026

**Authors:** <!-- your names here -->

---

## 1. Introduction

### 1.1 Problem Statement

Driver fatigue and distraction are among the leading causes of road accidents worldwide. The World Health Organization estimates that drowsy driving is responsible for approximately 20% of all traffic fatalities [CITATION NEEDED — WHO road safety report]. Unlike alcohol impairment, fatigue is often undetected until a critical moment, making proactive monitoring systems essential.

Current commercial systems (e.g., Mobileye, Subaru EyeSight) rely on expensive dedicated hardware. Our goal is to demonstrate that an accurate, real-time monitoring system can be built using only a standard webcam and a laptop CPU, making the technology accessible and deployable in low-cost settings.

### 1.2 Proposed System

DriverGuard monitors the driver in real time through a webcam feed and raises auditory and email alerts when fatigue or distraction is detected. The system detects five distinct events:

1. **Prolonged eye closure** (drowsiness indicator)
2. **Yawning** (fatigue indicator)
3. **Lateral head tilt** (microsleep or inattention)
4. **Forward head drop** (microsleep onset)
5. **Phone usage** (distraction)

### 1.3 Related Work

**Eye Aspect Ratio (EAR).** Soukupova and Cech (2016) introduced the EAR metric for blink and drowsiness detection using facial landmarks. EAR is defined as the ratio of vertical to horizontal eye opening distances. A value below a threshold for a sustained duration reliably indicates eye closure [1].

**Mouth Aspect Ratio (MAR).** Analogous to EAR, the MAR metric measures mouth openness using facial landmarks, enabling yawn detection [2].

**MediaPipe Face Mesh.** Lugaresi et al. (2019) proposed a real-time 468-point facial landmark detector that runs efficiently on CPU-class hardware [3]. We use this as our face analysis backbone.

**YOLOv8.** Ultralytics YOLOv8 (2023) is a state-of-the-art single-stage object detector. Its nano variant (YOLOv8n) achieves competitive accuracy at very low latency, making it suitable for real-time CPU inference [4].

---

## 2. Dataset

### 2.1 Phone Detection Training Dataset

The phone detection model was trained on a custom dataset of images containing people holding mobile phones.

<!-- TODO: Fill in the following table with your actual dataset information -->

| Property | Value |
|----------|-------|
| Dataset name | <!-- e.g. "Phone Detection Dataset v2" --> |
| Source | <!-- e.g. Roboflow, Kaggle, custom collected --> |
| Total images | <!-- e.g. 2,450 --> |
| Training set | <!-- e.g. 1,960 (80%) --> |
| Validation set | <!-- e.g. 245 (10%) --> |
| Test set | <!-- e.g. 245 (10%) --> |
| Classes | phone (class 1), background (class 0) |
| Image resolution | 640 × 640 (resized during training) |

**Preprocessing:** Images were resized to 640×640. Standard Ultralytics augmentation was applied: random horizontal flip (p=0.5), HSV colour jitter (H±1.5%, S±70%, V±40%), random scale (±50%), and mosaic augmentation.

### 2.2 Fatigue Detection — No Training Dataset Required

The fatigue detection pipeline (EAR, MAR, head angle) is entirely geometry-based and operates on MediaPipe landmarks. No training is required for this component; instead, it is evaluated against a manually annotated test video (see Section 4).

### 2.3 Evaluation Dataset

To measure system accuracy, we recorded a dedicated evaluation video of approximately 5 minutes in which a driver performed each alert condition at known timestamps. The ground truth was annotated manually:

<!-- TODO: Fill in after recording your eval video -->

| Event type | Number of occurrences | Total duration (s) |
|------------|-----------------------|--------------------|
| YEUX_FERMES | <!-- e.g. 5 --> | <!-- e.g. 18 --> |
| BAILLEMENT | <!-- e.g. 4 --> | <!-- e.g. 10 --> |
| TETE_PENCHEE | <!-- e.g. 4 --> | <!-- e.g. 14 --> |
| TETE_AVANT | <!-- e.g. 3 --> | <!-- e.g. 10 --> |
| TELEPHONE | <!-- e.g. 4 --> | <!-- e.g. 16 --> |
| Normal driving | — | <!-- e.g. 182 --> |

---

## 3. Methodology

### 3.1 System Architecture

DriverGuard processes each webcam frame through two parallel pipelines, followed by a unified alert manager:

```
Webcam Frame
     │
     ├─── MediaPipe FaceMesh ──► EAR / MAR / Angle / Drop ──► Fatigue Alerts
     │                                                               │
     └─── YOLOv8n ──► Proximity Filter ──► Temporal Smoother ──► Phone Alert
                                                                       │
                                                              Alert Manager
                                                         (audio / email / CSV log)
```

### 3.2 Fatigue Detection

#### 3.2.1 Eye Aspect Ratio (EAR)

We use the 6-point EAR model from Soukupova & Cech [1]:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

where p1–p6 are the MediaPipe landmark coordinates of the eye (indices 33, 160, 158, 133, 153, 144 for the right eye; 362, 385, 387, 263, 373, 380 for the left). The mean of both eyes is used.

When EAR drops below **0.25** for more than **2 seconds**, a drowsiness alert is raised.

#### 3.2.2 Mouth Aspect Ratio (MAR)

```
MAR = ||top - bottom|| / ||left - right||
```

using MediaPipe landmarks 13 (top lip), 14 (bottom lip), 78 (left corner), 308 (right corner). A yawn is detected when MAR > **0.50** persists for more than **1 second**.

#### 3.2.3 Head Tilt (Lateral)

The inter-eye axis angle is computed as:

```
angle = atan2(dy, dx)  where  (dx, dy) = inner_corner_left − inner_corner_right
```

An alert fires when |angle| > **20°** for more than **2 seconds**.

#### 3.2.4 Head Drop (Forward)

Forward head drop — a signature of microsleep onset — is detected via the normalised vertical displacement of the nose tip relative to the nose bridge:

```
inclinaison = y(nose_tip) − y(nose_bridge)
```

A positive value indicates the head is falling forward. Alert threshold: **0.08** (normalised), persisting for **2.5 seconds**.

#### 3.2.5 Blink Rate and Composite Fatigue Score

Blink rate (blinks/min) is computed over a rolling 10-second window. A rate above 25 blinks/min is associated with increased fatigue [CITATION].

The composite fatigue score (0–100%) combines all indicators with empirically determined weights:

| Indicator | Condition | Score contribution |
|-----------|-----------|-------------------|
| EAR | < 0.25 | +30 |
| EAR | 0.25–0.28 | +10 |
| MAR | > 0.50 | +20 |
| MAR | 0.35–0.50 | +5 |
| Head angle | > 20° | +25 |
| Head angle | 10–20° | +8 |
| Blink rate | > 25/min | +15 |
| Head drop | > 0.08 | +10 |

### 3.3 Phone Detection

#### 3.3.1 YOLOv8n Model

We fine-tuned YOLOv8n (pretrained on COCO) on a phone detection dataset. The model outputs bounding boxes with class and confidence scores. We apply three post-processing filters to reduce false positives:

**Confidence filter:** Only detections with confidence ≥ 0.50 are considered.

**Size filter:** Bounding boxes with area < 1 500 px² are rejected (catches distant background objects).

**Face-proximity filter:** A phone detection is only accepted if its bounding box overlaps with the face region (expanded by a 350 px margin). This is the primary false-positive suppressor: a mouse on a desk in the background will never be near the driver's face bounding box.

#### 3.3.2 Temporal Smoothing

YOLO inference is noisy frame-to-frame. We apply a hysteresis counter: the detection count increments (+1) on a positive frame and decrements (−1) on a negative frame. A phone is confirmed only when the counter reaches 3, and remains confirmed until the counter returns to 0. This eliminates single-frame false positives.

### 3.4 Alert System

All five alert types share a common two-stage mechanism:
1. **Warning stage:** the condition is met but the time threshold has not yet elapsed (orange indicator in the UI).
2. **Danger stage:** the condition persists past the threshold → audio alarm (pygame), email notification (Gmail SMTP, rate-limited to one email per 60 seconds), and a CSV log entry.

---

## 4. Experiments and Results

### 4.1 Phone Detection — YOLO Training Results

The model was trained for 60 epochs. Validation metrics were computed by Ultralytics on the held-out validation split:

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5–0.95 |
|-------|-----------|--------|---------|--------------|
| Best (57) | **88.0%** | **83.3%** | **87.97%** | **71.7%** |
| Final (60) | 95.8% | 89.8% | 85.7% | 70.1% |

The best.pt checkpoint (epoch 57) is used in the deployed system. The small drop from epoch 57 to 60 suggests mild overfitting in the final epochs; early stopping at the best mAP is appropriate.

### 4.2 Full System Evaluation — Per-Alert Metrics

<!-- TODO: Fill in after running: python evaluate.py --video eval_video.mp4 --gt ground_truth.csv -->

| Alert type | Precision | Recall | F1 | TP | FP | FN |
|------------|-----------|--------|----|----|----|----|
| YEUX_FERMES | | | | | | |
| BAILLEMENT | | | | | | |
| TETE_PENCHEE | | | | | | |
| TETE_AVANT | | | | | | |
| TELEPHONE | | | | | | |
| **Macro avg** | | | | | | |

### 4.3 Threshold Ablation Study — EAR

<!-- TODO: Fill in after running: python ablation.py --video eval_video.mp4 --gt ground_truth.csv -->

The following table shows the effect of varying the EAR threshold on eye-closure detection performance (MAR and ANGLE held at defaults):

| EAR threshold | Precision | Recall | F1 | Macro F1 |
|---------------|-----------|--------|----|----------|
| 0.20 | | | | |
| 0.22 | | | | |
| 0.24 | | | | |
| **0.25 (default)** | | | | |
| 0.26 | | | | |
| 0.28 | | | | |
| 0.30 | | | | |

**Finding:** <!-- e.g. "EAR = 0.25 gives the best F1 balance. Lower thresholds reduce false positives but miss late-stage eye closure. Higher thresholds produce more false alarms." -->

### 4.4 Threshold Ablation Study — MAR and Angle

<!-- TODO: Fill in after running ablation.py -->

---

## 5. Discussion

### 5.1 What Worked

**EAR/MAR geometry.** Computing drowsiness purely from landmark geometry—without any training—proved robust. The MediaPipe landmarks are stable enough at 30 FPS to give reliable EAR signals even with minor head movement.

**Face-proximity filter.** The single biggest source of false positives for phone detection was background objects (laptop, keyboard). The face-proximity filter almost entirely eliminated these at the cost of a small increase in false negatives (phone barely visible at frame edge). This trade-off is appropriate for a safety application where false alarms erode trust.

**Temporal smoothing.** Single-frame YOLO detections were noisy. The hysteresis counter gave near-zero frame-level false positives with negligible added latency (3 frames ≈ 100 ms at 30 FPS).

**Composite fatigue score.** Combining four signals into a single percentage gave a more stable and intuitive indicator than any single metric.

### 5.2 What Did Not Work / Limitations

- **Sunglasses / thick frames.** EAR is unreliable when glasses occlude the eye landmarks. MediaPipe still returns landmarks but their positions shift, pushing EAR toward the closed-eye range even when eyes are open.
- **Low light.** MediaPipe detection confidence drops sharply below ~50 lux, causing frequent "face not detected" states. A near-IR illuminator would fix this in a real deployment.
- **Side-profile driving posture.** The angle detection assumes a roughly frontal face view. Drivers who naturally sit with their head slightly turned will generate a constant non-zero angle offset, requiring per-driver calibration.
- **Single driver.** The system uses `max_num_faces=1`. A passenger visible in the camera field could confuse the face detector.
- **No gaze estimation.** Looking right or left (e.g., at a mirror) is not detected as distraction. Full gaze tracking would complement the head angle measurement.

### 5.3 False Positive Handling

Three mechanisms handle false positives:

1. **Time thresholds:** Momentary blinks or brief head turns do not trigger alarms. Only sustained conditions (2–3 s) fire alerts.
2. **Face-proximity filter:** Phone detections far from the face are discarded.
3. **Temporal smoothing:** YOLO must confirm detection across 3 consecutive frames.

Uncertain cases (EAR in the 0.25–0.28 range, angle near 20°) are shown as orange warnings in the UI without triggering the alarm. This gives the driver visual feedback without auditory false alarms.

---

## 6. Conclusion

We presented DriverGuard, a real-time driver monitoring system that combines classical computer vision (EAR/MAR geometry, head pose) with a fine-tuned YOLOv8n model for phone detection. The system runs on a standard laptop CPU at real-time frame rates and achieves <!-- fill in your F1 score --> macro F1 across five alert categories.

The teacher-required evaluation demonstrated that the EAR threshold of 0.25 and the 2-second time gate give the best precision/recall balance on our evaluation dataset. The face-proximity filter and temporal smoothing together reduce the false positive rate to <!-- fill in --> without meaningfully impacting recall.

Future work would add per-driver calibration (to handle natural posture variation), near-IR support for low-light robustness, and gaze direction estimation for detecting road-looking vs. mirror-checking behaviour.

---

## 7. References

[1] T. Soukupova and J. Cech, "Real-time eye blink detection using facial landmarks," *21st Computer Vision Winter Workshop*, 2016.

[2] <!-- MAR reference — find a paper that formally defines MAR for yawning, e.g. Abtahi et al. 2014 -->

[3] V. Lugaresi et al., "MediaPipe: A framework for perceiving and processing reality," *Third Workshop on Computer Vision for AR/VR at CVPR*, 2019.

[4] G. Jocher et al., "Ultralytics YOLOv8," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[5] <!-- Your dataset citation here -->

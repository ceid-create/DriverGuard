# ============================================================
#  DriverGuard — Simple Offline Evaluator
#  Uses frame-count timing (not wall clock) so results match
#  the live system regardless of CPU speed.
#  Evaluates: YEUX_FERMES, BAILLEMENT, TETE_AVANT
#  Phone detection: test live with detection.py (YOLO is too
#  slow on CPU to evaluate accurately in offline mode).
#
#  Usage:
#    python evaluate_simple.py --video eval_video.mp4 --gt ground_truth.csv
# ============================================================

import cv2
import math
import csv
import argparse
import os
from datetime import datetime

# ---- Thresholds (must match detection.py) ----
SEUIL_EAR         = 0.25
SEUIL_MAR         = 0.50
SEUIL_INCLINAISON = 0.15
SEUIL_TEMPS_YEUX  = 2.0   # seconds
SEUIL_TEMPS_BOUCHE = 1.0
SEUIL_TEMPS_AVANT  = 2.0
TOLERANCE          = 2.0  # seconds of matching tolerance

ALERT_TYPES = ["YEUX_FERMES", "BAILLEMENT", "TETE_AVANT"]

# ---- Landmark indices ----
OEIL_GAUCHE = [362, 385, 387, 263, 373, 380]
OEIL_DROIT  = [33,  160, 158, 133, 153, 144]
BOUCHE      = [13, 14, 78, 308]
NEZ_BOUT    = 4
NEZ_RACINE  = 6

def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def ear(pts, idx):
    p = [pts[i] for i in idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]))

def mar(pts):
    w = dist(pts[BOUCHE[2]], pts[BOUCHE[3]])
    return dist(pts[BOUCHE[0]], pts[BOUCHE[1]]) / w if w > 0 else 0.0

def inclinaison(pts):
    return pts[NEZ_BOUT].y - pts[NEZ_RACINE].y


def load_gt(path):
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            atype = parts[0].upper()
            if atype not in ALERT_TYPES:
                continue
            events.append({"type": atype, "start": float(parts[1]),
                           "end": float(parts[2]), "matched": False})
    return events


def run(video_path, gt_path):
    import mediapipe as mp
    mp_face  = mp.solutions.face_mesh
    detector = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.3,
                                 min_tracking_confidence=0.3)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_events    = load_gt(gt_path)

    print(f"\n[EvalSimple] {os.path.basename(video_path)}")
    print(f"[EvalSimple] {total_frames} frames @ {fps:.0f} FPS  "
          f"= {total_frames/fps:.1f}s")
    print(f"[EvalSimple] GT events loaded: {len(gt_events)}")
    print(f"[EvalSimple] Thresholds — EAR:{SEUIL_EAR}  MAR:{SEUIL_MAR}  "
          f"INCL:{SEUIL_INCLINAISON}\n")

    # ---- Per-alert state (frame-count based) ----
    timer_start  = {k: None for k in ALERT_TYPES}  # frame index when condition began
    active       = {k: False for k in ALERT_TYPES}
    fired        = []   # {"type", "start_s", "end_s"}

    # Face-hold: keep last known values for up to N frames when face is lost
    HOLD = 6
    no_face = 0
    last_ear_val = 0.30
    last_mar_val = 0.0
    last_incl_val = 0.0

    frames_detected = 0
    min_ear = 1.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps   # video time in seconds (frame-count based, not wall clock)
        frame_idx += 1

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.multi_face_landmarks:
            pts          = results.multi_face_landmarks[0].landmark
            ear_val      = (ear(pts, OEIL_GAUCHE) + ear(pts, OEIL_DROIT)) / 2.0
            mar_val      = mar(pts)
            incl_val     = inclinaison(pts)
            last_ear_val  = ear_val
            last_mar_val  = mar_val
            last_incl_val = incl_val
            no_face       = 0
            frames_detected += 1
            min_ear = min(min_ear, ear_val)
        else:
            no_face += 1
            if no_face <= HOLD:
                ear_val, mar_val, incl_val = last_ear_val, last_mar_val, last_incl_val
            else:
                ear_val, mar_val, incl_val = 0.30, 0.0, 0.0

        conditions = {
            "YEUX_FERMES": ear_val  < SEUIL_EAR,
            "BAILLEMENT":  mar_val  > SEUIL_MAR,
            "TETE_AVANT":  incl_val > SEUIL_INCLINAISON,
        }
        thresholds_t = {
            "YEUX_FERMES": SEUIL_TEMPS_YEUX,
            "BAILLEMENT":  SEUIL_TEMPS_BOUCHE,
            "TETE_AVANT":  SEUIL_TEMPS_AVANT,
        }

        for atype, triggered in conditions.items():
            seuil_t = thresholds_t[atype]
            if triggered:
                if timer_start[atype] is None:
                    timer_start[atype] = t
                duration = t - timer_start[atype]
                if duration >= seuil_t and not active[atype]:
                    active[atype] = True
                    fired.append({"type": atype, "start_s": t, "end_s": None})
            else:
                if active[atype]:
                    for f_ev in reversed(fired):
                        if f_ev["type"] == atype and f_ev["end_s"] is None:
                            f_ev["end_s"] = t
                            break
                    active[atype] = False
                timer_start[atype] = None

        if frame_idx % 200 == 0:
            print(f"  {frame_idx/total_frames*100:.0f}%  t={t:.1f}s"
                  f"  EAR={ear_val:.2f}  MAR={mar_val:.2f}  INCL={incl_val:.2f}",
                  end="\r")

    # Close still-active alerts
    end_t = frame_idx / fps
    for atype in ALERT_TYPES:
        if active[atype]:
            for f_ev in reversed(fired):
                if f_ev["type"] == atype and f_ev["end_s"] is None:
                    f_ev["end_s"] = end_t
                    break

    cap.release()
    detector.close()

    face_pct = frames_detected / max(frame_idx, 1) * 100
    print(f"\n  Done. {len(fired)} alerts fired.")
    print(f"  Face detected : {face_pct:.1f}% of frames")
    print(f"  Min EAR seen  : {min_ear:.3f}")

    # ---- Match fired alerts against ground truth ----
    results_by_type = {}
    for atype in ALERT_TYPES:
        gt_for   = [e for e in gt_events if e["type"] == atype]
        fire_for = [f for f in fired      if f["type"] == atype]

        tp, fp, matched_gt = 0, 0, set()

        for f_ev in fire_for:
            matched = False
            for i, gt_ev in enumerate(gt_for):
                if i in matched_gt:
                    continue
                fire_end = f_ev["end_s"] or f_ev["start_s"]
                if (f_ev["start_s"] <= gt_ev["end"] + TOLERANCE and
                        fire_end >= gt_ev["start"] - TOLERANCE):
                    tp += 1
                    matched_gt.add(i)
                    matched = True
                    break
            if not matched:
                fp += 1

        fn        = len(gt_for) - len(matched_gt)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        results_by_type[atype] = dict(tp=tp, fp=fp, fn=fn,
                                       precision=precision, recall=recall, f1=f1)

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS  (fatigue metrics only)")
    print("=" * 60)
    print(f"  {'Alert':<16} {'Precision':>10} {'Recall':>8} {'F1':>7}"
          f" {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 60)
    totals = [0, 0, 0]
    count  = 0
    for atype, m in results_by_type.items():
        if m["tp"] + m["fp"] + m["fn"] == 0:
            continue
        print(f"  {atype:<16} {m['precision']:>9.1%} {m['recall']:>7.1%}"
              f" {m['f1']:>6.1%} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")
        totals[0] += m["precision"]
        totals[1] += m["recall"]
        totals[2] += m["f1"]
        count      += 1
    if count:
        print("-" * 60)
        print(f"  {'MACRO AVG':<16} {totals[0]/count:>9.1%} {totals[1]/count:>7.1%}"
              f" {totals[2]/count:>6.1%}")
    print("=" * 60)
    print("\n  NOTE: Phone detection — test live with detection.py.")
    print("        YOLO on CPU is too slow for accurate video-based timing.\n")

    # ---- Save CSV ----
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/eval_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alert_type", "precision", "recall", "f1", "tp", "fp", "fn"])
        for atype, m in results_by_type.items():
            w.writerow([atype, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                        f"{m['f1']:.4f}", m["tp"], m["fp"], m["fn"]])
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt",    required=True)
    args = parser.parse_args()
    run(args.video, args.gt)

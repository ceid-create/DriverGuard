# ============================================================
#  DriverGuard — Offline Evaluation Script
#  Runs the detector on a recorded video file and compares
#  against a ground truth CSV to compute precision/recall/F1.
# ============================================================
#
#  HOW TO USE:
#  1. Record a test video where you perform each event (eyes closed,
#     yawn, head tilt, phone). Save it as eval_video.mp4.
#  2. Fill in ground_truth.csv with the timestamp of each event.
#     (run this script once with --make-gt to generate the template)
#  3. Run:  python evaluate.py --video eval_video.mp4 --gt ground_truth.csv
#  4. Results are printed and saved to logs/eval_results.csv
# ============================================================

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import cv2
import mediapipe as mp
import math
import csv
import argparse
import time
from datetime import datetime
from ultralytics import YOLO

# ---- Landmark indices (must match detection.py) ----
OEIL_GAUCHE = [362, 385, 387, 263, 373, 380]
OEIL_DROIT  = [33,  160, 158, 133, 153, 144]
BOUCHE      = [13, 14, 78, 308]
COIN_DROIT  = 33
COIN_GAUCHE = 263
NEZ_BOUT    = 4
NEZ_RACINE  = 6

# ---- Default thresholds (same as detection.py) ----
DEFAULT_THRESHOLDS = {
    "EAR":         0.25,
    "MAR":         0.50,
    "ANGLE":       20.0,
    "INCLINAISON": 0.15,   # raised from 0.08 — matches detection.py
}

# ---- Time-based alert triggers (seconds) ----
SEUIL_TEMPS = {
    "YEUX_FERMES": 2.0,
    "BAILLEMENT":  1.0,
    "TETE_PENCHEE": 2.0,
    "TETE_AVANT":   2.5,
    "TELEPHONE":    3.0,
}

ALERT_TYPES = ["YEUX_FERMES", "BAILLEMENT", "TETE_PENCHEE", "TETE_AVANT", "TELEPHONE"]

YOLO_CLASSE_TELEPHONE  = 1
YOLO_CONFIANCE_MIN     = 0.50
YOLO_TAILLE_MIN        = 1500
SEUIL_FRAMES_TELEPHONE = 3


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def calculer_ear(points, indices):
    p1, p2, p3, p4, p5, p6 = [points[i] for i in indices]
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * distance(p1, p4))


def calculer_mar(points, indices):
    haut, bas, gauche, droit = [points[i] for i in indices]
    largeur = distance(gauche, droit)
    return distance(haut, bas) / largeur if largeur > 0 else 0.0


def calculer_angle_tete(points):
    dx = points[COIN_GAUCHE].x - points[COIN_DROIT].x
    dy = points[COIN_GAUCHE].y - points[COIN_DROIT].y
    return math.degrees(math.atan2(dy, dx))


def calculer_inclinaison_verticale(points):
    return points[NEZ_BOUT].y - points[NEZ_RACINE].y


def detecter_telephone_yolo(resultats_yolo, hauteur_img):
    best_box, best_conf = None, 0.0
    for r in resultats_yolo:
        for box in r.boxes:
            if int(box.cls[0]) != YOLO_CLASSE_TELEPHONE:
                continue
            conf = float(box.conf[0])
            if conf < YOLO_CONFIANCE_MIN:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) < YOLO_TAILLE_MIN:
                continue
            if (y1 + y2) / 2 / hauteur_img > 0.92:
                continue
            if conf > best_conf:
                best_conf, best_box = conf, (x1, y1, x2, y2)
    return (True, best_box) if best_box else (False, None)


def bbox_proche_visage(bbox_tel, bbox_visage, marge=350):
    tx1, ty1, tx2, ty2 = bbox_tel
    fx1, fy1, fx2, fy2 = bbox_visage
    fx1 -= marge; fy1 -= marge; fx2 += marge; fy2 += marge
    return not (tx2 < fx1 or tx1 > fx2 or ty2 < fy1 or ty1 > fy2)


def load_ground_truth(gt_path):
    """
    Load ground truth CSV. Expected format (no header row needed, # = comment):
        YEUX_FERMES, 5.2, 8.5
        BAILLEMENT,  15.0, 17.5
        TETE_PENCHEE, 30.0, 33.5
        TETE_AVANT,  45.0, 48.5
        TELEPHONE,   60.0, 64.0

    Each row: alert_type, start_second, end_second
    """
    events = []
    with open(gt_path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            alert_type = parts[0].upper()
            start_s = float(parts[1])
            end_s   = float(parts[2])
            events.append({"type": alert_type, "start": start_s, "end": end_s, "matched": False})
    return events


def make_gt_template(output_path):
    """Write a ground truth template the user can fill in."""
    template = """\
# DriverGuard Ground Truth Template
# Fill in the timestamps (in seconds) for each event in your eval video.
# Format: ALERT_TYPE, start_second, end_second
# Delete rows for events you did NOT perform.
# Add multiple rows of the same type if the event occurs more than once.
#
# ALERT_TYPE options: YEUX_FERMES  BAILLEMENT  TETE_PENCHEE  TETE_AVANT  TELEPHONE
#
# Example: if you closed your eyes from t=5.2s to t=8.5s, write:
#   YEUX_FERMES, 5.2, 8.5
#
# Tip: play back your eval video in VLC and note the timestamps.

YEUX_FERMES,  0.0, 0.0
BAILLEMENT,   0.0, 0.0
TETE_PENCHEE, 0.0, 0.0
TETE_AVANT,   0.0, 0.0
TELEPHONE,    0.0, 0.0
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"Ground truth template written to: {output_path}")
    print("Fill in the timestamps, then run: python evaluate.py --video eval_video.mp4 --gt ground_truth.csv")


def run_evaluation(video_path, gt_path, thresholds=None, model_path=None, verbose=True):
    """
    Run the full detector on a video file and return per-type metrics.
    Returns a dict: { alert_type: {precision, recall, f1, tp, fp, fn} }
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # ---- Load YOLO ----
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "runs", "detect", "train3", "weights", "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO weights not found at: {model_path}")

    modele_yolo = YOLO(model_path)
    modele_yolo.to('cpu')

    # ---- Load face mesh ----
    mp_face   = mp.solutions.face_mesh
    # Lower confidence thresholds for recorded video — compressed frames are
    # harder to detect than a live camera stream, so 0.5 misses too many frames.
    detecteur = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # ---- Load ground truth ----
    gt_events = load_ground_truth(gt_path)

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"\n[Evaluate] Video : {os.path.basename(video_path)}")
        print(f"[Evaluate] FPS   : {fps:.1f}  |  Frames : {total_frames}")
        print(f"[Evaluate] GT    : {len(gt_events)} events")
        print(f"[Evaluate] Thresholds: EAR={thresholds['EAR']} MAR={thresholds['MAR']} "
              f"ANGLE={thresholds['ANGLE']} INCLINAISON={thresholds['INCLINAISON']}\n")

    # ---- State tracking ----
    timers        = {k: None for k in ALERT_TYPES}
    fired_alerts  = []
    active_alerts = {k: False for k in ALERT_TYPES}
    phone_frames  = 0

    # Face-hold: keep last known values for up to HOLD frames when face is lost,
    # so one or two missed frames do not reset the 2-second timer.
    HOLD_FRAMES       = 6
    frames_no_face    = 0
    last_ear          = 0.30
    last_mar          = 0.0
    last_angle        = 0.0
    last_incl         = 0.0
    last_face_bbox    = None

    # Debug stats
    frames_face_detected = 0
    min_ear_seen         = 1.0

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_idx / fps
        frame_idx += 1
        h, w = frame.shape[:2]

        # ---- Face mesh ----
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detecteur.process(rgb)

        if results.multi_face_landmarks:
            pts = results.multi_face_landmarks[0].landmark
            ear_moyen   = (calculer_ear(pts, OEIL_GAUCHE) + calculer_ear(pts, OEIL_DROIT)) / 2.0
            mar         = calculer_mar(pts, BOUCHE)
            angle       = calculer_angle_tete(pts)
            inclinaison = calculer_inclinaison_verticale(pts)
            xs = [int(lm.x * w) for lm in pts]
            ys = [int(lm.y * h) for lm in pts]
            face_bbox = (min(xs), min(ys), max(xs), max(ys))
            # Update hold values
            last_ear = ear_moyen; last_mar = mar
            last_angle = angle;   last_incl = inclinaison
            last_face_bbox = face_bbox
            frames_no_face = 0
            frames_face_detected += 1
            min_ear_seen = min(min_ear_seen, ear_moyen)
        else:
            frames_no_face += 1
            if frames_no_face <= HOLD_FRAMES:
                # Hold last known values — face briefly lost (blink / compression)
                ear_moyen, mar, angle, inclinaison = last_ear, last_mar, last_angle, last_incl
                face_bbox = last_face_bbox
            else:
                # Face genuinely gone — reset to safe defaults
                ear_moyen, mar, angle, inclinaison = 0.30, 0.0, 0.0, 0.0
                face_bbox = None

        # ---- YOLO phone ----
        res_yolo = modele_yolo.predict(frame, verbose=False, imgsz=416,
                                       conf=YOLO_CONFIANCE_MIN, device='cpu')
        phone_raw, phone_box = detecter_telephone_yolo(res_yolo, h)
        if phone_raw and face_bbox:
            if not bbox_proche_visage(phone_box, face_bbox, marge=350):
                phone_raw = False
        phone_frames = min(phone_frames + 1, 10) if phone_raw else max(phone_frames - 1, 0)
        phone_detected = phone_frames >= SEUIL_FRAMES_TELEPHONE

        # ---- Check each alert condition ----
        conditions = {
            "YEUX_FERMES":  ear_moyen < thresholds["EAR"],
            "BAILLEMENT":   mar > thresholds["MAR"],
            "TETE_PENCHEE": abs(angle) > thresholds["ANGLE"],
            "TETE_AVANT":   inclinaison > thresholds["INCLINAISON"],
            "TELEPHONE":    phone_detected,
        }

        for alert_type, triggered in conditions.items():
            seuil_t = SEUIL_TEMPS[alert_type]
            if triggered:
                if timers[alert_type] is None:
                    timers[alert_type] = t_sec
                duration = t_sec - timers[alert_type]
                if duration >= seuil_t and not active_alerts[alert_type]:
                    active_alerts[alert_type] = True
                    fired_alerts.append({"type": alert_type, "start": t_sec, "end": None})
            else:
                if active_alerts[alert_type]:
                    # close the alert
                    for fa in reversed(fired_alerts):
                        if fa["type"] == alert_type and fa["end"] is None:
                            fa["end"] = t_sec
                            break
                    active_alerts[alert_type] = False
                timers[alert_type] = None

        if verbose and frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  Processing... {pct:.0f}%  (t={t_sec:.1f}s)", end="\r")

    # Close any alerts still active at end of video
    end_t = frame_idx / fps
    for alert_type in ALERT_TYPES:
        if active_alerts[alert_type]:
            for fa in reversed(fired_alerts):
                if fa["type"] == alert_type and fa["end"] is None:
                    fa["end"] = end_t
                    break

    cap.release()
    detecteur.close()

    if verbose:
        face_pct = frames_face_detected / max(frame_idx, 1) * 100
        print(f"\n  Done. {len(fired_alerts)} alerts fired.")
        print(f"  Face detected : {frames_face_detected}/{frame_idx} frames ({face_pct:.1f}%)")
        print(f"  Min EAR seen  : {min_ear_seen:.3f}  "
              f"({'below threshold — eyes were detected closed' if min_ear_seen < thresholds['EAR'] else 'NEVER below threshold — eyes never registered as closed'})")

    # ---- Compute metrics ----
    # An alert is a TP if it overlaps with any unmatched GT event of the same type.
    # Overlap criterion: the fired alert's start falls within the GT window (±2s tolerance).
    TOLERANCE = 2.0

    results_by_type = {}
    for atype in ALERT_TYPES:
        gt_for_type     = [e for e in gt_events if e["type"] == atype]
        fired_for_type  = [a for a in fired_alerts if a["type"] == atype]

        tp, fp, fn = 0, 0, 0
        matched_gt = set()

        for fired in fired_for_type:
            matched = False
            for i, gt in enumerate(gt_for_type):
                if i in matched_gt:
                    continue
                # Overlap check: fired start is within GT window with tolerance
                if (fired["start"] <= gt["end"] + TOLERANCE and
                        (fired["end"] or fired["start"]) >= gt["start"] - TOLERANCE):
                    tp += 1
                    matched_gt.add(i)
                    matched = True
                    break
            if not matched:
                fp += 1

        fn = len(gt_for_type) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results_by_type[atype] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    return results_by_type, fired_alerts


def print_results(results_by_type):
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    print(f"  {'Alert Type':<18} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 65)
    p_sum = r_sum = f_sum = 0.0
    count = 0
    for atype, m in results_by_type.items():
        if m["tp"] + m["fp"] + m["fn"] == 0:
            continue
        print(f"  {atype:<18} {m['precision']:>9.1%} {m['recall']:>9.1%} "
              f"{m['f1']:>7.1%} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
        p_sum += m["precision"]; r_sum += m["recall"]; f_sum += m["f1"]
        count += 1
    if count:
        print("-" * 65)
        print(f"  {'MACRO AVERAGE':<18} {p_sum/count:>9.1%} {r_sum/count:>9.1%} {f_sum/count:>7.1%}")
    print("=" * 65 + "\n")


def save_results(results_by_type, fired_alerts, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alert_type", "precision", "recall", "f1", "tp", "fp", "fn"])
        for atype, m in results_by_type.items():
            writer.writerow([atype,
                             f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                             m["tp"], m["fp"], m["fn"]])
    print(f"Results saved to: {output_path}")


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriverGuard offline evaluator")
    parser.add_argument("--video",   type=str, help="Path to eval video file")
    parser.add_argument("--gt",      type=str, help="Path to ground truth CSV")
    parser.add_argument("--make-gt", type=str, metavar="OUTPUT",
                        help="Generate a ground truth template CSV and exit")
    parser.add_argument("--output",  type=str,
                        default=f"logs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        help="Where to save the results CSV")
    args = parser.parse_args()

    if args.make_gt:
        make_gt_template(args.make_gt)
    elif args.video and args.gt:
        results, fired = run_evaluation(args.video, args.gt)
        print_results(results)
        save_results(results, fired, args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python evaluate.py --make-gt ground_truth.csv")
        print("  python evaluate.py --video eval_video.mp4 --gt ground_truth.csv")

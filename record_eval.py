# ============================================================
#  DriverGuard — Evaluation Video Recorder
#  Records webcam + auto-generates ground_truth.csv
#  by pressing keys at the start/end of each event.
#
#  Usage:  python record_eval.py
#  Output: eval_video.mp4  +  ground_truth.csv
# ============================================================

import cv2
import csv
import time

OUTPUT_VIDEO = "eval_video.mp4"
OUTPUT_GT    = "ground_truth.csv"
FPS          = 20

# Key → alert type  (press once to START, press again to STOP)
KEY_MAP = {
    ord('e'): "YEUX_FERMES",
    ord('y'): "BAILLEMENT",
    ord('h'): "TETE_PENCHEE",
    ord('f'): "TETE_AVANT",
    ord('p'): "TELEPHONE",
}

LABELS = {
    "YEUX_FERMES":  "E — Eyes closed",
    "BAILLEMENT":   "Y — Yawn",
    "TETE_PENCHEE": "H — Head tilt",
    "TETE_AVANT":   "F — Head forward",
    "TELEPHONE":    "P — Phone",
}

# Colours
NOIR   = (15,  15,  15)
BLANC  = (240, 240, 240)
VERT   = (80,  220, 80)
ORANGE = (0,   165, 255)
ROUGE  = (50,  50,  220)
GRIS   = (140, 140, 140)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("ERROR: Cannot open webcam.")
    exit(1)

h, w       = frame.shape[:2]
fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
writer_out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

print("\n" + "=" * 42)
print("  EVALUATION RECORDER")
print("=" * 42)
print("  Keys (toggle start / stop):")
print("    E = eyes closed")
print("    Y = yawn")
print("    H = head tilt")
print("    F = head forward drop")
print("    P = phone in hand")
print("    Q = finish recording")
print("=" * 42 + "\n")

t0       = time.time()
active   = {}     # event_type → start_second
done     = []     # list of (event_type, start_s, end_s)

WINDOW = "DriverGuard — Eval Recorder  (Q to stop)"
cv2.namedWindow(WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = time.time() - t0

    # ── Timer (top-left) ────────────────────────────────
    cv2.rectangle(frame, (0, 0), (210, 48), NOIR, -1)
    cv2.putText(frame, f"t = {t:6.1f} s",
                (8, 34), cv2.FONT_HERSHEY_DUPLEX, 0.95, VERT, 2)

    # ── Blinking REC dot (top-right) ────────────────────
    if int(t * 2) % 2 == 0:
        cv2.circle(frame, (w - 28, 22), 10, ROUGE, -1)
    cv2.putText(frame, "REC", (w - 70, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROUGE, 2)

    # ── Active events (left side, below timer) ───────────
    y_txt = 75
    for event_type, label in LABELS.items():
        if event_type in active:
            dur = t - active[event_type]
            cv2.rectangle(frame, (0, y_txt - 18), (340, y_txt + 6), (0, 60, 120), -1)
            cv2.putText(frame, f"  {label}  [{dur:.1f}s]  <press again to stop>",
                        (4, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1)
        y_txt += 26

    # ── Completed events log (bottom-left) ──────────────
    log_lines = done[-4:]          # show last 4 completed
    ly = h - 10 - len(log_lines) * 18
    for ev, s, e in log_lines:
        cv2.putText(frame, f"  {ev}  {s:.1f}s → {e:.1f}s",
                    (4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS, 1)
        ly += 18

    # ── Key hints (very bottom) ──────────────────────────
    cv2.rectangle(frame, (0, h - 28), (w, h), NOIR, -1)
    cv2.putText(frame, "E=eyes  Y=yawn  H=head  F=fwd  P=phone  Q=stop",
                (6, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRIS, 1)

    writer_out.write(frame)
    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF

    # ── Exit ─────────────────────────────────────────────
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

    # ── Toggle event ─────────────────────────────────────
    if key in KEY_MAP:
        event_type = KEY_MAP[key]
        if event_type in active:
            start_s = active.pop(event_type)
            end_s   = round(t, 1)
            done.append((event_type, round(start_s, 1), end_s))
            print(f"  ✓ {event_type:<20} {start_s:.1f}s → {end_s:.1f}s")
        else:
            active[event_type] = t
            print(f"  ▶ {event_type} started at t={t:.1f}s  "
                  f"(press {chr(key).upper()} again to stop)")

# ── Close any events still open at end of recording ─────
for event_type, start_s in active.items():
    end_s = round(time.time() - t0, 1)
    done.append((event_type, round(start_s, 1), end_s))

cap.release()
writer_out.release()
cv2.destroyAllWindows()

# ── Save ground_truth.csv ────────────────────────────────
with open(OUTPUT_GT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(done)

print(f"\nSaved: {OUTPUT_VIDEO}")
print(f"Saved: {OUTPUT_GT}  ({len(done)} events)\n")
if done:
    print("Events logged:")
    for ev, s, e in done:
        print(f"  {ev:<20} {s:.1f}s → {e:.1f}s  ({e-s:.1f}s duration)")
print("\nNext step:")
print("  python evaluate.py --video eval_video.mp4 --gt ground_truth.csv")

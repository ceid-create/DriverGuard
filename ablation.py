# ============================================================
#  DriverGuard — Threshold Ablation Study
#  Sweeps EAR, MAR, and ANGLE thresholds and reports
#  precision/recall/F1 for each value.
#
#  HOW TO USE (after you have eval_video.mp4 + ground_truth.csv):
#    python ablation.py --video eval_video.mp4 --gt ground_truth.csv
#
#  Output: ablation_results.csv + printed table
# ============================================================

import argparse
import csv
import os
from datetime import datetime
from evaluate import run_evaluation, ALERT_TYPES

# ---- Threshold ranges to sweep ----
EAR_VALUES    = [0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30]
MAR_VALUES    = [0.40, 0.45, 0.50, 0.55, 0.60]
ANGLE_VALUES  = [15.0, 18.0, 20.0, 22.0, 25.0, 30.0]

# Default values (held fixed while sweeping the other)
DEFAULT_EAR   = 0.25
DEFAULT_MAR   = 0.50
DEFAULT_ANGLE = 20.0
DEFAULT_INCL  = 0.08


def macro_f1(results_by_type):
    scores = [m["f1"] for m in results_by_type.values() if m["tp"] + m["fp"] + m["fn"] > 0]
    return sum(scores) / len(scores) if scores else 0.0


def run_ablation(video_path, gt_path, output_path):
    rows = []

    print("\n" + "=" * 60)
    print("  ABLATION STUDY — Sweeping EAR threshold")
    print("=" * 60)
    print(f"  {'EAR':>6}  {'Prec(eyes)':>11}  {'Rec(eyes)':>10}  {'F1(eyes)':>9}  {'MacroF1':>8}")
    print("-" * 60)
    for ear in EAR_VALUES:
        thresh = {"EAR": ear, "MAR": DEFAULT_MAR,
                  "ANGLE": DEFAULT_ANGLE, "INCLINAISON": DEFAULT_INCL}
        res, _ = run_evaluation(video_path, gt_path, thresholds=thresh, verbose=False)
        m = res["YEUX_FERMES"]
        mf1 = macro_f1(res)
        print(f"  {ear:>6.2f}  {m['precision']:>10.1%}  {m['recall']:>9.1%}  "
              f"{m['f1']:>8.1%}  {mf1:>7.1%}")
        rows.append({"sweep": "EAR", "value": ear,
                     "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "macro_f1": mf1})

    print("\n" + "=" * 60)
    print("  ABLATION STUDY — Sweeping MAR threshold")
    print("=" * 60)
    print(f"  {'MAR':>6}  {'Prec(yawn)':>11}  {'Rec(yawn)':>10}  {'F1(yawn)':>9}  {'MacroF1':>8}")
    print("-" * 60)
    for mar in MAR_VALUES:
        thresh = {"EAR": DEFAULT_EAR, "MAR": mar,
                  "ANGLE": DEFAULT_ANGLE, "INCLINAISON": DEFAULT_INCL}
        res, _ = run_evaluation(video_path, gt_path, thresholds=thresh, verbose=False)
        m = res["BAILLEMENT"]
        mf1 = macro_f1(res)
        print(f"  {mar:>6.2f}  {m['precision']:>10.1%}  {m['recall']:>9.1%}  "
              f"{m['f1']:>8.1%}  {mf1:>7.1%}")
        rows.append({"sweep": "MAR", "value": mar,
                     "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "macro_f1": mf1})

    print("\n" + "=" * 60)
    print("  ABLATION STUDY — Sweeping ANGLE threshold (degrees)")
    print("=" * 60)
    print(f"  {'ANGLE':>6}  {'Prec(head)':>11}  {'Rec(head)':>10}  {'F1(head)':>9}  {'MacroF1':>8}")
    print("-" * 60)
    for angle in ANGLE_VALUES:
        thresh = {"EAR": DEFAULT_EAR, "MAR": DEFAULT_MAR,
                  "ANGLE": angle, "INCLINAISON": DEFAULT_INCL}
        res, _ = run_evaluation(video_path, gt_path, thresholds=thresh, verbose=False)
        m = res["TETE_PENCHEE"]
        mf1 = macro_f1(res)
        print(f"  {angle:>6.1f}  {m['precision']:>10.1%}  {m['recall']:>9.1%}  "
              f"{m['f1']:>8.1%}  {mf1:>7.1%}")
        rows.append({"sweep": "ANGLE", "value": angle,
                     "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "macro_f1": mf1})

    print()

    # ---- Save to CSV ----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sweep", "value",
                                                "precision", "recall", "f1", "macro_f1"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                             for k, v in row.items()})
    print(f"Ablation results saved to: {output_path}")

    # ---- Summary: best threshold per metric ----
    print("\n  Best thresholds by macro F1:")
    for sweep_name in ["EAR", "MAR", "ANGLE"]:
        sweep_rows = [r for r in rows if r["sweep"] == sweep_name]
        best = max(sweep_rows, key=lambda r: r["macro_f1"])
        print(f"    {sweep_name:<8} → best value = {best['value']:.2f}  "
              f"(macro F1 = {best['macro_f1']:.1%})")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriverGuard threshold ablation study")
    parser.add_argument("--video",  required=True, help="Path to eval video file")
    parser.add_argument("--gt",     required=True, help="Path to ground truth CSV")
    parser.add_argument("--output", default=f"logs/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        help="Where to save ablation results CSV")
    args = parser.parse_args()
    run_ablation(args.video, args.gt, args.output)

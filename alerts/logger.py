"""CSV session logger."""
import csv
import os
from datetime import datetime

LOG_DIR = "logs"


def init_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Horodatage", "Type_Alerte", "EAR", "MAR", "Angle", "Duree_s"])
    return path


def log_alerte(log_file, type_alerte, ear, mar, angle, duree):
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            type_alerte,
            f"{ear:.3f}", f"{mar:.3f}", f"{angle:.1f}", f"{duree:.1f}",
        ])

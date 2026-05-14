"""
Phone detection using YOLOv8.
Includes face-proximity filter and temporal smoothing to reduce false positives.
"""
import os
import time
from ultralytics import YOLO

YOLO_CLASSE_TELEPHONE  = 1      # class index in the custom model
YOLO_CONFIANCE_MIN     = 0.50
YOLO_TAILLE_MIN        = 1500   # minimum bbox area in pixels²
SEUIL_FRAMES_TELEPHONE = 3      # consecutive positive frames before confirming
SEUIL_TEMPS_TELEPHONE  = 3.0    # seconds phone must be present before alert


def charger_modele(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "runs", "detect", "train3", "weights", "best.pt")
    modele = YOLO(model_path)
    modele.to('cpu')
    return modele


def detecter_telephone_yolo(resultats_yolo, hauteur_img):
    """Return (detected: bool, best_bbox: tuple|None)."""
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
            # Reject detections at the very bottom of the frame (table/desk objects)
            if (y1 + y2) / 2 / hauteur_img > 0.92:
                continue
            if conf > best_conf:
                best_conf, best_box = conf, (x1, y1, x2, y2)
    return (True, best_box) if best_box else (False, None)


def bbox_proche_visage(bbox_tel, bbox_visage, marge=150):
    """
    True if the phone bbox is near the face region (expanded by marge pixels)
    but NOT sitting directly over the face itself.

    The marge is intentionally moderate (150px) — enough to accept a phone held
    at arm's length in front of the camera, but tight enough to reject background
    objects far from the driver.

    The face-overlap rejection prevents YOLO from misclassifying the face itself
    as a phone: if the phone bbox covers >40% of the face bbox area, it is
    rejected (it's almost certainly a face false-positive, not a real phone).
    """
    tx1, ty1, tx2, ty2 = bbox_tel
    fx1, fy1, fx2, fy2 = bbox_visage

    # Reject if phone bbox overlaps heavily with the face itself
    ix1, iy1 = max(tx1, fx1), max(ty1, fy1)
    ix2, iy2 = min(tx2, fx2), min(ty2, fy2)
    if ix2 > ix1 and iy2 > iy1:
        face_area  = max(1, (fx2 - fx1) * (fy2 - fy1))
        overlap    = (ix2 - ix1) * (iy2 - iy1)
        if overlap / face_area > 0.40:
            return False

    fx1 -= marge; fy1 -= marge; fx2 += marge; fy2 += marge
    return not (tx2 < fx1 or tx1 > fx2 or ty2 < fy1 or ty1 > fy2)


class PhoneDetector:
    """
    Stateful phone detector with temporal smoothing and face-proximity filter.
    Call process_frame() each frame.
    """

    def __init__(self, model_path=None):
        self.modele               = charger_modele(model_path)
        self.compteur_frames      = 0
        self.phone_bbox           = None
        self.temps_telephone      = None

    def process_frame(self, frame, hauteur_img, face_bbox=None):
        """
        frame      : BGR numpy array
        hauteur_img: frame height in pixels
        face_bbox  : (x1,y1,x2,y2) from MediaPipe, or None

        Returns dict with keys: detected (bool), bbox, duree_s, confirme
        """
        res = self.modele.predict(frame, verbose=False,
                                  imgsz=416, conf=YOLO_CONFIANCE_MIN, device='cpu')
        phone_raw, bbox_new = detecter_telephone_yolo(res, hauteur_img)

        # Face-proximity filter
        if phone_raw and face_bbox is not None:
            if not bbox_proche_visage(bbox_new, face_bbox, marge=350):
                phone_raw = False
                bbox_new  = None

        if bbox_new is not None:
            self.phone_bbox = bbox_new

        # Temporal smoothing (hysteresis counter)
        if phone_raw:
            self.compteur_frames = min(self.compteur_frames + 1, 10)
            if self.temps_telephone is None:
                self.temps_telephone = time.time()
        else:
            self.compteur_frames = max(self.compteur_frames - 1, 0)
            if self.compteur_frames == 0:
                self.phone_bbox      = None
                self.temps_telephone = None

        confirme = self.compteur_frames >= SEUIL_FRAMES_TELEPHONE
        duree    = (time.time() - self.temps_telephone) if self.temps_telephone else 0.0

        return {
            "detected": confirme,
            "bbox":     self.phone_bbox,
            "duree_s":  duree,
            "confirme": confirme,
        }

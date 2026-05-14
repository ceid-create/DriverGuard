"""
Fatigue detection from MediaPipe face landmarks.
Computes EAR, MAR, head tilt angle, vertical head drop, blink rate,
and a composite fatigue score.
"""
import math
import time
from .landmarks import (distance, OEIL_GAUCHE, OEIL_DROIT,
                         BOUCHE, COIN_DROIT, COIN_GAUCHE, NEZ_BOUT, NEZ_RACINE)

# Detection thresholds
SEUIL_EAR          = 0.25   # below → eyes closed
SEUIL_MAR          = 0.50   # above → yawning
SEUIL_ANGLE        = 20.0   # degrees, lateral head tilt
SEUIL_INCLINAISON  = 0.08   # normalised nose drop (forward head drop)
SEUIL_CLIGNEMENTS  = 25     # blinks/min above this adds to fatigue score

# Time-based alert triggers (seconds condition must persist)
SEUIL_TEMPS_YEUX   = 2.0
SEUIL_TEMPS_BOUCHE = 1.0
SEUIL_TEMPS_TETE   = 2.0
SEUIL_TEMPS_REGARD = 2.5


def calculer_ear(points, indices):
    """Eye Aspect Ratio — Soukupova & Cech (2016)."""
    p1, p2, p3, p4, p5, p6 = [points[i] for i in indices]
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * distance(p1, p4))


def calculer_mar(points, indices):
    """Mouth Aspect Ratio — analogous to EAR for yawn detection."""
    haut, bas, gauche, droit = [points[i] for i in indices]
    largeur = distance(gauche, droit)
    return distance(haut, bas) / largeur if largeur > 0 else 0.0


def calculer_angle_tete(points):
    """Lateral head tilt in degrees from the inter-eye axis."""
    dx = points[COIN_GAUCHE].x - points[COIN_DROIT].x
    dy = points[COIN_GAUCHE].y - points[COIN_DROIT].y
    return math.degrees(math.atan2(dy, dx))


def calculer_inclinaison_verticale(points):
    """Vertical nose drop — positive value means head falling forward."""
    return points[NEZ_BOUT].y - points[NEZ_RACINE].y


def calculer_score_fatigue(ear, mar, angle, clignements_par_min, inclinaison):
    """
    Composite fatigue score 0–100.
    Weighted sum of individual indicators; capped at 100.
    """
    score = 0
    if ear < SEUIL_EAR:           score += 30
    elif ear < 0.28:               score += 10
    if mar > SEUIL_MAR:            score += 20
    elif mar > 0.35:               score += 5
    if abs(angle) > SEUIL_ANGLE:  score += 25
    elif abs(angle) > 10:         score += 8
    if clignements_par_min > SEUIL_CLIGNEMENTS: score += 15
    if inclinaison > SEUIL_INCLINAISON:         score += 10
    return min(score, 100)


class FatigueDetector:
    """
    Stateful detector: call process_frame() each frame.
    Returns a FatigueState with current metrics and active alerts.
    """

    def __init__(self):
        self.temps_yeux   = None
        self.temps_bouche = None
        self.temps_tete   = None
        self.temps_regard = None

        self.compteur_clignements  = 0
        self.temps_dernier_clin    = time.time()
        self.clignements_par_min   = 0
        self.oeil_ouvert_precedent = True

        self.historique_ear = []
        self.HISTORIQUE_MAX = 100

    def process_frame(self, points):
        """
        points: mediapipe landmark list for one face.
        Returns dict with keys: ear, mar, angle, inclinaison,
                                 score, alerts (dict of alert_type → duration_s or None)
        """
        ear_g = calculer_ear(points, OEIL_GAUCHE)
        ear_d = calculer_ear(points, OEIL_DROIT)
        ear   = (ear_g + ear_d) / 2.0
        mar   = calculer_mar(points, BOUCHE)
        angle = calculer_angle_tete(points)
        incl  = calculer_inclinaison_verticale(points)

        # EAR history for the live graph
        self.historique_ear.append(ear)
        if len(self.historique_ear) > self.HISTORIQUE_MAX:
            self.historique_ear.pop(0)

        # Blink counting (falling edge of EAR)
        oeil_ouvert = ear >= SEUIL_EAR
        if not oeil_ouvert and self.oeil_ouvert_precedent:
            self.compteur_clignements += 1
        self.oeil_ouvert_precedent = oeil_ouvert

        elapsed = time.time() - self.temps_dernier_clin
        if elapsed >= 10:
            self.clignements_par_min = int(self.compteur_clignements * (60 / elapsed))
            self.compteur_clignements = 0
            self.temps_dernier_clin = time.time()

        score = calculer_score_fatigue(ear, mar, angle, self.clignements_par_min, incl)

        # Alert timers
        now = time.time()
        alerts = {}

        # Eyes closed
        if ear < SEUIL_EAR:
            if self.temps_yeux is None:
                self.temps_yeux = now
            alerts["YEUX_FERMES"] = now - self.temps_yeux
        else:
            self.temps_yeux = None
            alerts["YEUX_FERMES"] = None

        # Yawning
        if mar > SEUIL_MAR:
            if self.temps_bouche is None:
                self.temps_bouche = now
            alerts["BAILLEMENT"] = now - self.temps_bouche
        else:
            self.temps_bouche = None
            alerts["BAILLEMENT"] = None

        # Lateral head tilt
        if abs(angle) > SEUIL_ANGLE:
            if self.temps_tete is None:
                self.temps_tete = now
            alerts["TETE_PENCHEE"] = now - self.temps_tete
        else:
            self.temps_tete = None
            alerts["TETE_PENCHEE"] = None

        # Forward head drop
        if incl > SEUIL_INCLINAISON:
            if self.temps_regard is None:
                self.temps_regard = now
            alerts["TETE_AVANT"] = now - self.temps_regard
        else:
            self.temps_regard = None
            alerts["TETE_AVANT"] = None

        return {
            "ear": ear, "mar": mar, "angle": angle,
            "inclinaison": incl, "score": score,
            "clignements_par_min": self.clignements_par_min,
            "historique_ear": self.historique_ear,
            "alerts": alerts,
        }

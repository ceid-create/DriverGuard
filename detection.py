# ============================================================
#  DriverGuard — Driver Fatigue & Distraction Detection
#  Authors : Joe Haddad · Sem4 CV Project · USJ 2025-2026
# ============================================================

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'   # suppress pygame banner

import cv2
import mediapipe as mp
import math
import time
import pygame
import csv
import argparse
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from ultralytics import YOLO
from config   import GMAIL_EXPEDITEUR, GMAIL_MOT_PASSE, GMAIL_DESTINATAIRE
try:
    from hardware import ArduinoHardware
    hw = ArduinoHardware()
except Exception:
    class _NoHW:
        def is_active(self):   return True
        def send_alarm(self):  pass
        def disconnect(self):  pass
    hw = _NoHW()
    print("[HW] Hardware module unavailable — running in software-only mode")

# ---- Eval mode flag ----
_parser = argparse.ArgumentParser()
_parser.add_argument("--eval", action="store_true",
                     help="Run in evaluation mode (press E/Y/F/P to mark events)")
EVAL_MODE = _parser.parse_args().eval

#  Initialiser le son
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
script_dir = os.path.dirname(os.path.abspath(__file__))
pygame.mixer.music.load(os.path.join(script_dir, "alarme.mp3"))

#  Dossier de logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

#  Initialiser le fichier CSV
with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Horodatage", "Type_Alerte", "EAR", "MAR", "Angle", "Duree_s"])

def log_alerte(type_alerte, ear, mar, angle, duree):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            type_alerte,
            f"{ear:.3f}",
            f"{mar:.3f}",
            f"{angle:.1f}",
            f"{duree:.1f}"
        ])

#  Fonction pour envoyer un Email
def envoyer_sms(message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = "🚨 ALERTE DriverGuard"
        msg["From"]    = f"DriverGuard <{GMAIL_EXPEDITEUR}>"
        msg["To"]      = GMAIL_DESTINATAIRE
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_EXPEDITEUR, GMAIL_MOT_PASSE)
            smtp.send_message(msg)
        print(f"Email envoyé : {message}")
    except Exception as e:
        print(f"Erreur Email : {e}")

#  Indices des points des yeux
OEIL_GAUCHE = [362, 385, 387, 263, 373, 380]
OEIL_DROIT  = [33,  160, 158, 133, 153, 144]

#  Indices des points de la bouche
BOUCHE = [13, 14, 78, 308]

#  Indices pour la détection de la tête
COIN_DROIT  = 33
COIN_GAUCHE = 263

#  Nez (pour détection regard hors route)
NEZ_BOUT   = 4
NEZ_RACINE = 6

#  Seuils
SEUIL_EAR             = 0.25
SEUIL_MAR             = 0.5
SEUIL_INCLINAISON     = 0.10
SEUIL_TEMPS_YEUX      = 2.0
SEUIL_TEMPS_BOUCHE    = 1.0
SEUIL_TEMPS_AVANT     = 2.0   # forward head drop → alarm after 2 seconds
SEUIL_TEMPS_TELEPHONE = 3.0
SEUIL_CLIGNEMENTS  = 25
DELAI_SMS          = 60

#  YOLO — détection téléphone
YOLO_CLASSE_TELEPHONE  = 1     # custom model : phone = classe 1
YOLO_CONFIANCE_MIN     = 0.50  # phone: 0.50-0.90, mouse rejected by proximity filter
YOLO_TAILLE_MIN        = 1500  # surface bbox minimale en pixels²
SEUIL_FRAMES_TELEPHONE = 3     # frames positives consécutives avant confirmation

# Couleurs (BGR)
NOIR       = (10,  10,  10)
BLANC      = (240, 240, 240)
ROUGE      = (40,  40,  220)
VERT       = (80,  220, 80)
ORANGE     = (0,   160, 255)
JAUNE      = (0,   230, 230)
BLEU_FONCE = (45,  35,  25)   # dark navy panel background
PANEL_HDR  = (60,  40,  20)   # sidebar header
GRIS       = (130, 130, 130)
CYAN       = (220, 220, 20)

# ---- Démarrage ----
print("\n" + "=" * 52)
print("   ██████╗ ██████╗ ██╗██╗   ██╗███████╗")
print("   ██╔══██╗██╔══██╗██║██║   ██║██╔════╝")
print("   ██║  ██║██████╔╝██║██║   ██║█████╗  ")
print("   ██║  ██║██╔══██╗██║╚██╗ ██╔╝██╔══╝  ")
print("   ██████╔╝██║  ██║██║ ╚████╔╝ ███████╗")
print("   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝")
print("            G U A R D")
print("=" * 52)
print("  Driver Fatigue & Distraction Monitor")
print("  USJ — Computer Vision Project 2025-2026")
print("=" * 52 + "\n")

# ---- Chargement modèle YOLO (nano — rapide sur CPU) ----
print("[*] Loading YOLO detection model...")
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "runs", "detect", "train3", "weights", "best.pt")
modele_yolo = YOLO(_model_path)
modele_yolo.to('cpu')
print("[+] YOLO model ready.")
print("[*] Starting camera feed...\n")

#  Fonction distance
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

#  Fonction EAR
def calculer_ear(points, indices):
    p1 = points[indices[0]]
    p2 = points[indices[1]]
    p3 = points[indices[2]]
    p4 = points[indices[3]]
    p5 = points[indices[4]]
    p6 = points[indices[5]]
    hauteur1 = distance(p2, p6)
    hauteur2 = distance(p3, p5)
    largeur  = distance(p1, p4)
    return (hauteur1 + hauteur2) / (2.0 * largeur)

#  Fonction MAR
def calculer_mar(points, indices):
    haut   = points[indices[0]]
    bas    = points[indices[1]]
    gauche = points[indices[2]]
    droit  = points[indices[3]]
    hauteur = distance(haut, bas)
    largeur = distance(gauche, droit)
    if largeur == 0:
        return 0
    return hauteur / largeur

#  Fonction angle de la tête
def calculer_angle_tete(points):
    oeil_droit  = points[COIN_DROIT]
    oeil_gauche = points[COIN_GAUCHE]
    dx = oeil_gauche.x - oeil_droit.x
    dy = oeil_gauche.y - oeil_droit.y
    return math.degrees(math.atan2(dy, dx))

#  Inclinaison verticale (assoupissement)
def calculer_inclinaison_verticale(points):
    nez_bout   = points[NEZ_BOUT]
    nez_racine = points[NEZ_RACINE]
    dy = nez_bout.y - nez_racine.y
    return dy  # > 0 = tête penchée vers le bas


def detecter_telephone_yolo(resultats_yolo, hauteur_img):
    """
    Détection simple : confiance + taille minimale + pas trop bas dans le cadre.
    Le filtre principal contre les faux positifs est la proximité avec le visage
    (appliquée après, dans la boucle principale).
    """
    meilleure_box  = None
    meilleure_conf = 0.0
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
            if conf > meilleure_conf:
                meilleure_conf = conf
                meilleure_box  = (x1, y1, x2, y2)
    return (True, meilleure_box) if meilleure_box is not None else (False, None)


def bbox_proche_visage(bbox_tel, bbox_visage, marge=150):
    """
    Retourne True si la bbox du téléphone est dans la zone élargie du visage,
    MAIS rejette les détections qui couvrent le visage lui-même (faux positifs
    où YOLO confond le visage avec un téléphone).
    """
    tx1, ty1, tx2, ty2 = bbox_tel
    fx1, fy1, fx2, fy2 = bbox_visage

    # Rejeter si la bbox téléphone chevauche fortement le visage lui-même
    ix1, iy1 = max(tx1, fx1), max(ty1, fy1)
    ix2, iy2 = min(tx2, fx2), min(ty2, fy2)
    if ix2 > ix1 and iy2 > iy1:
        face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        overlap   = (ix2 - ix1) * (iy2 - iy1)
        if overlap / face_area > 0.40:
            return False

    fx1 -= marge;  fy1 -= marge
    fx2 += marge;  fy2 += marge
    return not (tx2 < fx1 or tx1 > fx2 or ty2 < fy1 or ty1 > fy2)

#  Cadre autour du téléphone détecté avec timer
def dessiner_cadre_telephone(image, bbox, confirme, duree=0.0):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    couleur = ROUGE if confirme else ORANGE

    # Fond semi-transparent
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                  (0, 0, 180) if confirme else (0, 80, 180), -1)
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    # Bordure
    cv2.rectangle(image, (x1, y1), (x2, y2), couleur, 2)

    # Coins accentués
    ep, lg = 3, 20
    for (cx, cy, sx, sy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(image, (cx, cy), (cx + sx * lg, cy), couleur, ep + 1)
        cv2.line(image, (cx, cy), (cx, cy + sy * lg), couleur, ep + 1)

    # Label au-dessus de la box
    label = "TELEPHONE !" if confirme else "Tel. detecte"
    timer_txt = f"  {duree:.1f}s / 3.0s"
    full_label = label + timer_txt
    (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ly = max(y1 - 8, th + 8)
    cv2.rectangle(image, (x1, ly - th - 6), (x1 + tw + 10, ly + 4), couleur, -1)
    cv2.putText(image, full_label, (x1 + 5, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, NOIR, 2)

    # Barre de progression (3 secondes)
    bar_w = max(x2 - x1, 60)
    fill = int(min(duree / SEUIL_TEMPS_TELEPHONE, 1.0) * bar_w)
    bar_y = y2 + 4
    cv2.rectangle(image, (x1, bar_y), (x1 + bar_w, bar_y + 8), (40, 40, 40), -1)
    if fill > 0:
        cv2.rectangle(image, (x1, bar_y), (x1 + fill, bar_y + 8), couleur, -1)
    cv2.rectangle(image, (x1, bar_y), (x1 + bar_w, bar_y + 8), GRIS, 1)

#  Score de fatigue global
def calculer_score_fatigue(ear, mar, clignements_par_min, inclinaison):
    score = 0
    if ear < SEUIL_EAR:          score += 35
    elif ear < 0.28:              score += 12
    if mar > SEUIL_MAR:          score += 25
    elif mar > 0.35:              score += 8
    if clignements_par_min > SEUIL_CLIGNEMENTS: score += 20
    if inclinaison > SEUIL_INCLINAISON:         score += 20
    return min(score, 100)

#  Panneau d'information (barre latérale)
def dessiner_panneau(image, titre, valeur, statut, x, y, largeur=200, hauteur=60):
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), BLEU_FONCE, -1)
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), (60, 60, 60), 1)
    if statut == "ok":
        couleur = VERT
    elif statut == "warning":
        couleur = ORANGE
    else:
        couleur = ROUGE
    cv2.rectangle(image, (x, y), (x + 5, y + hauteur), couleur, -1)
    cv2.putText(image, titre,
        (x + 12, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRIS, 1)
    cv2.putText(image, valeur,
        (x + 12, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

#  Barre de score de fatigue
def dessiner_barre_fatigue(image, score, x, y, largeur=200, hauteur=20):
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), BLEU_FONCE, -1)
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), (60, 60, 60), 1)
    fill = int((score / 100) * largeur)
    if score < 30:
        couleur = VERT
    elif score < 60:
        couleur = ORANGE
    else:
        couleur = ROUGE
    if fill > 0:
        cv2.rectangle(image, (x, y), (x + fill, y + hauteur), couleur, -1)
    cv2.putText(image, f"FATIGUE: {score}%",
        (x + 5, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLANC, 1)

#  Alerte téléphone stylée avec icône et barre de progression
def dessiner_alerte_telephone(image, duree, largeur_video):
    pulse = abs(math.sin(time.time() * 4.0))

    box_w, box_h = 420, 100
    x0 = (largeur_video - box_w) // 2
    y0 = 14
    x1, y1 = x0 + box_w, y0 + box_h

    # Fond semi-transparent
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 160), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    # Bordure pulsante
    bord_intensity = int(180 + pulse * 75)
    cv2.rectangle(image, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3),
                  (0, 0, bord_intensity), 3)
    cv2.rectangle(image, (x0, y0), (x1, y1), (30, 30, 220), 1)

    # Icône téléphone
    ix, iy = x0 + 14, y0 + 10
    cv2.rectangle(image, (ix, iy), (ix + 26, iy + 48), BLANC, 2)
    cv2.rectangle(image, (ix + 4, iy + 6), (ix + 22, iy + 36), (80, 80, 200), -1)
    cv2.circle(image, (ix + 13, iy + 43), 3, BLANC, -1)
    cv2.line(image, (ix + 8, iy + 3), (ix + 18, iy + 3), GRIS, 2)

    # Texte principal
    cv2.putText(image, "TELEPHONE DETECTE !",
                (x0 + 52, y0 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.72, BLANC, 2)

    # Barre de progression
    bar_x, bar_y, bar_w, bar_h = x0 + 52, y0 + 52, 300, 14
    progression = min(duree / SEUIL_TEMPS_TELEPHONE, 1.0)
    fill = int(progression * bar_w)
    bar_color = ORANGE if duree < SEUIL_TEMPS_TELEPHONE else ROUGE
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    if fill > 0:
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GRIS, 1)
    cv2.putText(image, f"{duree:.1f}s",
                (bar_x + bar_w + 8, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLANC, 1)

    # Sous-message une fois confirmé
    if duree >= SEUIL_TEMPS_TELEPHONE:
        msg_color = (int(255 * pulse), int(255 * pulse), 255)
        cv2.putText(image, "  Gardez les yeux sur la route !",
                    (x0 + 52, y0 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, msg_color, 1)

# ---- Configuration MediaPipe (Face Mesh uniquement) ----
mp_face   = mp.solutions.face_mesh

detecteur = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---- Ouvrir la caméra et lire les dimensions réelles ----
camera = cv2.VideoCapture(1)  # 1 = external USB camera (0 = built-in webcam)
_ok, _frame = camera.read()
_cam_h, _cam_w = _frame.shape[:2] if _ok else (480, 640)

# ---- Fonction bouton STOP (coordonnées dynamiques) ----
def clic_sur_stop(event, x, y, _flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # STOP button: full sidebar width, bottom 34px of frame
        if x >= _cam_w and y >= _cam_h - 34:
            param[0] = True

# ---- Variables de suivi ----
temps_yeux          = None
temps_bouche        = None
temps_regard        = None
temps_telephone     = None
alarme_active       = False
arreter             = [False]
dernier_sms         = 0
compteur_alertes    = 0
heure_debut         = time.time()

# ---- Nouvelles variables ----
compteur_clignements      = 0
temps_dernier_clin        = time.time()
clignements_par_min       = 0
oeil_ouvert_precedent     = True
score_fatigue             = 0
historique_ear            = []
HISTORIQUE_MAX            = 100
compteur_frames_telephone = 0    # lissage temporel YOLO
phone_bbox                = None # dernière bbox YOLO

# ---- Drapeaux "alerte déjà lancée" — évite de logger/compter chaque frame ----
alerte_yeux_lancee       = False
alerte_bouche_lancee     = False
alerte_regard_lancee     = False
alerte_telephone_lancee  = False
alarme_raison            = None  # quelle condition a déclenché l'alarme en cours

# ---- Eval mode state ----
EVAL_KEYS = {ord('e'): "YEUX_FERMES", ord('y'): "BAILLEMENT",
             ord('f'): "TETE_AVANT",  ord('p'): "TELEPHONE"}
EVAL_TYPES = list(EVAL_KEYS.values())

eval_open     = {t: False for t in EVAL_TYPES}  # window currently open?
eval_start    = {t: None  for t in EVAL_TYPES}  # when window opened
eval_detected = {t: False for t in EVAL_TYPES}  # detection fired inside window?
eval_tp       = {t: 0 for t in EVAL_TYPES}
eval_fp       = {t: 0 for t in EVAL_TYPES}
eval_fn       = {t: 0 for t in EVAL_TYPES}
eval_feedback = []   # [(message, color, expiry_time)]

def eval_mark_fired(atype):
    """Call this whenever an alert fires. Tracks TP vs FP."""
    if not EVAL_MODE:
        return
    if eval_open[atype]:
        eval_detected[atype] = True   # fired inside window → potential TP
    else:
        eval_fp[atype] += 1           # fired outside window → FP

def eval_close_window(atype):
    """Call when user presses the key a second time to close the window."""
    eval_open[atype]  = False
    eval_start[atype] = None
    if eval_detected[atype]:
        eval_tp[atype] += 1
        eval_feedback.append((f"OK  {atype}", (80, 220, 80), time.time() + 2.5))
    else:
        eval_fn[atype] += 1
        eval_feedback.append((f"MISS  {atype}", (50, 50, 220), time.time() + 2.5))
    eval_detected[atype] = False

WINDOW_NAME = "DriverGuard v1.0 - Driver Monitoring"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, clic_sur_stop, arreter)

while True:

    ok, image = camera.read()
    if not ok:
        break

    hauteur_img, largeur_img = image.shape[:2]
    panneau = 220

    # Vehicle gate — show camera but pause detection when car is stopped
    if not hw.is_active():
        waiting = image.copy()
        cv2.putText(waiting, "En attente du vehicule...",
                    (20, hauteur_img // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 200), 2)
        cv2.imshow(WINDOW_NAME, waiting)
        key = cv2.waitKey(200) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        continue

    cadre = cv2.copyMakeBorder(image, 0, 0, 0, panneau, cv2.BORDER_CONSTANT, value=NOIR)
    x_panneau = largeur_img + 10

    # ---- Logo et titre ----
    cv2.rectangle(cadre, (largeur_img, 0), (largeur_img + panneau, 65), PANEL_HDR, -1)
    cv2.rectangle(cadre, (largeur_img, 0), (largeur_img + 4, 65), VERT, -1)   # accent stripe
    cv2.putText(cadre, "DRIVER",
        (x_panneau, 28), cv2.FONT_HERSHEY_DUPLEX, 0.78, VERT, 2)
    cv2.putText(cadre, "GUARD",
        (x_panneau, 52), cv2.FONT_HERSHEY_DUPLEX, 0.78, BLANC, 2)
    cv2.putText(cadre, "Anti-fatigue system",
        (x_panneau + 2, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRIS, 1)
    cv2.line(cadre, (largeur_img, 66), (largeur_img + panneau, 66), (0, 180, 80), 2)

    # ---- Statuts par défaut ----
    statut_yeux       = "ok"
    statut_bouche     = "ok"
    statut_tete       = "ok"
    statut_telephone  = "ok"
    valeur_yeux       = "Ouverts"
    valeur_bouche     = "Fermee"
    valeur_tete       = "Normal"
    valeur_telephone  = "Aucun"
    ear_moyen         = 0.30
    mar               = 0.0
    angle             = 0.0
    inclinaison       = 0.0

    # ================================================================
    # 1. Face Mesh — fatigue + extraction bbox visage pour filtre YOLO
    # ================================================================
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resultats  = detecteur.process(image_rgb)
    face_bbox_courante = None   # sera utilisé comme filtre pour YOLO

    if resultats.multi_face_landmarks:
        for visage in resultats.multi_face_landmarks:
            points = visage.landmark

            # Bbox du visage en pixels (pour filtre de proximité téléphone)
            face_xs = [int(lm.x * largeur_img) for lm in points]
            face_ys = [int(lm.y * hauteur_img) for lm in points]
            face_bbox_courante = (min(face_xs), min(face_ys),
                                  max(face_xs), max(face_ys))

            ear_gauche = calculer_ear(points, OEIL_GAUCHE)
            ear_droit  = calculer_ear(points, OEIL_DROIT)
            ear_moyen  = (ear_gauche + ear_droit) / 2.0
            mar        = calculer_mar(points, BOUCHE)
            angle      = calculer_angle_tete(points)
            inclinaison = calculer_inclinaison_verticale(points)

            # ---- Historique EAR pour mini-graphique ----
            historique_ear.append(ear_moyen)
            if len(historique_ear) > HISTORIQUE_MAX:
                historique_ear.pop(0)

            # ---- Comptage des clignements ----
            oeil_ouvert = ear_moyen >= SEUIL_EAR
            if not oeil_ouvert and oeil_ouvert_precedent:
                compteur_clignements += 1
            oeil_ouvert_precedent = oeil_ouvert

            elapsed = time.time() - temps_dernier_clin
            if elapsed >= 10:
                clignements_par_min = int(compteur_clignements * (60 / elapsed))
                compteur_clignements = 0
                temps_dernier_clin = time.time()

            # ---- Score de fatigue ----
            score_fatigue = calculer_score_fatigue(
                ear_moyen, mar, clignements_par_min, inclinaison)

            # ---- Détection fatigue (yeux) ----
            if ear_moyen < SEUIL_EAR:
                if temps_yeux is None:
                    temps_yeux = time.time()
                duree_yeux = time.time() - temps_yeux
                statut_yeux = "warning" if duree_yeux < SEUIL_TEMPS_YEUX else "danger"
                valeur_yeux = f"Fermes {duree_yeux:.1f}s"
                if duree_yeux >= SEUIL_TEMPS_YEUX and not alerte_yeux_lancee:
                    alerte_yeux_lancee = True
                    eval_mark_fired("YEUX_FERMES")
                    compteur_alertes += 1
                    log_alerte("YEUX_FERMES", ear_moyen, mar, angle, duree_yeux)
                    pygame.mixer.music.play(-1)
                    hw.send_alarm()
                    alarme_active = True
                    alarme_raison  = "YEUX"
                    if time.time() - dernier_sms > DELAI_SMS:
                        envoyer_sms("ALERTE DriverGuard : Conducteur fatigué ! Yeux fermés depuis 2 secondes.")
                        dernier_sms = time.time()
            else:
                if alerte_yeux_lancee and alarme_raison == "YEUX":
                    pygame.mixer.music.stop()
                    alarme_active = False
                    alarme_raison = None
                alerte_yeux_lancee = False
                temps_yeux = None

            # ---- Détection bâillement ----
            if mar > SEUIL_MAR:
                if temps_bouche is None:
                    temps_bouche = time.time()
                duree_bouche = time.time() - temps_bouche
                statut_bouche = "warning" if duree_bouche < SEUIL_TEMPS_BOUCHE else "danger"
                valeur_bouche = f"Ouverte {duree_bouche:.1f}s"
                if duree_bouche >= SEUIL_TEMPS_BOUCHE and not alerte_bouche_lancee:
                    alerte_bouche_lancee = True
                    eval_mark_fired("BAILLEMENT")
                    compteur_alertes += 1
                    log_alerte("BAILLEMENT", ear_moyen, mar, angle, duree_bouche)
                    if not alarme_active:
                        pygame.mixer.music.play(-1)
                        hw.send_alarm()
                        alarme_active = True
                        alarme_raison = "BOUCHE"
                    if time.time() - dernier_sms > DELAI_SMS:
                        envoyer_sms("ALERTE DriverGuard : Conducteur bâille ! Signes de fatigue détectés.")
                        dernier_sms = time.time()
            else:
                if alerte_bouche_lancee and alarme_raison == "BOUCHE":
                    pygame.mixer.music.stop()
                    alarme_active = False
                    alarme_raison = None
                alerte_bouche_lancee = False
                temps_bouche = None

            # ---- Détection tête vers l'avant (microsommeil) ----
            if inclinaison > SEUIL_INCLINAISON:
                if temps_regard is None:
                    temps_regard = time.time()
                duree_regard = time.time() - temps_regard
                statut_tete = "warning" if duree_regard < SEUIL_TEMPS_AVANT else "danger"
                valeur_tete = f"Avant {duree_regard:.1f}s"
                if duree_regard >= SEUIL_TEMPS_AVANT and not alerte_regard_lancee:
                    alerte_regard_lancee = True
                    eval_mark_fired("TETE_AVANT")
                    compteur_alertes += 1
                    log_alerte("TETE_AVANT", ear_moyen, mar, 0, duree_regard)
                    if not alarme_active:
                        pygame.mixer.music.play(-1)
                        hw.send_alarm()
                        alarme_active = True
                        alarme_raison = "REGARD"
                    if time.time() - dernier_sms > DELAI_SMS:
                        envoyer_sms("ALERTE DriverGuard : Tête du conducteur inclinée vers l'avant !")
                        dernier_sms = time.time()
            else:
                if alerte_regard_lancee and alarme_raison == "REGARD":
                    pygame.mixer.music.stop()
                    alarme_active = False
                    alarme_raison = None
                alerte_regard_lancee = False
                temps_regard = None

            # ---- EAR et MAR sur la vidéo ----
            couleur_ear = ROUGE if ear_moyen < SEUIL_EAR else VERT
            couleur_mar = ROUGE if mar > SEUIL_MAR else VERT
            label_ear   = "FERMES !" if ear_moyen < SEUIL_EAR else "Ouverts"
            label_mar   = "BAILLEMENT !" if mar > SEUIL_MAR else "Normal"

            cv2.rectangle(cadre, (5, hauteur_img - 175), (320, hauteur_img - 150), (20, 20, 20), -1)
            cv2.putText(cadre, f"EAR: {ear_moyen:.2f}  ->  {label_ear}",
                (10, hauteur_img - 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_ear, 2)

            cv2.rectangle(cadre, (5, hauteur_img - 148), (320, hauteur_img - 123), (20, 20, 20), -1)
            cv2.putText(cadre, f"MAR: {mar:.2f}  ->  {label_mar}",
                (10, hauteur_img - 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_mar, 2)

            cv2.putText(cadre, f"Clin/min: {clignements_par_min}",
                (10, hauteur_img - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRIS, 1)


    else:
        statut_yeux   = "danger"
        valeur_yeux   = "Non detecte"
        statut_bouche = "danger"
        valeur_bouche = "Non detecte"

        # Face out of frame → treat as head drop forward
        if temps_regard is None:
            temps_regard = time.time()
        duree_absence = time.time() - temps_regard
        statut_tete = "warning" if duree_absence < SEUIL_TEMPS_AVANT else "danger"
        valeur_tete = f"Avant {duree_absence:.1f}s"
        if duree_absence >= SEUIL_TEMPS_AVANT and not alerte_regard_lancee:
            alerte_regard_lancee = True
            eval_mark_fired("TETE_AVANT")
            compteur_alertes += 1
            log_alerte("TETE_AVANT", 0, 0, 0, duree_absence)
            if not alarme_active:
                pygame.mixer.music.play(-1)
                hw.send_alarm()
                alarme_active = True
                alarme_raison = "REGARD"
            if time.time() - dernier_sms > DELAI_SMS:
                envoyer_sms("ALERTE DriverGuard : Tête du conducteur hors champ !")
                dernier_sms = time.time()

    # ================================================================
    # 2. YOLO — détection téléphone + filtre proximité visage
    # ================================================================
    res_yolo = modele_yolo.predict(image, verbose=False,
                                   imgsz=416, conf=YOLO_CONFIANCE_MIN, device='cpu')
    phone_raw, phone_bbox_new = detecter_telephone_yolo(res_yolo, hauteur_img)

    # Filtre proximité : le téléphone doit être proche du visage détecté
    if phone_raw and face_bbox_courante is not None:
        if not bbox_proche_visage(phone_bbox_new, face_bbox_courante):
            phone_raw      = False
            phone_bbox_new = None

    # Conserver la bbox visible
    if phone_bbox_new is not None:
        phone_bbox = phone_bbox_new

    # Lissage temporel
    if phone_raw:
        compteur_frames_telephone = min(compteur_frames_telephone + 1, 10)
        if temps_telephone is None:
            temps_telephone = time.time()
    else:
        compteur_frames_telephone = max(compteur_frames_telephone - 1, 0)
        if compteur_frames_telephone == 0:
            phone_bbox      = None
            temps_telephone = None

    phone_detected = compteur_frames_telephone >= SEUIL_FRAMES_TELEPHONE

    # Durée depuis la 1ère détection (sert au timer visuel ET à l'alerte)
    duree_telephone = (time.time() - temps_telephone) if temps_telephone is not None else 0.0

    # Cadre visuel sur le téléphone avec timer
    dessiner_cadre_telephone(cadre, phone_bbox, phone_detected, duree_telephone)

    # ---- Alerte et panneau téléphone ----
    if phone_detected:
        statut_telephone = "warning" if duree_telephone < SEUIL_TEMPS_TELEPHONE else "danger"
        valeur_telephone = f"Detecte {duree_telephone:.1f}s"
        dessiner_alerte_telephone(cadre, duree_telephone, largeur_img)
        if duree_telephone >= SEUIL_TEMPS_TELEPHONE and not alerte_telephone_lancee:
            alerte_telephone_lancee = True
            eval_mark_fired("TELEPHONE")
            compteur_alertes += 1
            log_alerte("TELEPHONE", ear_moyen, mar, angle, duree_telephone)
            if not alarme_active:
                pygame.mixer.music.play(-1)
                hw.send_alarm()
                alarme_active = True
                alarme_raison = "TELEPHONE"
            if time.time() - dernier_sms > DELAI_SMS:
                envoyer_sms("ALERTE DriverGuard : Conducteur utilise un téléphone !")
                dernier_sms = time.time()
    else:
        if alerte_telephone_lancee and alarme_raison == "TELEPHONE":
            pygame.mixer.music.stop()
            alarme_active = False
            alarme_raison = None
        alerte_telephone_lancee = False

    # ---- Panneaux d'information (fixed y — no overlap at any resolution) ----
    dessiner_panneau(cadre, "EYES",      valeur_yeux,      statut_yeux,      x_panneau,  68)
    dessiner_panneau(cadre, "MOUTH",     valeur_bouche,    statut_bouche,    x_panneau, 132)
    dessiner_panneau(cadre, "HEAD",      valeur_tete,      statut_tete,      x_panneau, 196)
    dessiner_panneau(cadre, "PHONE",     valeur_telephone, statut_telephone, x_panneau, 260)

    # ---- Durée de session ----
    duree_session = int(time.time() - heure_debut)
    minutes  = duree_session // 60
    secondes = duree_session % 60
    dessiner_panneau(cadre, "SESSION",
        f"{minutes:02d}:{secondes:02d}", "ok", x_panneau, 324)

    # ---- Barre score fatigue ----
    dessiner_barre_fatigue(cadre, score_fatigue, x_panneau, 390)

    # ---- Statut alarme ----
    if alarme_active:
        cv2.rectangle(cadre, (largeur_img, 412),
            (largeur_img + panneau, 446), ROUGE, -1)
        cv2.putText(cadre, "ALARME ACTIVE !",
            (x_panneau, 436),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANC, 2)
    else:
        cv2.rectangle(cadre, (largeur_img, 412),
            (largeur_img + panneau, 446), (0, 80, 0), -1)
        cv2.putText(cadre, "Systeme actif",
            (x_panneau, 436),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, VERT, 2)

    # ---- Bouton STOP ----
    cv2.rectangle(cadre, (largeur_img, hauteur_img - 34),
        (largeur_img + panneau, hauteur_img), ROUGE, -1)
    cv2.putText(cadre, "STOP",
        (x_panneau + 40, hauteur_img - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLANC, 2)

    # ---- Eval mode overlay ----
    if EVAL_MODE:
        now = time.time()

        # Active windows — shown at top-left of video
        ey = 20
        for atype, is_open in eval_open.items():
            if is_open:
                dur = now - eval_start[atype]
                cv2.rectangle(cadre, (0, ey - 16), (370, ey + 6), (0, 60, 0), -1)
                cv2.putText(cadre, f"TESTING {atype}  [{dur:.1f}s]",
                            (6, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2)
                ey += 28

        # Feedback messages (OK / MISS)
        eval_feedback[:] = [f for f in eval_feedback if f[2] > now]
        for i, (msg, color, _) in enumerate(eval_feedback):
            cv2.rectangle(cadre, (0, hauteur_img//2 - 22 + i*36),
                          (420, hauteur_img//2 + 14 + i*36), (20, 20, 20), -1)
            cv2.putText(cadre, msg,
                        (8, hauteur_img//2 + 8 + i*36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

        # Key hints at bottom of video
        cv2.rectangle(cadre, (0, hauteur_img - 22), (largeur_img, hauteur_img), (20,20,20), -1)
        cv2.putText(cadre, "EVAL:  E=eyes  Y=yawn  F=head-fwd  P=phone",
                    (6, hauteur_img - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

    cv2.imshow(WINDOW_NAME, cadre)

    key = cv2.waitKey(1) & 0xFF

    # ---- Eval mode key handling ----
    if EVAL_MODE and key in EVAL_KEYS:
        atype = EVAL_KEYS[key]
        if not eval_open[atype]:
            eval_open[atype]     = True
            eval_start[atype]    = time.time()
            eval_detected[atype] = False
            print(f"[EVAL] Window opened: {atype}")
        else:
            print(f"[EVAL] Window closed: {atype} — "
                  f"{'DETECTED ✓' if eval_detected[atype] else 'MISSED ✗'}")
            eval_close_window(atype)

    if (key == ord('q') or key == 27           # Q or ESC
            or arreter[0]                       # STOP button clicked
            or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1):  # X button
        break

# ---- Rapport de fin de session ----
duree_totale = int(time.time() - heure_debut)
print(f"\n{'='*52}")
print(f"  SESSION REPORT — DriverGuard v1.0")
print(f"{'='*52}")
print(f"  Duration     : {duree_totale // 60}m {duree_totale % 60}s")
print(f"  Alerts fired : {compteur_alertes}")
print(f"  Log saved    : {LOG_FILE}")
print(f"{'='*52}\n")

camera.release()
cv2.destroyAllWindows()
pygame.mixer.quit()


# ---- Eval results ----
if EVAL_MODE:
    print(f"\n{'='*58}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*58}")
    print(f"  {'Alert':<20} {'Prec':>7} {'Recall':>7} {'F1':>7}"
          f"  {'TP':>3} {'FP':>3} {'FN':>3}")
    print(f"  {'-'*54}")
    totals = [0.0, 0.0, 0.0]
    count  = 0
    for atype in EVAL_TYPES:
        tp = eval_tp[atype]; fp = eval_fp[atype]; fn = eval_fn[atype]
        if tp + fp + fn == 0:
            continue
        p  = tp/(tp+fp) if tp+fp > 0 else 0.0
        r  = tp/(tp+fn) if tp+fn > 0 else 0.0
        f1 = 2*p*r/(p+r) if p+r > 0 else 0.0
        print(f"  {atype:<20} {p:>6.0%} {r:>7.0%} {f1:>7.0%}"
              f"  {tp:>3} {fp:>3} {fn:>3}")
        totals[0] += p; totals[1] += r; totals[2] += f1
        count += 1
    if count:
        print(f"  {'-'*54}")
        print(f"  {'MACRO AVERAGE':<20} {totals[0]/count:>6.0%}"
              f" {totals[1]/count:>7.0%} {totals[2]/count:>7.0%}")
    print(f"{'='*58}\n")

    # Save results to CSV
    eval_csv = os.path.join(LOG_DIR,
                            f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alert_type", "precision", "recall", "f1", "tp", "fp", "fn"])
        for atype in EVAL_TYPES:
            tp = eval_tp[atype]; fp = eval_fp[atype]; fn = eval_fn[atype]
            if tp + fp + fn == 0:
                continue
            p  = tp/(tp+fp) if tp+fp > 0 else 0.0
            r  = tp/(tp+fn) if tp+fn > 0 else 0.0
            f1 = 2*p*r/(p+r) if p+r > 0 else 0.0
            w.writerow([atype, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", tp, fp, fn])
    print(f"  Eval results saved: {eval_csv}")

hw.disconnect()
print("Programme arrêté proprement ✅")

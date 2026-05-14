# ============================================================
#  DriverGuard — Main entry point
#  Run: python main.py
# ============================================================
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import cv2
import mediapipe as mp
import time

from config import GMAIL_EXPEDITEUR, GMAIL_MOT_PASSE, GMAIL_DESTINATAIRE

from detection.fatigue import (FatigueDetector,
                                SEUIL_EAR, SEUIL_MAR, SEUIL_ANGLE,
                                SEUIL_TEMPS_YEUX, SEUIL_TEMPS_BOUCHE,
                                SEUIL_TEMPS_TETE, SEUIL_TEMPS_REGARD)
from detection.phone   import PhoneDetector, SEUIL_TEMPS_TELEPHONE
from alerts.logger      import init_log, log_alerte
from alerts.audio       import jouer as alarme_jouer, arreter as alarme_arreter, quitter as alarme_quitter
from alerts.email_alert import EmailAlerter
from ui.dashboard       import (dessiner_header, dessiner_panneau, dessiner_barre_fatigue,
                                 dessiner_cadre_telephone, dessiner_alerte_telephone,
                                 dessiner_ear_info, dessiner_alarme_statut,
                                 NOIR, BLANC, ROUGE, VERT, GRIS)

# ---- Startup banner ----
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

# ---- Initialise subsystems ----
log_file = init_log()
emailer  = EmailAlerter(GMAIL_EXPEDITEUR, GMAIL_MOT_PASSE, GMAIL_DESTINATAIRE)

print("[*] Loading YOLO phone detection model...")
phone_detector = PhoneDetector()
print("[+] YOLO model ready.")

mp_face   = mp.solutions.face_mesh
detecteur = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
fatigue_detector = FatigueDetector()

print("[*] Starting camera feed...\n")
camera = cv2.VideoCapture(0)

# ---- Session state ----
alarme_active    = False
arreter          = [False]
compteur_alertes = 0
heure_debut      = time.time()

PANNEAU_W = 220


def clic_sur_stop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        largeur = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        if x >= largeur and y >= camera.get(cv2.CAP_PROP_FRAME_HEIGHT) - 50:
            param[0] = True


cv2.namedWindow("DriverGuard v1.0 — Driver Monitoring")
cv2.setMouseCallback("DriverGuard v1.0 — Driver Monitoring", clic_sur_stop, arreter)

# ---- Main loop ----
while True:
    if arreter[0]:
        break

    ok, image = camera.read()
    if not ok:
        break

    h, w = image.shape[:2]
    cadre    = cv2.copyMakeBorder(image, 0, 0, 0, PANNEAU_W,
                                  cv2.BORDER_CONSTANT, value=NOIR)
    x_panel  = w + 10

    dessiner_header(cadre, w, PANNEAU_W, x_panel)

    # ---- Defaults (shown when no face detected) ----
    statut_yeux      = statut_bouche = statut_tete = statut_telephone = "ok"
    valeur_yeux      = "Ouverts"
    valeur_bouche    = "Fermee"
    valeur_tete      = "Droite"
    valeur_telephone = "Aucun"
    ear_moyen = 0.30
    mar = angle = 0.0
    score_fatigue = 0
    face_bbox = None
    alerts = {}   # safe default — avoids NameError when no face is detected

    # ---- Face mesh ----
    rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results  = detecteur.process(rgb)

    if results.multi_face_landmarks:
        pts = results.multi_face_landmarks[0].landmark

        xs = [int(lm.x * w) for lm in pts]
        ys = [int(lm.y * h) for lm in pts]
        face_bbox = (min(xs), min(ys), max(xs), max(ys))

        state = fatigue_detector.process_frame(pts)
        ear_moyen     = state["ear"]
        mar           = state["mar"]
        angle         = state["angle"]
        score_fatigue = state["score"]
        alerts        = state["alerts"]

        dessiner_ear_info(cadre, ear_moyen, mar, state["clignements_par_min"],
                          state["historique_ear"], h, SEUIL_EAR, SEUIL_MAR)

        # ---- Eyes ----
        dur = alerts.get("YEUX_FERMES")
        if dur is not None:
            statut_yeux = "warning" if dur < SEUIL_TEMPS_YEUX else "danger"
            valeur_yeux = f"Fermes {dur:.1f}s"
            if dur >= SEUIL_TEMPS_YEUX:
                compteur_alertes += 1
                log_alerte(log_file, "YEUX_FERMES", ear_moyen, mar, angle, dur)
                if not alarme_active:
                    alarme_jouer(); alarme_active = True
                emailer.envoyer("ALERTE DriverGuard : Conducteur fatigué ! Yeux fermés depuis 2 secondes.")

        # ---- Yawn ----
        dur = alerts.get("BAILLEMENT")
        if dur is not None:
            statut_bouche = "warning" if dur < SEUIL_TEMPS_BOUCHE else "danger"
            valeur_bouche = f"Ouverte {dur:.1f}s"
            if dur >= SEUIL_TEMPS_BOUCHE:
                compteur_alertes += 1
                log_alerte(log_file, "BAILLEMENT", ear_moyen, mar, angle, dur)
                if not alarme_active:
                    alarme_jouer(); alarme_active = True
                emailer.envoyer("ALERTE DriverGuard : Conducteur bâille ! Signes de fatigue détectés.")

        # ---- Head tilt ----
        dur = alerts.get("TETE_PENCHEE")
        if dur is not None:
            cote = "Gauche" if angle > 0 else "Droite"
            statut_tete = "warning" if dur < SEUIL_TEMPS_TETE else "danger"
            valeur_tete = f"Penchee {cote}"
            if dur >= SEUIL_TEMPS_TETE:
                compteur_alertes += 1
                log_alerte(log_file, "TETE_PENCHEE", ear_moyen, mar, angle, dur)
                if not alarme_active:
                    alarme_jouer(); alarme_active = True
                emailer.envoyer("ALERTE DriverGuard : Tête du conducteur penche ! Risque de somnolence.")

        # ---- Head drop forward ----
        dur = alerts.get("TETE_AVANT")
        if dur is not None and dur >= SEUIL_TEMPS_REGARD:
            compteur_alertes += 1
            log_alerte(log_file, "TETE_AVANT", ear_moyen, mar, angle, dur)
            if not alarme_active:
                alarme_jouer(); alarme_active = True
            emailer.envoyer("ALERTE DriverGuard : Tête du conducteur inclinée vers l'avant !")

        # ---- Reset alarm when all conditions clear ----
        all_clear = (alerts.get("YEUX_FERMES") is None and
                     alerts.get("BAILLEMENT")  is None and
                     alerts.get("TETE_PENCHEE") is None)
        if alarme_active and all_clear:
            alarme_arreter()
            alarme_active = False

    else:
        statut_yeux = statut_bouche = statut_tete = "danger"
        valeur_yeux = valeur_bouche = valeur_tete = "Non detecte"

    # ---- Phone detection ----
    phone_state = phone_detector.process_frame(image, h, face_bbox)
    dessiner_cadre_telephone(cadre, phone_state["bbox"],
                              phone_state["confirme"], phone_state["duree_s"])

    if phone_state["detected"]:
        dur = phone_state["duree_s"]
        statut_telephone = "warning" if dur < SEUIL_TEMPS_TELEPHONE else "danger"
        valeur_telephone = f"Detecte {dur:.1f}s"
        dessiner_alerte_telephone(cadre, dur, w)
        if dur >= SEUIL_TEMPS_TELEPHONE:
            compteur_alertes += 1
            log_alerte(log_file, "TELEPHONE", ear_moyen, mar, angle, dur)
            if not alarme_active:
                alarme_jouer(); alarme_active = True
            emailer.envoyer("ALERTE DriverGuard : Conducteur utilise un téléphone !")
    else:
        if alarme_active and alerts.get("YEUX_FERMES") is None:
            alarme_arreter()
            alarme_active = False

    # ---- Sidebar panels (fixed y — no overlap at any resolution) ----
    dessiner_panneau(cadre, "EYES",  valeur_yeux,      statut_yeux,      x_panel,  68)
    dessiner_panneau(cadre, "MOUTH", valeur_bouche,    statut_bouche,    x_panel, 132)
    dessiner_panneau(cadre, "HEAD",  valeur_tete,      statut_tete,      x_panel, 196)
    dessiner_panneau(cadre, "PHONE", valeur_telephone, statut_telephone, x_panel, 260)

    duree_s = int(time.time() - heure_debut)
    dessiner_panneau(cadre, "SESSION",
                     f"{duree_s // 60:02d}:{duree_s % 60:02d}", "ok", x_panel, 324)

    dessiner_barre_fatigue(cadre, score_fatigue, x_panel, 390)
    dessiner_alarme_statut(cadre, alarme_active, w, PANNEAU_W, h, x_panel)

    # ---- STOP button ----
    cv2.rectangle(cadre, (w, h - 34), (w + PANNEAU_W, h), ROUGE, -1)
    cv2.putText(cadre, "STOP", (x_panel + 40, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLANC, 2)

    cv2.imshow("DriverGuard v1.0 — Driver Monitoring", cadre)
    if cv2.waitKey(1) == ord('q'):
        break

# ---- Session report ----
duree_totale = int(time.time() - heure_debut)
print(f"\n{'='*52}")
print(f"  SESSION REPORT — DriverGuard v1.0")
print(f"{'='*52}")
print(f"  Duration     : {duree_totale // 60}m {duree_totale % 60}s")
print(f"  Alerts fired : {compteur_alertes}")
print(f"  Log saved    : {log_file}")
print(f"{'='*52}\n")

camera.release()
cv2.destroyAllWindows()
alarme_quitter()
print("Programme arrêté proprement ✅")

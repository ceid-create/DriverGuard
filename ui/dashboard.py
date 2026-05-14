"""All OpenCV drawing functions for the DriverGuard dashboard."""
import cv2
import math
import time

# Colour palette (BGR)
NOIR       = (10,  10,  10)
BLANC      = (240, 240, 240)
ROUGE      = (40,  40,  220)
VERT       = (80,  220, 80)
ORANGE     = (0,   160, 255)
JAUNE      = (0,   230, 230)
BLEU_FONCE = (45,  35,  25)
PANEL_HDR  = (60,  40,  20)
GRIS       = (130, 130, 130)
CYAN       = (220, 220, 20)

SEUIL_TEMPS_TELEPHONE = 3.0


def dessiner_panneau(image, titre, valeur, statut, x, y, largeur=200, hauteur=60):
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), BLEU_FONCE, -1)
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), (60, 60, 60), 1)
    couleur = VERT if statut == "ok" else ORANGE if statut == "warning" else ROUGE
    cv2.rectangle(image, (x, y), (x + 5, y + hauteur), couleur, -1)
    cv2.putText(image, titre,  (x + 12, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRIS, 1)
    cv2.putText(image, valeur, (x + 12, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  couleur, 2)


def dessiner_barre_fatigue(image, score, x, y, largeur=200, hauteur=20):
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), BLEU_FONCE, -1)
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), (60, 60, 60), 1)
    fill = int((score / 100) * largeur)
    couleur = VERT if score < 30 else ORANGE if score < 60 else ROUGE
    if fill > 0:
        cv2.rectangle(image, (x, y), (x + fill, y + hauteur), couleur, -1)
    cv2.putText(image, f"FATIGUE: {score}%",
                (x + 5, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLANC, 1)


def dessiner_cadre_telephone(image, bbox, confirme, duree=0.0):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    couleur = ROUGE if confirme else ORANGE
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                  (0, 0, 180) if confirme else (0, 80, 180), -1)
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), couleur, 2)
    ep, lg = 3, 20
    for (cx, cy, sx, sy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(image, (cx, cy), (cx + sx * lg, cy), couleur, ep + 1)
        cv2.line(image, (cx, cy), (cx, cy + sy * lg), couleur, ep + 1)
    label = "TELEPHONE !" if confirme else "Tel. detecte"
    full_label = label + f"  {duree:.1f}s / 3.0s"
    (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ly = max(y1 - 8, th + 8)
    cv2.rectangle(image, (x1, ly - th - 6), (x1 + tw + 10, ly + 4), couleur, -1)
    cv2.putText(image, full_label, (x1 + 5, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, NOIR, 2)
    bar_w = max(x2 - x1, 60)
    fill  = int(min(duree / SEUIL_TEMPS_TELEPHONE, 1.0) * bar_w)
    bar_y = y2 + 4
    cv2.rectangle(image, (x1, bar_y), (x1 + bar_w, bar_y + 8), (40, 40, 40), -1)
    if fill > 0:
        cv2.rectangle(image, (x1, bar_y), (x1 + fill, bar_y + 8), couleur, -1)
    cv2.rectangle(image, (x1, bar_y), (x1 + bar_w, bar_y + 8), GRIS, 1)


def dessiner_alerte_telephone(image, duree, largeur_video):
    pulse = abs(math.sin(time.time() * 4.0))
    box_w, box_h = 420, 100
    x0 = (largeur_video - box_w) // 2
    y0 = 14
    x1, y1 = x0 + box_w, y0 + box_h
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 160), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    bord = int(180 + pulse * 75)
    cv2.rectangle(image, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 0, bord), 3)
    cv2.rectangle(image, (x0, y0), (x1, y1), (30, 30, 220), 1)
    ix, iy = x0 + 14, y0 + 10
    cv2.rectangle(image, (ix, iy), (ix + 26, iy + 48), BLANC, 2)
    cv2.rectangle(image, (ix + 4, iy + 6), (ix + 22, iy + 36), (80, 80, 200), -1)
    cv2.circle(image, (ix + 13, iy + 43), 3, BLANC, -1)
    cv2.line(image, (ix + 8, iy + 3), (ix + 18, iy + 3), GRIS, 2)
    cv2.putText(image, "TELEPHONE DETECTE !",
                (x0 + 52, y0 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.72, BLANC, 2)
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
    if duree >= SEUIL_TEMPS_TELEPHONE:
        msg_color = (int(255 * pulse), int(255 * pulse), 255)
        cv2.putText(image, "  Gardez les yeux sur la route !",
                    (x0 + 52, y0 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, msg_color, 1)


def dessiner_header(cadre, largeur_img, panneau, x_panneau):
    cv2.rectangle(cadre, (largeur_img, 0), (largeur_img + panneau, 65), PANEL_HDR, -1)
    cv2.rectangle(cadre, (largeur_img, 0), (largeur_img + 4, 65), VERT, -1)
    cv2.putText(cadre, "DRIVER", (x_panneau, 28), cv2.FONT_HERSHEY_DUPLEX, 0.78, VERT,  2)
    cv2.putText(cadre, "GUARD",  (x_panneau, 52), cv2.FONT_HERSHEY_DUPLEX, 0.78, BLANC, 2)
    cv2.putText(cadre, "Anti-fatigue system",
                (x_panneau + 2, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRIS, 1)
    cv2.line(cadre, (largeur_img, 66), (largeur_img + panneau, 66), (0, 180, 80), 2)


def dessiner_ear_info(cadre, ear, mar, clignements_par_min,
                      historique_ear, hauteur_img, seuil_ear, seuil_mar):
    couleur_ear = ROUGE if ear < seuil_ear else VERT
    couleur_mar = ROUGE if mar > seuil_mar else VERT
    label_ear   = "FERMES !" if ear < seuil_ear else "Ouverts"
    label_mar   = "BAILLEMENT !" if mar > seuil_mar else "Normal"

    cv2.rectangle(cadre, (5, hauteur_img - 100), (320, hauteur_img - 75), (20, 20, 20), -1)
    cv2.putText(cadre, f"EAR: {ear:.2f}  ->  {label_ear}",
                (10, hauteur_img - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_ear, 2)
    cv2.rectangle(cadre, (5, hauteur_img - 72), (320, hauteur_img - 47), (20, 20, 20), -1)
    cv2.putText(cadre, f"MAR: {mar:.2f}  ->  {label_mar}",
                (10, hauteur_img - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_mar, 2)
    cv2.putText(cadre, f"Clin/min: {clignements_par_min}",
                (10, hauteur_img - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRIS, 1)


def dessiner_alarme_statut(cadre, alarme_active, largeur_img, panneau, hauteur_img, x_panneau):
    if alarme_active:
        cv2.rectangle(cadre, (largeur_img, 412), (largeur_img + panneau, 446), ROUGE, -1)
        cv2.putText(cadre, "ALARME ACTIVE !",
                    (x_panneau, 436), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANC, 2)
    else:
        cv2.rectangle(cadre, (largeur_img, 412), (largeur_img + panneau, 446), (0, 80, 0), -1)
        cv2.putText(cadre, "Systeme actif",
                    (x_panneau, 436), cv2.FONT_HERSHEY_SIMPLEX, 0.5, VERT, 2)

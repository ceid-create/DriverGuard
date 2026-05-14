"""MediaPipe landmark indices and geometry helpers."""
import math

# Eye landmark indices (6-point EAR model)
OEIL_GAUCHE = [362, 385, 387, 263, 373, 380]
OEIL_DROIT  = [33,  160, 158, 133, 153, 144]

# Mouth landmark indices (4-point MAR model)
BOUCHE = [13, 14, 78, 308]

# Head pose reference points
COIN_DROIT  = 33
COIN_GAUCHE = 263
NEZ_BOUT    = 4
NEZ_RACINE  = 6


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

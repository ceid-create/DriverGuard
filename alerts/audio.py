"""Pygame alarm audio management."""
import os
import pygame

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

_loaded = False

def charger_alarme():
    global _loaded
    if not _loaded:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "alarme.mp3")
        pygame.mixer.music.load(path)
        _loaded = True


def jouer():
    charger_alarme()
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)


def arreter():
    pygame.mixer.music.stop()


def quitter():
    pygame.mixer.quit()

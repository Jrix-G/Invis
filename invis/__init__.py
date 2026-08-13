"""Station sol vision pour le pont ESP32-CAM.

Lecture seule: ce paquet consomme le flux HTTP de la carte et affiche une
analyse. Il n'ecrit jamais sur les endpoints de pilotage.
"""

__all__ = ["config", "mjpeg_client", "detector", "overlay", "recorder"]

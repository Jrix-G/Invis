"""Version du code applicatif.

Ce numero identifie le *payload* -- le code Python -- et non l'executable qui
l'heberge. Les deux evoluent a des rythmes tres differents: l'executable
embarque Python, OpenCV et numpy (environ 150 Mo) et ne change que rarement,
alors que le code du projet pese 200 Ko et change souvent. Separer les deux
permet de livrer une correction en telechargeant 200 Ko au lieu de 150 Mo.

Format: MAJEUR.MINEUR.CORRECTIF, compare numeriquement.
"""

from __future__ import annotations

from typing import Tuple

VERSION = "1.0.2"
APP_NAME = "esp32cam-vision"


def parse(version: str) -> Tuple[int, ...]:
    """Convertit "1.2.3" en (1, 2, 3), pour une comparaison numerique.

    Comparer des chaines donnerait "1.10.0" < "1.9.0", ce qui ferait rater
    des mises a jour au dixieme incrementiel.
    """
    parts = []
    for chunk in version.strip().split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def is_newer(candidate: str, reference: str) -> bool:
    return parse(candidate) > parse(reference)

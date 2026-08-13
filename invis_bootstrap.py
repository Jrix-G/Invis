"""Choix du code a executer au demarrage: embarque ou mis a jour.

L'executable embarque une version du code. Les mises a jour en installent
d'autres dans l'espace utilisateur. Ce module choisit la plus recente et la
place devant les autres dans le chemin d'import, avant que le paquet `invis`
ne soit importe.

Pourquoi ce fichier vit hors du paquet `invis`
----------------------------------------------
Deux raisons, et les deux comptent.

La premiere est technique: des qu'on importe quoi que ce soit depuis `invis`,
Python fixe l'emplacement du paquet. Modifier le chemin d'import ensuite n'a
plus aucun effet -- le code embarque serait charge quoi qu'il arrive. Ce
module n'importe donc rien du projet.

La seconde tient a la robustesse: ce fichier n'est jamais mis a jour, puisqu'il
n'est pas dans les archives publiees. Une version defectueuse ne peut donc pas
casser le demarrage de facon definitive; il reste toujours possible de revenir
au code embarque dans l'executable.

Toute anomalie -- dossier illisible, version absente, contenu incoherent --
fait silencieusement retomber sur le code embarque. Un demarrage qui
fonctionne vaut mieux qu'un message d'erreur juste.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Optional, Tuple

APP_NAME = "esp32cam-vision"
_VERSION_RE = re.compile(r"""^VERSION\s*=\s*["']([^"']+)["']""", re.MULTILINE)


def _parse(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in version.strip().split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _read_version(version_file: str) -> Optional[str]:
    """Lit VERSION dans un fichier version.py, sans l'importer.

    Importer le fichier executerait du code venu du reseau avant meme d'avoir
    decide s'il faut lui faire confiance. Une lecture de texte suffit.
    """
    try:
        with open(version_file, encoding="utf-8") as fh:
            match = _VERSION_RE.search(fh.read(4096))
    except OSError:
        return None
    return match.group(1) if match else None


def data_dir(app_name: str = APP_NAME) -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, app_name)


def bundled_version(bundle_root: Optional[str] = None) -> Optional[str]:
    root = bundle_root or getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return _read_version(os.path.join(root, "invis", "version.py"))


def candidates(payload_root: str) -> List[Tuple[Tuple[int, ...], str, str]]:
    """Versions installees, valides, de la plus recente a la plus ancienne."""
    found = []
    try:
        names = os.listdir(payload_root)
    except OSError:
        return found
    for name in names:
        directory = os.path.join(payload_root, name)
        version = _read_version(os.path.join(directory, "invis", "version.py"))
        if version:
            found.append((_parse(version), version, directory))
    found.sort(reverse=True)
    return found


def activate(app_name: str = APP_NAME,
             bundle_root: Optional[str] = None) -> Optional[str]:
    """Place le code mis a jour devant le code embarque, s'il est plus recent.

    Renvoie le dossier retenu, ou None si le code embarque fait foi.
    """
    try:
        current = bundled_version(bundle_root)
        if current is None:
            return None
        installed = candidates(os.path.join(data_dir(app_name), "payload"))
        for parsed, _version, directory in installed:
            if parsed > _parse(current):
                sys.path.insert(0, directory)
                return directory
    except Exception:  # noqa: BLE001 - le demarrage prime sur tout le reste
        return None
    return None

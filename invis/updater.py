"""Mise a jour du code applicatif, proposee a l'utilisateur.

Principe
--------
L'executable embarque Python, OpenCV et numpy: environ 150 Mo qui ne bougent
presque jamais. Le code du projet pese 200 Ko et change souvent. On ne met
donc a jour que le second, sous forme d'archive signee.

Ce que ce module accepte d'installer
------------------------------------
Une mise a jour automatique est, par construction, de l'execution de code a
distance: qui controle l'adresse de publication controle les machines qui la
consultent. Et cette application dialogue avec un drone. La verification n'est
donc pas une precaution de confort.

Trois conditions, toutes obligatoires, verifiees AVANT d'ecrire quoi que ce
soit sur le disque:

1. l'adresse est en HTTPS;
2. l'archive telechargee porte une signature Ed25519 valide, verifiee avec la
   cle publique compilee dans le programme;
3. la version proposee est strictement superieure a celle installee.

Une empreinte SHA-256 seule ne suffirait pas: qui remplace l'archive sur le
serveur remplace aussi son empreinte. Seule une signature qu'il ne peut pas
fabriquer -- faute de la cle privee -- l'en empeche. L'empreinte reste
verifiee, mais pour detecter une corruption de transfert, pas une attaque.

L'installation n'est jamais silencieuse: ce module signale une version
disponible, et n'installe que sur demande explicite.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

from .version import APP_NAME, VERSION, is_newer

# Cle publique de publication, en hexadecimal (32 octets Ed25519).
# La cle privee correspondante ne doit jamais entrer dans ce depot.
PUBLIC_KEY_HEX = "4c71f61eb64862b833b9c56ef293d5d0258f73bc1f5473d5a5c8cd8916828b8e"

# Adresse du manifeste. Doit etre en HTTPS.
MANIFEST_URL = "https://github.com/Jrix-G/Invis/releases/latest/download/manifest.json"

# Bornes de securite sur ce qu'on accepte de telecharger.
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
NETWORK_TIMEOUT_S = 15.0
MANIFEST_MAX_BYTES = 64 * 1024


class UpdateError(Exception):
    """Toute condition qui empeche d'installer, y compris un refus voulu."""


@dataclass
class Release:
    version: str
    url: str
    signature_hex: str
    sha256: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Emplacements
# ---------------------------------------------------------------------------

def data_dir() -> str:
    """Repertoire inscriptible propre a l'utilisateur.

    L'executable peut etre installe dans un emplacement en lecture seule
    (Program Files, /opt). Le code mis a jour va donc dans l'espace
    utilisateur, ce qui evite aussi de demander des droits administrateur
    pour une simple mise a jour.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_NAME)


def payload_root() -> str:
    return os.path.join(data_dir(), "payload")


def installed_versions() -> list:
    """Versions de code installees, de la plus recente a la plus ancienne."""
    root = payload_root()
    if not os.path.isdir(root):
        return []
    found = []
    for name in os.listdir(root):
        marker = os.path.join(root, name, "invis", "version.py")
        if os.path.exists(marker):
            found.append(name)
    return sorted(found, key=lambda v: tuple(int(x) if x.isdigit() else 0
                                             for x in v.split(".")), reverse=True)


def active_version() -> str:
    """Version reellement en cours d'execution."""
    return VERSION


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_signature(payload: bytes, signature_hex: str) -> None:
    if not PUBLIC_KEY_HEX:
        raise UpdateError("aucune cle publique compilee: mise a jour refusee")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover
        raise UpdateError(f"bibliotheque de signature absente: {exc}") from exc

    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(PUBLIC_KEY_HEX))
        key.verify(bytes.fromhex(signature_hex), payload)
    except InvalidSignature as exc:
        raise UpdateError("signature invalide: archive refusee") from exc
    except ValueError as exc:
        raise UpdateError(f"signature illisible: {exc}") from exc


def _verify_digest(payload: bytes, expected: str) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    if digest.lower() != expected.strip().lower():
        raise UpdateError("empreinte differente: telechargement corrompu")


def _safe_extract(archive: str, destination: str) -> None:
    """Extrait en refusant tout chemin qui sortirait du dossier cible.

    Une archive peut contenir des chemins absolus ou remontants ("../..") et
    ecrire n'importe ou sur le disque. La signature rend ce cas improbable,
    mais une verification qui ne coute rien ne se discute pas.
    """
    destination = os.path.abspath(destination)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(destination, member))
            if not target.startswith(destination + os.sep) and target != destination:
                raise UpdateError(f"archive suspecte: chemin hors dossier ({member})")
        zf.extractall(destination)


# ---------------------------------------------------------------------------
# Consultation
# ---------------------------------------------------------------------------

def fetch_manifest(url: str = MANIFEST_URL) -> Release:
    if not url.lower().startswith("https://"):
        raise UpdateError("adresse de mise a jour non securisee (HTTPS exige)")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_S) as resp:
        raw = resp.read(MANIFEST_MAX_BYTES + 1)
    if len(raw) > MANIFEST_MAX_BYTES:
        raise UpdateError("manifeste anormalement volumineux")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateError(f"manifeste illisible: {exc}") from exc

    missing = [k for k in ("version", "url", "signature", "sha256") if k not in data]
    if missing:
        raise UpdateError(f"manifeste incomplet, champs manquants: {', '.join(missing)}")
    return Release(version=str(data["version"]), url=str(data["url"]),
                   signature_hex=str(data["signature"]), sha256=str(data["sha256"]),
                   notes=str(data.get("notes", "")))


def check(url: str = MANIFEST_URL, current: Optional[str] = None) -> Optional[Release]:
    """Renvoie la version disponible si elle est plus recente, sinon None."""
    release = fetch_manifest(url)
    if is_newer(release.version, current or active_version()):
        return release
    return None


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install(release: Release, current: Optional[str] = None,
            on_progress: Optional[Callable[[str], None]] = None) -> str:
    """Telecharge, verifie et installe. Renvoie le dossier installe.

    Rien n'est ecrit a l'emplacement definitif avant que la signature ne soit
    validee, et l'installation elle-meme se fait par renommage: soit la
    nouvelle version est entierement en place, soit l'ancienne reste intacte.
    """
    log = on_progress or (lambda msg: None)
    current = current or active_version()

    if not is_newer(release.version, current):
        raise UpdateError(f"version {release.version} pas plus recente que {current}")
    if not release.url.lower().startswith("https://"):
        raise UpdateError("archive proposee hors HTTPS: refusee")

    log(f"telechargement de la version {release.version}")
    req = urllib.request.Request(release.url)
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_S) as resp:
        payload = resp.read(MAX_PAYLOAD_BYTES + 1)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise UpdateError("archive trop volumineuse: refusee")

    log("verification de l'empreinte et de la signature")
    _verify_digest(payload, release.sha256)
    _verify_signature(payload, release.signature_hex)

    root = payload_root()
    os.makedirs(root, exist_ok=True)
    final = os.path.join(root, release.version)
    staging = tempfile.mkdtemp(prefix=f".{release.version}.", dir=root)

    try:
        archive = os.path.join(staging, "payload.zip")
        with open(archive, "wb") as fh:
            fh.write(payload)
        extracted = os.path.join(staging, "content")
        _safe_extract(archive, extracted)
        os.remove(archive)

        if not os.path.exists(os.path.join(extracted, "invis", "version.py")):
            raise UpdateError("archive sans code applicatif: refusee")

        if os.path.exists(final):
            shutil.rmtree(final, ignore_errors=True)
        os.replace(extracted, final)
        log(f"version {release.version} installee")
        return final
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prune(keep: int = 2) -> None:
    """Ne conserve que les dernieres versions, pour pouvoir revenir en arriere."""
    for old in installed_versions()[keep:]:
        shutil.rmtree(os.path.join(payload_root(), old), ignore_errors=True)


# ---------------------------------------------------------------------------
# Verification en arriere-plan
# ---------------------------------------------------------------------------

class UpdateWatcher:
    """Consulte le manifeste sans bloquer l'interface.

    L'application doit rester utilisable meme si l'adresse de publication est
    injoignable -- cas normal en vol, ou le PC est sur le reseau du drone et
    n'a aucun acces exterieur. Un echec de consultation n'est donc pas une
    erreur affichee, seulement une absence de proposition.
    """

    def __init__(self, url: str = MANIFEST_URL,
                 on_found: Optional[Callable[[Release], None]] = None) -> None:
        self.url = url
        self._on_found = on_found or (lambda rel: None)
        self.available: Optional[Release] = None
        self.last_error: str = ""
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="update-check", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            release = check(self.url)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return
        if release is not None:
            self.available = release
            self._on_found(release)

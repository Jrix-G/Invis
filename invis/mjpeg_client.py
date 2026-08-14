"""Client MJPEG / snapshot pour le pont ESP32-CAM.

Le firmware ne sert qu'un flux video a la fois et n'a aucune marge pour
accumuler du retard. Le client suit donc une regle simple: il ne conserve
jamais qu'une seule image, la plus recente. Une image qui n'a pas ete
consommee a temps est perdue volontairement, pas mise en file.

Deuxieme regle, moins evidente et plus importante: la carte ne sert pas que
la video, elle porte aussi le lien de pilotage (/pilot, en WebSocket). Les
deux partagent le meme serveur HTTP et le meme pool de sockets, purge en LRU.
Toute socket ouverte ici est donc prise sur la marge du lien pilote. Ce module
ouvre au plus config.MAX_ESP_SOCKETS sockets vers la carte, les reutilise au
lieu d'en rouvrir, et refuse d'insister quand il n'y a rien a lire. Invis
reste strictement en lecture: aucun appel vers /pilot, jamais de commande.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Tuple

from . import config

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def _split_timeout(timeout) -> Tuple[float, float]:
    """Accepte un delai unique ou un couple (connexion, lecture)."""
    if isinstance(timeout, (tuple, list)):
        return float(timeout[0]), float(timeout[1])
    return float(timeout), float(timeout)


class Reply:
    """Reponse HTTP, avec le peu d'API dont ce module a besoin.

    Volontairement minuscule et bati sur `http.client`. Ce module ne peut pas
    dependre de `requests`: le code applicatif est livre en mise a jour signee
    de 200 Ko contenant *uniquement* des .py, alors que l'executable qui
    l'heberge embarque ses dependances et ne change que rarement. Importer une
    bibliotheque tierce ici rendrait la mise a jour impossible a installer sur
    les executables deja distribues -- ils planteraient au demarrage, a chaque
    lancement. Pour un correctif de securite de vol, c'est le pire resultat
    possible: la station sol devient inutilisable au lieu d'etre corrigee.
    """

    def __init__(self, raw: http.client.HTTPResponse) -> None:
        self.raw = raw
        self.status = raw.status
        self.headers = raw.headers
        self._content: Optional[bytes] = None
        self.drained = False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise ConnectionError(f"HTTP {self.status} {self.raw.reason}")

    @property
    def content(self) -> bytes:
        """Corps complet. Le lire jusqu'au bout est ce qui rend la socket
        reutilisable: une reponse a moitie lue oblige a fermer la connexion."""
        if self._content is None:
            self._content = self.raw.read()
            self.drained = True
        return self._content


class _BoardSession:
    """Session HTTP partagee vers une carte: sockets reutilisees et comptees.

    Deux garde-fous, qui ne font pas la meme chose:

    - le vivier de connexions inactives evite de rouvrir une socket a chaque
      appel. C'est ce que l'en-tete `Connection: close` interdisait: chaque
      /status, chaque /jpg, chaque reprise de flux ouvrait une socket neuve;
    - le semaphore borne le nombre d'appels *simultanes*. Le vivier seul ne
      suffirait pas: rien n'empeche deux fils de sortir chacun une connexion
      neuve quand le vivier est vide.

    Une instance par (hote, port), partagee par le flux video et les appels
    utilitaires -- sans quoi chaque appelant aurait son propre vivier et le
    plafond ne voudrait plus rien dire.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._slots = threading.BoundedSemaphore(config.MAX_ESP_SOCKETS)
        self._idle: list = []
        self._lock = threading.Lock()

    def url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    # -- vivier de connexions ---------------------------------------------

    def _take(self):
        """Une connexion inactive s'il y en a, sinon rien."""
        with self._lock:
            return self._idle.pop() if self._idle else None

    def _give_back(self, conn) -> None:
        with self._lock:
            if len(self._idle) < config.MAX_ESP_SOCKETS:
                self._idle.append(conn)
                return
        conn.close()

    def _connect(self, connect_t: float, read_t: float):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=connect_t)
        conn.connect()
        # Le delai de connexion et celui de lecture ne mesurent pas la meme
        # chose: une carte qui repond puis se tait doit etre lachee sur le
        # second, plus long, pas sur le premier.
        conn.sock.settimeout(read_t)
        return conn

    def _send(self, conn, path: str) -> http.client.HTTPResponse:
        conn.request("GET", path, headers={
            "Host": f"{self.host}:{self.port}",
            # Keep-alive est le defaut en HTTP/1.1; on l'ecrit pour que la
            # difference avec l'ancien "Connection: close" reste lisible.
            "Connection": "keep-alive",
            "Accept-Encoding": "identity",
        })
        return conn.getresponse()

    @contextlib.contextmanager
    def open(self, path: str, timeout, stream: bool = False) -> Iterator[Reply]:
        """Ouvre un appel en tenant un des jetons de socket pendant sa duree."""
        connect_t, read_t = _split_timeout(timeout)
        if not self._slots.acquire(timeout=config.SOCKET_ACQUIRE_TIMEOUT_S):
            raise TimeoutError(f"aucune socket libre vers {self.host} "
                               f"({config.MAX_ESP_SOCKETS} deja en cours)")
        conn = None
        keep = False
        try:
            conn = self._take()
            if conn is not None:
                # Une connexion tiree du vivier a pu etre fermee par la carte
                # entre-temps, sans qu'on l'apprenne avant d'ecrire dedans. Un
                # echec ici n'est donc pas une panne: on rouvre et on rejoue,
                # une seule fois. Toutes les requetes de ce module sont des
                # GET, donc rejouables sans effet de bord.
                try:
                    if conn.sock is None:      # deja fermee de notre cote
                        raise OSError("connexion inactive fermee")
                    conn.sock.settimeout(read_t)
                    raw = self._send(conn, path)
                except (OSError, http.client.HTTPException):
                    conn.close()
                    conn = None
            if conn is None:
                conn = self._connect(connect_t, read_t)
                raw = self._send(conn, path)

            reply = Reply(raw)
            yield reply
            # Une socket ne retourne au vivier que si la carte n'a pas annonce
            # sa fermeture et que le corps a ete lu en entier. Un flux
            # multipart ne remplit jamais ces conditions: il est sans fin, donc
            # sa connexion lui reste dediee puis se ferme avec lui.
            keep = reply.drained and not raw.will_close and not stream
        finally:
            if conn is not None:
                if keep:
                    self._give_back(conn)
                else:
                    conn.close()
            self._slots.release()

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
        for conn in idle:
            conn.close()


_sessions: dict = {}
_sessions_lock = threading.Lock()


def board_session(host: str, port: int = config.DEFAULT_PORT) -> _BoardSession:
    """Session partagee pour cette carte, creee au premier appel."""
    key = (host, port)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _BoardSession(host, port)
            _sessions[key] = session
        return session


def _parse_boundary(content_type: str) -> Optional[bytes]:
    """Extrait la borne annoncee par l'en-tete multipart."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            value = part[len("boundary="):].strip().strip('"')
            if value:
                return b"--" + value.encode("ascii", "replace")
    return None


def _content_length(headers: bytes) -> Optional[int]:
    """Taille annoncee d'une partie, ou None si l'en-tete ne la donne pas."""
    for line in headers.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                return int(line.split(b":", 1)[1].strip())
            except ValueError:
                return None
    return None


@dataclass
class Frame:
    """Une image JPEG brute et l'instant ou elle a fini d'arriver."""

    jpeg: bytes
    recv_time: float
    index: int


@dataclass
class LinkStats:
    """Compteurs du lien video, lus par l'interface."""

    frames: int = 0
    dropped: int = 0
    bytes_total: int = 0
    fps: float = 0.0
    kbps: float = 0.0
    last_error: str = ""
    connected: bool = False
    reconnects: int = 0
    corrupt: int = 0
    _window: list = field(default_factory=list, repr=False)


class VideoLink:
    """Thread reseau: telecharge le flux et expose la derniere image.

    Deux modes:
      - "stream": multipart/x-mixed-replace sur /stream, le mode nominal;
      - "snapshot": interrogation repetee de /jpg, repli quand le flux
        multipart se coupe sans arret (client fantome, Wi-Fi qui decroche).
    """

    def __init__(
        self,
        host: str = config.DEFAULT_HOST,
        port: int = config.DEFAULT_PORT,
        mode: str = "stream",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.mode = mode
        self._on_log = on_log or (lambda msg: None)
        self._http = board_session(host, port)
        # Etat "camera de la carte" tel que rapporte par /status: True tant
        # qu'on n'a pas la preuve du contraire, avec l'instant de la derniere
        # interrogation pour ne pas y revenir plus vite que STATUS_POLL_S.
        self._camera_on = True
        self._camera_checked = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._counter = 0
        self.stats = LinkStats()

    # -- cycle de vie ------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.stats = LinkStats()
        self._thread = threading.Thread(target=self._run, name="video-link", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None
        self.stats.connected = False
        with self._lock:
            self._latest = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- acces aux images --------------------------------------------------

    def take_latest(self) -> Optional[Frame]:
        """Retire et renvoie la derniere image, ou None si rien de neuf."""
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def _publish(self, jpeg: bytes) -> None:
        now = time.time()
        self._counter += 1
        with self._lock:
            if self._latest is not None:
                self.stats.dropped += 1
            self._latest = Frame(jpeg=jpeg, recv_time=now, index=self._counter)
        self.stats.frames += 1
        self.stats.bytes_total += len(jpeg)
        self._tick_rate(now, len(jpeg))

    def _tick_rate(self, now: float, nbytes: int) -> None:
        window = self.stats._window
        window.append((now, nbytes))
        cutoff = now - 2.0
        while window and window[0][0] < cutoff:
            window.pop(0)
        if len(window) >= 2:
            span = window[-1][0] - window[0][0]
            if span > 0:
                self.stats.fps = (len(window) - 1) / span
                self.stats.kbps = sum(b for _, b in window[1:]) / span / 1024.0

    # -- URLs --------------------------------------------------------------

    def _url(self, path: str) -> str:
        return self._http.url(path)

    # -- etat de la camera de la carte -------------------------------------

    def _camera_ready(self) -> bool:
        """La carte a-t-elle une camera active, d'apres /status.

        Sans cette question, Invis passait son temps a redemander un flux qui
        ne pouvait pas exister: /stream echouait aussitot, la boucle rouvrait
        une socket, et le seul effet observable etait de vider le pool de la
        carte -- donc de purger le lien pilote. Une camera eteinte n'est pas
        une panne a rattraper, c'est un etat a attendre.

        En cas de doute (carte injoignable, /status muet ou sans le champ) on
        repond oui: c'est a la tentative de flux, et a son backoff, de trancher.
        """
        now = time.time()
        if now - self._camera_checked < config.STATUS_POLL_S:
            return self._camera_on
        self._camera_checked = now
        try:
            with self._http.open(config.STATUS_PATH,
                                 timeout=(config.CONNECT_TIMEOUT_S, config.READ_TIMEOUT_S)) as resp:
                payload = json.loads(resp.content.decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001 - /status est indicatif, pas bloquant
            return self._camera_on
        enabled = payload.get("camera_enabled") if isinstance(payload, dict) else None
        if not isinstance(enabled, bool):
            return self._camera_on
        if enabled != self._camera_on:
            self._on_log("camera de la carte activee" if enabled
                         else "camera de la carte desactivee, flux suspendu")
        self._camera_on = enabled
        return enabled

    # -- boucle ------------------------------------------------------------

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            if not self._camera_ready():
                self.stats.connected = False
                self.stats.last_error = "camera desactivee sur la carte"
                attempt = 0
                self._stop.wait(config.CAMERA_OFF_POLL_S)
                continue

            started = time.time()
            delivered = self.stats.frames
            try:
                if self.mode == "snapshot":
                    self._pump_snapshots()
                else:
                    self._pump_stream()
                attempt = 0
            except Exception as exc:  # noqa: BLE001 - on veut vraiment tout rattraper
                if self._stop.is_set():
                    break
                self.stats.connected = False
                self.stats.last_error = f"{type(exc).__name__}: {exc}"

                # Un flux qui a reellement fonctionne remet le compteur a zero.
                #
                # Sans cela le decompte ne redescendait jamais: un flux qui
                # marche ne rend pas la main, il ne fait que lever une
                # exception en mourant. Trois coupures suffisaient donc a
                # coller l'attente au maximum, definitivement -- et la carte
                # coupant le flux toutes les quelques secondes, l'attente
                # devenait plus longue que le flux lui-meme. C'etait la cause
                # des saccades observees, cote client et non cote carte.
                lived = time.time() - started
                if lived >= config.RECONNECT_STABLE_S and self.stats.frames > delivered:
                    attempt = 0

                delay = config.RECONNECT_BACKOFF_S[min(attempt, len(config.RECONNECT_BACKOFF_S) - 1)]
                self.stats.reconnects += 1
                self._on_log(f"lien perdu ({self.stats.last_error}), "
                             f"reprise dans {delay:.2f}s (flux tenu {lived:.1f}s)")
                attempt += 1
                self._stop.wait(delay)
        self.stats.connected = False

    def _pump_stream(self) -> None:
        url = self._url(config.STREAM_PATH)
        self._on_log(f"connexion {url}")
        with self._http.open(config.STREAM_PATH,
                             timeout=(config.CONNECT_TIMEOUT_S, config.READ_TIMEOUT_S),
                             stream=True) as resp:
            resp.raise_for_status()
            self.stats.connected = True
            self.stats.last_error = ""
            boundary = _parse_boundary(resp.headers.get("Content-Type", ""))
            self._on_log(f"flux ouvert (bornes {boundary.decode(errors='replace')})"
                         if boundary else "flux ouvert (sans bornes annoncees)")
            buffer = bytearray()
            while not self._stop.is_set():
                # Lecture brute: le decodage de requests decouperait le flux en
                # "contenu", notion qui n'a pas de sens pour un multipart sans
                # fin -- c'est nous qui savons ou sont les bornes.
                chunk = resp.raw.read(4096)
                if not chunk:
                    raise ConnectionError("flux ferme par la carte")
                buffer.extend(chunk)
                if boundary:
                    self._drain_multipart(buffer, boundary)
                else:
                    self._drain_jpegs(buffer)

    def _drain_multipart(self, buffer: bytearray, boundary: bytes) -> None:
        """Decoupe le flux en s'appuyant sur les en-tetes de chaque partie.

        Pourquoi ne pas se contenter de chercher les marqueurs JPEG: si une
        image arrive tronquee -- ce qui se produit reellement sur cette carte,
        le decodeur signalant "premature end of data segment" -- la recherche
        par marqueurs attrape le debut de l'image coupee puis la *fin de la
        suivante*, et fabrique une image chimere. Celle-ci se decode en un
        aplat gris, que l'analyse prend pour une image floue et non pour une
        image perdue. Deux images sont ainsi gachees au lieu d'une, et le
        diagnostic est fausse.

        L'en-tete Content-Length envoye par le firmware donne la taille exacte:
        on sait alors si l'image est complete, et on jette proprement celles
        qui ne le sont pas.
        """
        while True:
            start = buffer.find(boundary)
            if start < 0:
                if len(buffer) > config.MAX_JPEG_BYTES:
                    del buffer[:-len(boundary)]
                return

            head_end = buffer.find(b"\r\n\r\n", start)
            if head_end < 0:
                if start > 0:
                    del buffer[:start]
                return
            head_end += 4

            headers = bytes(buffer[start:head_end])
            length = _content_length(headers)
            if length is None:
                # Partie sans taille annoncee: repli sur les marqueurs, en se
                # limitant a ce qui suit l'en-tete.
                del buffer[:head_end]
                self._drain_jpegs(buffer)
                return
            if length <= 0 or length > config.MAX_JPEG_BYTES:
                self.stats.corrupt += 1
                del buffer[:head_end]
                continue
            if len(buffer) < head_end + length:
                if start > 0:
                    del buffer[:start]
                return

            payload = bytes(buffer[head_end:head_end + length])
            del buffer[:head_end + length]

            if payload.startswith(SOI) and payload.endswith(EOI):
                self._publish(payload)
            else:
                # Taille annoncee tenue mais contenu non conforme: image
                # abimee en vol, on la jette au lieu de la faire analyser.
                self.stats.corrupt += 1

    def _drain_jpegs(self, buffer: bytearray) -> None:
        """Extrait toutes les images completes presentes dans le tampon.

        On ignore volontairement les en-tetes de partie multipart: chercher
        SOI/EOI marche quel que soit le format exact des bornes, et resiste a
        une bordure tronquee.
        """
        while True:
            start = buffer.find(SOI)
            if start < 0:
                if len(buffer) > config.MAX_JPEG_BYTES:
                    del buffer[:-2]
                return
            end = buffer.find(EOI, start + 2)
            if end < 0:
                if start > 0:
                    del buffer[:start]
                if len(buffer) > config.MAX_JPEG_BYTES:
                    # Tampon incoherent: on jette et on repart au prochain SOI.
                    buffer.clear()
                return
            end += 2
            self._publish(bytes(buffer[start:end]))
            del buffer[:end]

    def _pump_snapshots(self) -> None:
        url = self._url(config.SNAPSHOT_PATH)
        self._on_log(f"mode snapshot sur {url}")
        self.stats.connected = True
        while not self._stop.is_set():
            # Meme socket d'une image a l'autre: c'est tout l'interet de la
            # session partagee. Avant, chaque /jpg en ouvrait une neuve.
            with self._http.open(config.SNAPSHOT_PATH,
                                 timeout=(config.CONNECT_TIMEOUT_S,
                                          config.READ_TIMEOUT_S)) as resp:
                resp.raise_for_status()
                data = resp.content
            if not data.startswith(SOI):
                raise ValueError("reponse /jpg non JPEG")
            self._publish(data)


def http_get(host: str, path: str, port: int = config.DEFAULT_PORT,
             timeout: float = 3.0) -> str:
    """Appel utilitaire pour /status et /control. Renvoie le corps en texte.

    Passe par la session partagee de la carte: la socket est reutilisee, et
    l'appel compte dans le plafond de config.MAX_ESP_SOCKETS comme les autres.
    """
    with board_session(host, port).open(path, timeout=timeout) as resp:
        return resp.content.decode("utf-8", errors="replace")

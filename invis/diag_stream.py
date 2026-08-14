"""Diagnostic des coupures du flux video.

Le flux se coupe toutes les quelques secondes. Deux familles de causes sont
possibles et ce programme les separe, au lieu de les supposer:

  - la carte coupe d'elle-meme (envoi trop lent, expiration, purge de socket);
  - c'est le client qui la fait couper -- lecture trop lente parce que Python
    est occupe ailleurs, en-tete de connexion mal choisi, reconnexions en
    rafale qui saturent les sept sockets du serveur.

On compare donc plusieurs facons de lire le meme flux, de la plus depouillee
(socket brute, aucun traitement) a la plus chargee. Si la coupure survient
identiquement dans le cas le plus depouille, elle vient de la carte. Si elle
n'apparait que sous charge, elle vient du client.

Usage:
    python -m invis.diag_stream --host 192.168.4.50
    python -m invis.diag_stream --host 192.168.4.50 --seconds 40
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from invis import config
else:
    from . import config

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


@dataclass
class Run:
    label: str
    sessions: List[float] = field(default_factory=list)   # duree de chaque connexion
    frames: int = 0
    bytes_total: int = 0
    errors: List[str] = field(default_factory=list)
    stalls: List[float] = field(default_factory=list)     # trous entre deux lectures

    def report(self, duration: float) -> str:
        n = len(self.sessions)
        med = statistics.median(self.sessions) if self.sessions else 0.0
        worst = max(self.stalls) if self.stalls else 0.0
        p95 = (statistics.quantiles(self.stalls, n=20)[-1]
               if len(self.stalls) >= 20 else worst)
        return (f"{self.label:<28} coupures={n:<3} duree_med={med:6.2f}s  "
                f"images={self.frames:<5} {self.bytes_total / max(duration, 1e-6) / 1024:6.1f} kB/s  "
                f"pause_max={worst * 1000:6.0f}ms p95={p95 * 1000:5.0f}ms")


def raw_socket_read(host: str, port: int, seconds: float, load_ms: float = 0.0,
                    connection_close: bool = True) -> Run:
    """Lit /stream sur une socket brute, sans urllib ni decodage.

    `load_ms` simule le travail d'analyse pour mesurer l'effet de la charge
    Python sur la tenue du flux.
    """
    label = f"socket brute"
    if connection_close:
        label += " (close)"
    else:
        label += " (keep-alive)"
    if load_ms:
        label += f" +{load_ms:.0f}ms charge"

    run = Run(label=label)
    end = time.time() + seconds

    while time.time() < end:
        started = time.time()
        try:
            s = socket.create_connection((host, port), timeout=6.0)
            s.settimeout(6.0)
            headers = f"GET {config.STREAM_PATH} HTTP/1.1\r\nHost: {host}\r\n"
            headers += "Connection: close\r\n" if connection_close else "Connection: keep-alive\r\n"
            headers += "\r\n"
            s.sendall(headers.encode())

            buffer = bytearray()
            last_read = time.time()
            while time.time() < end:
                chunk = s.recv(8192)
                now = time.time()
                run.stalls.append(now - last_read)
                last_read = now
                if not chunk:
                    raise ConnectionError("flux ferme par la carte")
                run.bytes_total += len(chunk)
                buffer.extend(chunk)

                while True:
                    a = buffer.find(SOI)
                    if a < 0:
                        break
                    b = buffer.find(EOI, a + 2)
                    if b < 0:
                        break
                    run.frames += 1
                    del buffer[:b + 2]

                if load_ms:
                    time.sleep(load_ms / 1000.0)
            s.close()
        except Exception as exc:  # noqa: BLE001
            run.errors.append(f"{type(exc).__name__}: {exc}")
        run.sessions.append(time.time() - started)
    return run


def urllib_read(host: str, port: int, seconds: float) -> Run:
    """Le chemin utilise par le programme, pour comparaison.

    Passe par la session partagee, comme gcs_vision: la socket est reutilisee
    d'un essai a l'autre. Un diagnostic qui ouvrirait une socket par tentative
    mesurerait un autre programme que celui qu'on veut diagnostiquer -- et
    prendrait au passage la place du lien pilote dans le pool de la carte.
    """
    from .mjpeg_client import board_session

    http = board_session(host, port)
    run = Run(label="session partagee (comme gcs_vision)")
    end = time.time() + seconds
    while time.time() < end:
        started = time.time()
        try:
            with http.open(config.STREAM_PATH, timeout=(4.0, 6.0), stream=True) as resp:
                resp.raise_for_status()
                buffer = bytearray()
                last_read = time.time()
                while time.time() < end:
                    chunk = resp.raw.read(4096)
                    now = time.time()
                    run.stalls.append(now - last_read)
                    last_read = now
                    if not chunk:
                        raise ConnectionError("flux ferme")
                    run.bytes_total += len(chunk)
                    buffer.extend(chunk)
                    while True:
                        a = buffer.find(SOI)
                        if a < 0:
                            break
                        b = buffer.find(EOI, a + 2)
                        if b < 0:
                            break
                        run.frames += 1
                        del buffer[:b + 2]
        except Exception as exc:  # noqa: BLE001
            run.errors.append(f"{type(exc).__name__}: {exc}")
        run.sessions.append(time.time() - started)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic des coupures du flux")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--seconds", type=float, default=30.0, help="duree par essai")
    args = parser.parse_args()

    print(f"cible http://{args.host}:{args.port}{config.STREAM_PATH}, "
          f"{args.seconds:.0f}s par essai\n")

    runs = [
        raw_socket_read(args.host, args.port, args.seconds, 0.0, True),
        raw_socket_read(args.host, args.port, args.seconds, 0.0, False),
        raw_socket_read(args.host, args.port, args.seconds, 15.0, True),
        urllib_read(args.host, args.port, args.seconds),
    ]

    print()
    for r in runs:
        print(r.report(args.seconds))
        for e in dict.fromkeys(r.errors):
            print(f"{'':<28} -> {e[:100]}")

    print()
    print("Lecture:")
    print("  coupures identiques partout      -> la carte coupe d'elle-meme")
    print("  seulement avec la charge         -> le client ne lit pas assez vite")
    print("  seulement avec 'close'           -> l'en-tete de connexion est en cause")
    print("  duree_med proche de la duree     -> pas de coupure, le flux tient")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

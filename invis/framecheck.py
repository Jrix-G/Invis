"""Controle de qualite des images avant analyse.

Le lien video de cette carte livre par moments des images abimees: blocs de
couleur, pave bleu, image franchement floue. Les laisser entrer dans le suivi
de points coute plus cher que de les jeter.

Ce que ce module ne fait pas: rattraper la qualite
--------------------------------------------------
Mesure faite contre la verite terrain du simulateur, en injectant des paves
bleus et du flou a taux croissant (erreur quadratique sur la distance):

    taux de corruption      0 %     5 %    12 %    25 %
    sans filtre           0,21 m  0,39 m  0,56 m  0,59 m
    avec filtre           0,21 m  0,48 m  0,53 m  0,49 m

L'ecart entre les deux lignes tient dans le bruit de mesure. Ce qui compte est
la premiere ligne: la corruption triple presque l'erreur, et le filtrage ne la
recupere pas. La raison est simple -- ecarter une image coute en continuite de
suivi a peu pres ce qu'elle coutait en donnees fausses, d'autant que cette
camera ne fournit que sept images par seconde. Une image manquante n'est pas
gratuite.

La conclusion est donc a l'oppose de l'intuition: la qualite se corrige a la
source (horloge capteur, chemin DMA), pas au sol.

Ce que ce module fait
---------------------
1. Ecarter ce qui est franchement inexploitable: JPEG tronque, image beaucoup
   plus floue que d'habitude.
2. Surtout, *mesurer et signaler* la degradation, pour qu'un flux qui s'abime
   se voie au lieu de degrader les distances en silence.

Les criteres sont relatifs a ce que le flux produit habituellement, jamais
absolus: la nettete normale depend de la scene et de la lumiere, et une scene
naturellement coloree depasserait en permanence un seuil de couleur fixe. Ce
qui trahit une corruption, c'est qu'elle surgit.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config

VERDICT_OK = "ok"
VERDICT_DEGRADED = "degradee"
VERDICT_REJECT = "rejetee"


@dataclass
class FrameQuality:
    verdict: str = VERDICT_OK
    reason: str = ""
    sharpness: float = 0.0
    sharpness_ratio: float = 1.0
    chroma_defect: float = 0.0
    chroma_excess: float = 0.0
    size_ratio: float = 1.0

    @property
    def usable(self) -> bool:
        return self.verdict != VERDICT_REJECT


class FrameGate:
    """Juge chaque image par rapport aux precedentes du meme flux."""

    def __init__(self, dump_dir: Optional[str] = None) -> None:
        self.dump_dir = dump_dir
        self._dumped = 0
        self._streak = 0
        self._sharpness: deque = deque(maxlen=config.QUALITY_WINDOW)
        self._sizes: deque = deque(maxlen=config.QUALITY_WINDOW)
        self._chroma: deque = deque(maxlen=config.QUALITY_WINDOW)
        self.rejected = 0
        self.degraded = 0
        self.total = 0

    def reset(self) -> None:
        self._sharpness.clear()
        self._sizes.clear()
        self._chroma.clear()
        self._streak = 0
        self.rejected = 0
        self.degraded = 0
        self.total = 0

    def check(self, bgr: np.ndarray, jpeg_size: int = 0) -> FrameQuality:
        self.total += 1
        height = max(1, int(bgr.shape[0] * config.QUALITY_WIDTH / bgr.shape[1]))
        small = cv2.resize(bgr, (config.QUALITY_WIDTH, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        quality = FrameQuality()
        quality.sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        quality.chroma_defect = _chroma_defect(small)

        ref = float(np.median(self._sharpness)) if len(self._sharpness) >= 5 else 0.0
        quality.sharpness_ratio = quality.sharpness / ref if ref > 1e-6 else 1.0

        # La couleur se juge elle aussi par rapport a l'habitude du flux. Un
        # seuil absolu ne marche pas: une scene naturellement coloree le
        # depasse en permanence, tandis qu'une corruption breve sur une scene
        # terne reste en dessous. Ce qui distingue la corruption, c'est qu'elle
        # *surgit* -- donc on compare a la mediane recente.
        # Tant qu'aucune reference n'existe, le critere couleur n'a pas d'avis.
        #
        # Le calculer quand meme revenait a comparer l'ecart *absolu* -- jusqu'a
        # 1,0 sur une scene coloree -- a un seuil de 0,02 pense pour un *exces*
        # relatif. La premiere image etait donc rejetee, or seules les images
        # acceptees alimentent la reference: celle-ci ne se formait jamais et le
        # filtre rejetait indefiniment une scene parfaitement saine.
        ref_chroma = float(np.median(self._chroma)) if len(self._chroma) >= 5 else None
        quality.chroma_excess = (quality.chroma_defect - ref_chroma
                                 if ref_chroma is not None else 0.0)

        if jpeg_size > 0:
            ref_size = float(np.median(self._sizes)) if len(self._sizes) >= 5 else 0.0
            quality.size_ratio = jpeg_size / ref_size if ref_size > 1e-6 else 1.0
            self._sizes.append(jpeg_size)

        if quality.sharpness_ratio < config.QUALITY_SHARPNESS_REJECT:
            quality.verdict = VERDICT_REJECT
            quality.reason = f"floue ({quality.sharpness_ratio:.0%} de la nettete habituelle)"
        elif quality.size_ratio < config.QUALITY_SIZE_REJECT:
            quality.verdict = VERDICT_REJECT
            quality.reason = f"image tronquee ({quality.size_ratio:.0%} de la taille habituelle)"
        elif quality.chroma_excess > config.QUALITY_CHROMA_REJECT:
            quality.verdict = VERDICT_REJECT
            quality.reason = (f"corruption couleur ({quality.chroma_defect:.1%} de l'image, "
                              f"{quality.chroma_excess:+.1%} au-dessus de l'habitude)")
        elif (quality.sharpness_ratio < config.QUALITY_SHARPNESS_WARN
                or quality.chroma_excess > config.QUALITY_CHROMA_WARN):
            quality.verdict = VERDICT_DEGRADED
            quality.reason = "image degradee mais exploitable"

        # Garde-fou contre le blocage sur une scene reellement pauvre.
        #
        # La nettete d'une image floue et celle d'une scene sans texture se
        # ressemblent: un mur uni donne le meme signal qu'une image ratee. Si
        # le filtre rejette sans discontinuer, l'hypothese "image ratee" ne
        # tient plus -- une camera ne rate pas dix images d'affilee alors que
        # la precedente etait nette. On repart alors d'une reference neuve,
        # calee sur ce que la camera montre maintenant.
        if quality.verdict == VERDICT_REJECT:
            self._streak += 1
            if self._streak >= config.QUALITY_MAX_STREAK:
                quality.verdict = VERDICT_DEGRADED
                quality.reason = "scene inhabituelle, reference reetalonnee"
                # Reetalonner veut dire *se caler sur ce que la camera montre
                # maintenant*, pas oublier. Vider les references les laissait
                # vides -- une image DEGRADED ne les alimente pas non plus --
                # et le cycle de rejets repartait a l'identique. On les amorce
                # donc explicitement sur l'image courante.
                self._sharpness.clear()
                self._chroma.clear()
                for _ in range(5):
                    self._sharpness.append(quality.sharpness)
                    self._chroma.append(quality.chroma_defect)
                self._streak = 0
        else:
            self._streak = 0

        # Seules les images *pleinement saines* alimentent la reference.
        #
        # Deux pieges symetriques justifient cette severite. Une serie d'images
        # floures abaisserait le niveau attendu, et le filtre finirait par les
        # accepter toutes. Inversement -- et c'est le cas qui a ete observe --
        # une image a paves corrompus contient des bords francs, donc *plus* de
        # hautes frequences qu'une image normale: elle gonfle la reference de
        # nettete, et les images saines suivantes se retrouvent jugees floues
        # par comparaison. Le filtre se mettait alors a jeter les bonnes.
        if quality.verdict == VERDICT_OK:
            self._sharpness.append(quality.sharpness)
            self._chroma.append(quality.chroma_defect)
        elif quality.verdict == VERDICT_REJECT:
            self.rejected += 1
            self._dump(bgr, quality)
        if quality.verdict == VERDICT_DEGRADED:
            self.degraded += 1

        return quality


    def _dump(self, bgr: np.ndarray, quality: "FrameQuality") -> None:
        """Conserve les images ecartees, pour pouvoir les regarder ensuite.

        Un chiffre de nettete ne dit pas si l'image est floue, tronquee ou
        simplement vide de details. L'oeil, lui, tranche en une seconde.
        """
        if not self.dump_dir or self._dumped >= config.QUALITY_DUMP_MAX:
            return
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            tag = "flou" if quality.sharpness_ratio < config.QUALITY_SHARPNESS_REJECT else "couleur"
            name = (f"{self.total:05d}_{tag}_net{quality.sharpness_ratio:.2f}"
                    f"_chroma{quality.chroma_defect:.3f}.jpg")
            cv2.imwrite(os.path.join(self.dump_dir, name), bgr)
            self._dumped += 1
        except Exception:  # noqa: BLE001 - l'enregistrement ne doit jamais gener l'analyse
            pass


def _chroma_defect(bgr: np.ndarray) -> float:
    """Part de l'image dont la couleur est aberrante.

    Une corruption du flux DVP se traduit par des pixels ou une composante
    part seule, typiquement le bleu. On mesure l'ecart de chaque canal a la
    luminance locale: une scene naturelle reste modere, un bloc corrompu non.
    """
    img = bgr.astype(np.int16)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    luma = (b + g + r) / 3.0
    spread = np.maximum.reduce([np.abs(b - luma), np.abs(g - luma), np.abs(r - luma)])
    return float((spread > config.QUALITY_CHROMA_PIXEL).mean())

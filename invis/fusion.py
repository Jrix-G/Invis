"""Filtrage d'etat: modele a vitesse constante, mesures fenetrees.

Pourquoi un filtre ici
----------------------
L'odometrie visuelle rend un deplacement par image, mesure sur une base tres
courte. Chaque mesure est donc juste en moyenne mais bruitee, et l'integrer
telle quelle transmet tout ce bruit a la trajectoire, au nuage de points et a
toutes les distances qui en dependent.

Un modele a vitesse constante dit ce que la mesure ne dit pas: entre deux
images, le drone a continue son mouvement. Cette connaissance-la vaut une
mesure supplementaire, et c'est exactement ce qu'un filtre de Kalman exploite:
il pondere mesure et prediction par leurs incertitudes respectives au lieu de
choisir arbitrairement l'une des deux.

Le gain reel n'est pas la douceur de la courbe -- c'est le rejet des mesures
aberrantes. Une homographie qui accroche un mur produit un deplacement
fantaisiste; avant ce filtre, seul un plafond de vitesse en dur l'arretait, et
uniquement s'il etait franchement absurde. Ici l'ecart est compare a
l'incertitude *courante* de l'etat: une mesure incompatible est ecartee meme
quand elle reste dans les limites physiques.

Cout
----
L'etat fait 2n composantes avec n <= 2. Toutes les matrices sont 4x4 au plus,
l'inversion porte sur du 2x2. Le cout par image est celui de quelques dizaines
d'operations flottantes: negligeable devant le suivi de points.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class ConstantVelocity:
    """Etat (position, vitesse) en dimension n, mesure sur la vitesse.

    La mesure porte sur la vitesse et non sur la position parce que c'est ce
    que l'odometrie fournit reellement: un deplacement relatif entre deux
    images. La position, elle, n'est observee par rien -- elle resulte de
    l'integration, et son incertitude croit donc sans borne. C'est la
    description honnete du systeme: sans point de reference exterieur, une
    odometrie derive.
    """

    def __init__(self, dim: int = 2, accel_sigma: float = 1.5,
                 meas_sigma: float = 0.35, gate_sigma: float = 3.5) -> None:
        self.dim = int(dim)
        self.accel_sigma = float(accel_sigma)
        self.meas_sigma = float(meas_sigma)
        self.gate_sigma = float(gate_sigma)
        self._x = np.zeros(2 * self.dim)
        self._P = np.eye(2 * self.dim)
        self.reset()

    # -- etat --------------------------------------------------------------

    def reset(self, position: Optional[np.ndarray] = None) -> None:
        n = self.dim
        self._x = np.zeros(2 * n)
        if position is not None:
            self._x[:n] = np.asarray(position, dtype=np.float64).reshape(n)
        self._P = np.eye(2 * n)
        # La position de depart est connue par convention (origine), la vitesse
        # ne l'est pas: lui donner une incertitude franche evite que la
        # premiere mesure soit ignoree au profit d'un zero arbitraire.
        self._P[:n, :n] *= 1e-4
        self._P[n:, n:] *= 1.0

    @property
    def position(self) -> np.ndarray:
        return self._x[:self.dim].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._x[self.dim:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self._x[self.dim:]))

    @property
    def position_sigma(self) -> float:
        """Incertitude de position, en ecart-type moyen sur les axes."""
        n = self.dim
        return float(np.sqrt(max(0.0, np.trace(self._P[:n, :n]) / n)))

    def set_position(self, position: np.ndarray) -> None:
        """Impose une position (fermeture de boucle) sans toucher a la vitesse."""
        n = self.dim
        self._x[:n] = np.asarray(position, dtype=np.float64).reshape(n)
        self._P[:n, :n] *= 0.25

    # -- cycle -------------------------------------------------------------

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        n = self.dim
        # x <- F x, avec F = [[I, dt I], [0, I]]. Ecrit directement plutot que
        # par produit matriciel: meme resultat, sans allocation.
        self._x[:n] += self._x[n:] * dt

        P = self._P
        pp, pv, vv = P[:n, :n], P[:n, n:], P[n:, n:]
        # P <- F P F^T, developpe bloc par bloc.
        new_pp = pp + dt * (pv + pv.T) + dt * dt * vv
        new_pv = pv + dt * vv
        P[:n, :n] = new_pp
        P[:n, n:] = new_pv
        P[n:, :n] = new_pv.T

        # Bruit de modele: acceleration blanche d'ecart-type accel_sigma. C'est
        # la formulation qui correspond a "le drone peut changer de vitesse",
        # et elle garde la matrice coherente entre position et vitesse.
        q = self.accel_sigma ** 2
        eye = np.eye(n)
        P[:n, :n] += q * (dt ** 4 / 4.0) * eye
        P[:n, n:] += q * (dt ** 3 / 2.0) * eye
        P[n:, :n] += q * (dt ** 3 / 2.0) * eye
        P[n:, n:] += q * (dt ** 2) * eye

    def update_velocity(self, measurement: np.ndarray,
                        sigma: Optional[float] = None) -> bool:
        """Corrige l'etat avec une mesure de vitesse.

        Renvoie False si la mesure est incompatible avec l'etat courant: elle
        est alors ecartee, et seule la prediction subsiste. C'est le garde-fou
        qui compte -- une homographie ajustee sur un mur produit un deplacement
        qui n'a rien a voir avec le precedent, et ce test le voit.
        """
        n = self.dim
        z = np.asarray(measurement, dtype=np.float64).reshape(n)
        if not np.isfinite(z).all():
            return False

        s = self.meas_sigma if sigma is None else max(1e-6, float(sigma))
        R = (s ** 2) * np.eye(n)
        S = self._P[n:, n:] + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        innovation = z - self._x[n:]
        # Distance de Mahalanobis: l'ecart mesure/prediction rapporte a
        # l'incertitude des deux. Un seuil en metres par seconde serait faux
        # des que l'etat est mal connu -- typiquement juste apres un decrochage
        # du suivi, ou il faut au contraire accepter la mesure.
        d2 = float(innovation @ S_inv @ innovation)
        if d2 > (self.gate_sigma ** 2) * n:
            return False

        K = self._P[:, n:] @ S_inv
        self._x += K @ innovation
        # Forme de Joseph: garde P symetrique definie positive meme apres de
        # longues series de mises a jour, ce que la forme courte (I-KH)P ne
        # garantit pas en flottant.
        H = np.zeros((n, 2 * n))
        H[:, n:] = np.eye(n)
        A = np.eye(2 * n) - K @ H
        self._P = A @ self._P @ A.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)
        return True


def wrap_angle(a: float) -> float:
    """Ramene un angle dans [-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class HeadingFilter:
    """Cap et vitesse de lacet, avec repliement d'angle correct.

    Le cap ne peut pas etre filtre comme une grandeur ordinaire: entre 179 et
    -179 degres il y a deux degres, pas trois cent cinquante-huit. Le filtre
    travaille donc sur la *vitesse* de lacet, qui elle est continue, et
    reconstitue le cap par integration repliee.
    """

    def __init__(self, rate_sigma: float = 25.0, meas_sigma: float = 8.0,
                 gate_sigma: float = 3.5) -> None:
        # Les reglages sont donnes en degres -- c'est ainsi qu'on raisonne sur
        # un cap -- mais l'etat travaille en radians, comme tout le reste du
        # module. La conversion est faite ici, une fois.
        self._filter = ConstantVelocity(dim=1, accel_sigma=np.radians(rate_sigma),
                                        meas_sigma=np.radians(meas_sigma),
                                        gate_sigma=gate_sigma)
        self._yaw = 0.0

    def reset(self) -> None:
        self._filter.reset()
        self._yaw = 0.0

    @property
    def yaw_rad(self) -> float:
        return self._yaw

    @property
    def rate_dps(self) -> float:
        return float(np.degrees(self._filter.velocity[0]))

    def set_yaw(self, yaw_rad: float) -> None:
        self._yaw = wrap_angle(float(yaw_rad))

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        rate = self._filter.velocity[0]
        self._filter.predict(dt)
        self._yaw = wrap_angle(self._yaw + rate * dt)

    def update_delta(self, delta_rad: float, dt: float,
                     sigma_dps: Optional[float] = None) -> bool:
        """Corrige avec un increment de cap mesure sur dt secondes."""
        if dt <= 0.0:
            return False
        delta = wrap_angle(float(delta_rad))
        sigma = None if sigma_dps is None else np.radians(sigma_dps)
        rate_before = self._filter.velocity[0]
        accepted = self._filter.update_velocity(np.array([delta / dt]), sigma)
        if accepted:
            # Le cap suit la correction apportee a la vitesse de lacet: sans
            # cela le filtre lisserait la vitesse mais le cap resterait celui
            # de la mesure brute, et le lissage ne servirait a rien.
            self._yaw = wrap_angle(self._yaw + (self._filter.velocity[0] - rate_before) * dt)
        return accepted


def sigma_from_parallax(depth_m: float, baseline_m: float, focal_px: float,
                        pixel_sigma: float = 0.5) -> float:
    """Incertitude de profondeur d'un point triangule.

    La profondeur d'une triangulation vaut z = f B / d, avec d la disparite en
    pixels. Une erreur de d se propage en dz = z^2 sigma_px / (f B): l'erreur
    croit avec le *carre* de la distance et s'effondre quand la base grandit.

    C'est la formule qui explique tout le comportement observe: un point
    lointain vu sur une base courte n'est pas un peu moins precis, il est
    inutilisable. La retenir permet de ponderer au lieu de seuiller.
    """
    b = abs(float(baseline_m))
    if b < 1e-9 or focal_px <= 0.0:
        return float("inf")
    return float(depth_m * depth_m * pixel_sigma / (focal_px * b))

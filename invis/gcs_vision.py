"""Station sol vision: 4 panneaux, analyse et reconstruction 3D en direct.

Lancement:
    python -m invis.gcs_vision                    # interface
    python -m invis.gcs_vision --source sim --connect
    python invis/gcs_vision.py --host 192.168.4.50

Disposition:
    1. haut gauche  -- image camera, champ de vecteurs, reperes de distance
    2. haut droite  -- mesures en direct et vue de dessus
    3. bas gauche   -- reconstruction 3D qui se construit au fil du vol
    4. bas droite   -- libre

Le programme est en lecture seule vis-a-vis du vol. Les seuls appels pouvant
modifier quelque chose sur la carte visent /control, qui ne touche que la
camera (taille d'image, qualite, cadence). Aucun endpoint /pilot n'est appele.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional, Tuple

import cv2
import numpy as np

if __package__ in (None, ""):  # execution directe du fichier
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from invis import config, geometry, overlay, panels  # noqa: E402
    from invis.detector import (  # noqa: E402
        STATE_CLEAR, STATE_NO_FLOW, STATE_OBSTACLE, ObstacleDetector,
    )
    from invis.framecheck import VERDICT_REJECT, FrameGate  # noqa: E402
    from invis.mapper import Mapper  # noqa: E402
    from invis.mjpeg_client import VideoLink, http_get  # noqa: E402
    from invis.recorder import SessionRecorder  # noqa: E402
    from invis.render3d import COLOUR_MODES, MODE_REAL, Renderer3D  # noqa: E402
    from invis.simulator import FlightSimulator, SimulatedLink, Wall  # noqa: E402
    from invis.updater import UpdateWatcher, install  # noqa: E402
    from invis.version import VERSION  # noqa: E402
else:
    from . import config, geometry, overlay, panels
    from .detector import STATE_CLEAR, STATE_NO_FLOW, STATE_OBSTACLE, ObstacleDetector
    from .framecheck import VERDICT_REJECT, FrameGate
    from .mapper import Mapper
    from .mjpeg_client import VideoLink, http_get
    from .recorder import SessionRecorder
    from .render3d import COLOUR_MODES, MODE_REAL, Renderer3D
    from .simulator import FlightSimulator, SimulatedLink, Wall
    from .updater import UpdateWatcher, install
    from .version import VERSION


class VisionApp(tk.Tk):
    def __init__(self, host: str, source: str) -> None:
        super().__init__()
        self.title(f"ESP32-CAM - station vision 3D  v{VERSION}")
        self.geometry("1440x900")
        self.minsize(1100, 720)

        self.link = None
        self.detector = ObstacleDetector()
        self.mapper = Mapper()
        self.gate = FrameGate(dump_dir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config.SESSION_DIR,
            "rejets_" + time.strftime("%Y%m%d_%H%M%S")))
        self.renderer = Renderer3D()
        self.recorder: Optional[SessionRecorder] = None

        self._worker: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._render_slot: Optional[Tuple[bytes, str]] = None
        self._render_lock = threading.Lock()
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._photo: Optional[tk.PhotoImage] = None
        self._cell = (480, 352)
        self._drag: Optional[Tuple[int, int]] = None
        self._pan: Optional[Tuple[int, int]] = None

        self._analysis_fps = 0.0
        self._analysis_window: list = []
        self._last_state: Optional[str] = None
        self._last_obstacle_log = 0.0
        self._quality_warned = 0.0
        self._last_result = None
        self._last_mapframe = None
        self._update_dismissed = False
        self._t0 = time.time()

        self.var_host = tk.StringVar(value=host)
        self.var_source = tk.StringVar(value=source)
        self.var_flow = tk.BooleanVar(value=True)
        self.var_ranges = tk.BooleanVar(value=True)
        self.var_follow = tk.BooleanVar(value=True)
        self.var_spin = tk.BooleanVar(value=False)
        self.var_surface = tk.BooleanVar(value=True)
        self.var_colour = tk.StringVar(value=MODE_REAL)
        self.var_record = tk.BooleanVar(value=False)
        self.var_gate = tk.BooleanVar(value=True)
        self.var_flip_h = tk.BooleanVar(value=config.CAMERA_FLIP_H)
        self.var_flip_v = tk.BooleanVar(value=config.CAMERA_FLIP_V)
        self.var_height = tk.StringVar(value=f"{config.DEFAULT_HEIGHT_M:.2f}")
        self.var_sigma = tk.StringVar(value=f"{config.DEFAULT_SIGMA_H_M:.2f}")
        self.var_sensitivity = tk.DoubleVar(value=1.0)
        self.var_status = tk.StringVar(value="deconnecte")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(30, self._pump_ui)

        # Consultation en arriere-plan. Un echec est normal et silencieux: en
        # vol le PC est sur le reseau du drone, sans acces exterieur.
        self._update = UpdateWatcher(on_found=lambda rel: self._log_queue.put(
            f"[maj] version {rel.version} disponible"))
        self._update.start()
        self.after(2000, self._poll_update)

    # -- interface ---------------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Hote").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.var_host, width=15).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(top, text="Source").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.var_source, values=("stream", "snapshot", "sim"),
                     width=9, state="readonly").pack(side=tk.LEFT, padx=(4, 8))
        self.btn_connect = ttk.Button(top, text="Connect", command=self._toggle_connect)
        self.btn_connect.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(top, text="h (m)").pack(side=tk.LEFT)
        e_h = ttk.Entry(top, textvariable=self.var_height, width=6)
        e_h.pack(side=tk.LEFT, padx=(4, 6))
        e_h.bind("<Return>", lambda _e: self._apply_height())
        ttk.Label(top, text="+/-").pack(side=tk.LEFT)
        e_s = ttk.Entry(top, textvariable=self.var_sigma, width=5)
        e_s.pack(side=tk.LEFT, padx=(4, 4))
        e_s.bind("<Return>", lambda _e: self._apply_height())
        ttk.Button(top, text="Appliquer", command=self._apply_height).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Checkbutton(top, text="Vecteurs", variable=self.var_flow).pack(side=tk.LEFT)
        ttk.Checkbutton(top, text="Distances", variable=self.var_ranges).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(top, text="Suivre", variable=self.var_follow,
                        command=self._apply_view).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(top, text="Rotation", variable=self.var_spin,
                        command=self._apply_view).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(top, text="Surface", variable=self.var_surface,
                        command=self._apply_view).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Combobox(top, textvariable=self.var_colour, values=COLOUR_MODES,
                     width=9, state="readonly").pack(side=tk.LEFT, padx=(6, 10))
        self.var_colour.trace_add("write", lambda *_: self._apply_view())
        ttk.Button(top, text="Reset 3D", command=self._reset_map).pack(side=tk.LEFT)
        ttk.Checkbutton(top, text="Miroir H", variable=self.var_flip_h,
                        command=self._on_flip).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(top, text="Miroir V", variable=self.var_flip_v,
                        command=self._on_flip).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Checkbutton(top, text="Filtre qualite", variable=self.var_gate).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(top, text="Enregistrer", variable=self.var_record,
                        command=self._toggle_record).pack(side=tk.LEFT, padx=(8, 10))

        ttk.Label(top, text="Sensi").pack(side=tk.LEFT)
        ttk.Scale(top, from_=0.5, to=2.0, variable=self.var_sensitivity,
                  orient=tk.HORIZONTAL, length=90,
                  command=self._on_sensitivity).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(top, text="Status", command=self._query_status).pack(side=tk.LEFT)

        body = ttk.PanedWindow(self, orient=tk.VERTICAL)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="#121214", highlightthickness=0)
        body.add(self.canvas, weight=4)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        # Bouton droit ou milieu: deplacement lateral de la vue.
        for press, motion in (("<ButtonPress-3>", "<B3-Motion>"),
                              ("<ButtonPress-2>", "<B2-Motion>")):
            self.canvas.bind(press, self._on_pan_start)
            self.canvas.bind(motion, self._on_pan)
        self.canvas.bind("<ButtonRelease-3>", lambda _e: setattr(self, "_pan", None))
        self.canvas.bind("<ButtonRelease-2>", lambda _e: setattr(self, "_pan", None))
        self.canvas.bind("<Double-Button-1>", self._on_recenter)
        self.canvas.bind("<MouseWheel>", self._on_wheel)

        console_frame = ttk.Frame(body)
        body.add(console_frame, weight=1)
        self.console = ScrolledText(console_frame, height=8, font=("Consolas", 9),
                                    bg="#141416", fg="#d8d8d8", insertbackground="#d8d8d8")
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.configure(state=tk.DISABLED)

        # Bandeau de mise a jour, masque tant qu'il n'y a rien a proposer.
        self.update_bar = ttk.Frame(self, padding=(8, 4))
        self.var_update = tk.StringVar(value="")
        ttk.Label(self.update_bar, textvariable=self.var_update).pack(side=tk.LEFT)
        self.btn_update = ttk.Button(self.update_bar, text="Installer",
                                     command=self._do_update)
        self.btn_update.pack(side=tk.LEFT, padx=(10, 4))
        ttk.Button(self.update_bar, text="Plus tard",
                   command=self._dismiss_update).pack(side=tk.LEFT)

        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bar, textvariable=self.var_status).pack(side=tk.LEFT)
        ttk.Label(bar, text=f"v{VERSION}").pack(side=tk.RIGHT)

    def log(self, msg: str) -> None:
        self._log_queue.put(f"[{time.time() - self._t0:7.2f}] {msg}")

    def _flush_log(self) -> None:
        lines = []
        while True:
            try:
                lines.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        if not lines:
            return
        self.console.configure(state=tk.NORMAL)
        for line in lines:
            self.console.insert(tk.END, line + "\n")
        if int(self.console.index("end-1c").split(".")[0]) > 2000:
            self.console.delete("1.0", "800.0")
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)

    # -- reglages ----------------------------------------------------------

    def _apply_height(self) -> None:
        try:
            h = float(self.var_height.get().replace(",", "."))
            s = float(self.var_sigma.get().replace(",", "."))
        except ValueError:
            self.log("hauteur invalide")
            return
        self.mapper.set_height(h, s)
        self.log(f"hauteur de vol {h:.2f} m +/- {s:.2f} m "
                 f"(toutes les distances sont proportionnelles a cette valeur)")

    def _on_flip(self) -> None:
        """L'orientation change la geometrie, donc la reconstruction repart."""
        self.detector.reset()
        self.mapper.reset()
        self.log(f"orientation: miroir H={self.var_flip_h.get()} V={self.var_flip_v.get()} "
                 f"-- reconstruction remise a zero, les distances en dependent")

    def _apply_view(self) -> None:
        self.renderer.auto_follow = self.var_follow.get()
        self.renderer.spin_dps = 12.0 if self.var_spin.get() else 0.0
        self.renderer.show_surface = self.var_surface.get()
        self.renderer.colour_mode = self.var_colour.get()

    def _reset_map(self) -> None:
        self.mapper.reset()
        self.detector.reset()
        self.log("reconstruction remise a zero")

    def _on_sensitivity(self, _value: str) -> None:
        self.detector.sensitivity = float(self.var_sensitivity.get())

    # -- souris sur la vue 3D ---------------------------------------------

    def _in_view3d(self, x: int, y: int) -> bool:
        cw, ch = self._cell
        return x < cw and y > ch

    def _on_drag_start(self, event) -> None:
        if self._in_view3d(event.x, event.y):
            self._drag = (event.x, event.y)

    def _on_drag(self, event) -> None:
        if self._drag is None:
            return
        dx = event.x - self._drag[0]
        dy = event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self.renderer.camera.orbit(-dx * 0.4, dy * 0.3)

    def _on_pan_start(self, event) -> None:
        if self._in_view3d(event.x, event.y):
            self._pan = (event.x, event.y)

    def _on_pan(self, event) -> None:
        """Deplace la vue, et lache le suivi automatique.

        Viser une zone precise pendant que la vue se recentre sur le drone a
        chaque image est impossible: se deplacer implique donc de cesser de
        suivre. La case *Suivre* le reactive.
        """
        if self._pan is None:
            return
        dx = event.x - self._pan[0]
        dy = event.y - self._pan[1]
        self._pan = (event.x, event.y)
        if self.var_follow.get():
            self.var_follow.set(False)
            self.renderer.auto_follow = False
            self.log("vue libre: le suivi du drone est relache (case Suivre pour revenir)")
        self.renderer.camera.pan(-dx, -dy)

    def _on_recenter(self, event) -> None:
        """Double-clic: revenir sur le drone."""
        if not self._in_view3d(event.x, event.y):
            return
        self.var_follow.set(True)
        self._apply_view()
        self.log("vue recentree sur le drone")

    def _on_wheel(self, event) -> None:
        if not self._in_view3d(event.x, event.y):
            return
        self.renderer.camera.zoom(0.88 if event.delta > 0 else 1.14)

    def _on_canvas_resize(self, event) -> None:
        cw = max(220, (event.width - 3) // 2)
        ch = max(180, (event.height - 3) // 2)
        self._cell = (cw, ch)
        self.renderer.resize(cw, ch - 16)

    # -- connexion ---------------------------------------------------------

    def _toggle_connect(self) -> None:
        if self.link is not None and self.link.running:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        source = self.var_source.get()
        self.detector.reset()
        self.mapper.reset()
        self.gate.reset()
        self._quality_warned = 0.0
        self._apply_height()
        self._apply_view()

        if source == "sim":
            sim = FlightSimulator(height_m=self.mapper.height_m, speed_mps=0.8,
                                  walls=[Wall(x_m=5.0, width_m=1.6, height_m=1.4)])
            self.link = SimulatedLink(sim, fps=7.0, on_log=self.log)
        else:
            host = self.var_host.get().strip() or config.DEFAULT_HOST
            self.link = VideoLink(host=host, mode=source, on_log=self.log)
            self.log(f"connexion a {host} ({source})")
            self.log("note: la carte ne sert qu'un flux, l'onglet navigateur sera coupe")
        self.link.start()

        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._analysis_loop, name="analysis", daemon=True)
        self._worker.start()
        self.btn_connect.configure(text="Disconnect")

    def _disconnect(self) -> None:
        self._worker_stop.set()
        if self._worker:
            self._worker.join(timeout=2.0)
        self._worker = None
        if self.link:
            self.link.stop()
        self._stop_record()
        self.btn_connect.configure(text="Connect")
        self.var_status.set("deconnecte")
        self.log("deconnecte")

    # -- commandes camera --------------------------------------------------

    def _query_status(self) -> None:
        host = self.var_host.get().strip()

        def work() -> None:
            try:
                self.log(f"/status {http_get(host, config.STATUS_PATH).strip()[:400]}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"/status echec: {exc}")

        threading.Thread(target=work, daemon=True).start()

    # -- enregistrement ----------------------------------------------------

    def _toggle_record(self) -> None:
        if self.var_record.get():
            try:
                self.recorder = SessionRecorder()
                self.log(f"enregistrement -> {self.recorder.dir}")
            except Exception as exc:  # noqa: BLE001
                self.var_record.set(False)
                self.log(f"enregistrement impossible: {exc}")
        else:
            self._stop_record()

    def _stop_record(self) -> None:
        if self.recorder:
            self.recorder.close()
            self.log("enregistrement arrete")
            self.recorder = None
        self.var_record.set(False)

    # -- boucle d'analyse (thread) ----------------------------------------

    def _analysis_loop(self) -> None:
        last = time.time()
        while not self._worker_stop.is_set():
            link = self.link
            if link is None:
                break
            frame = link.take_latest()
            if frame is None:
                time.sleep(0.004)
                continue

            bgr = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                self.log("image JPEG illisible, ignoree")
                continue

            # Redressement avant toute analyse: la ligne d'image porte la
            # distance, et un miroir simple inverse le sens du repere.
            bgr = geometry.orient_frame(bgr, self.var_flip_h.get(), self.var_flip_v.get())

            # Une image ecartee ne doit pas geler l'ecran.
            #
            # Le rejet portait aussi sur l'affichage: l'image restait figee
            # jusqu'au prochain passage du filtre, soit huit images -- plus
            # d'une seconde a la cadence de cette camera. Le flux, lui,
            # arrivait normalement. On saute donc l'*analyse*, jamais le
            # *rendu*: la derniere mesure valable reste affichee, signalee
            # comme telle, et l'image continue de defiler.
            quality = self.gate.check(bgr, jpeg_size=len(frame.jpeg))
            stale = self.var_gate.get() and quality.verdict == VERDICT_REJECT
            if stale:
                self._warn_quality(quality)
                result, mapframe = self._last_result, self._last_mapframe
                if result is None:
                    continue

            else:
                try:
                    result = self.detector.process(bgr, frame.recv_time)
                    mapframe = self.mapper.update(result, bgr)
                except cv2.error as exc:
                    self.log(f"analyse en echec: {exc}")
                    self.detector.reset()
                    continue
                self._last_result, self._last_mapframe = result, mapframe

            now = time.time()
            dt_render, last = now - last, now
            self._tick_analysis_fps()
            self._report(result, mapframe)

            if self.recorder and not stale:
                try:
                    self.recorder.write(result, frame.jpeg, frame_time=frame.recv_time)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"ecriture session en echec: {exc}")

            composed = self._compose(bgr, result, mapframe, dt_render, stale=stale)
            ppm = overlay.to_ppm(composed)
            if ppm:
                with self._render_lock:
                    self._render_slot = (ppm, self._status_line(result, mapframe))

    def _compose(self, bgr, result, mapframe, dt_render: float,
                 stale: bool = False) -> np.ndarray:
        cw, ch = self._cell
        view = overlay.draw(bgr, result, show_flow=self.var_flow.get(),
                            mapframe=mapframe, show_ranges=self.var_ranges.get(),
                            stale=stale)
        link_fps = self.link.stats.fps if self.link else 0.0
        measures = panels.draw_measures((cw, ch - 16), result, mapframe,
                                        link_fps=link_fps,
                                        analysis_fps=self._analysis_fps,
                                        sigma_h=self.mapper.sigma_h_m,
                                        gate=self.gate)
        view3d = self.renderer.render(self.mapper, mapframe, dt_render)
        spare = panels.draw_spare((cw, ch - 16))
        return panels.compose(view, measures, view3d, spare, (cw, ch))

    def _warn_quality(self, quality) -> None:
        """Signale une degradation du flux, sans inonder la console.

        Le taux de rejet compte plus que l'image isolee: il dit si le lien
        video se degrade, ce qu'aucun filtrage au sol ne rattrapera.
        """
        now = time.time()
        if now - self._quality_warned < 3.0:
            return
        self._quality_warned = now
        rate = self.gate.rejected / max(1, self.gate.total)
        corrupt = self.link.stats.corrupt if self.link else 0
        self.log(f"image ecartee: {quality.reason}  ({rate:.0%} du flux depuis la connexion)")
        if rate > 0.15:
            # Message volontairement sans diagnostic: la cause n'est pas
            # etablie. L'horloge capteur avait ete accusee a tort.
            self.log(f"flux video degrade ({rate:.0%} ecartees, {corrupt} images "
                     f"incompletes recues): les distances perdent en precision.")
            self.log(f"images ecartees conservees dans {self.gate.dump_dir}")

    def _tick_analysis_fps(self) -> None:
        now = time.time()
        self._analysis_window.append(now)
        while self._analysis_window and self._analysis_window[0] < now - 2.0:
            self._analysis_window.pop(0)
        if len(self._analysis_window) >= 2:
            span = self._analysis_window[-1] - self._analysis_window[0]
            self._analysis_fps = (len(self._analysis_window) - 1) / span if span > 0 else 0.0

    def _status_line(self, result, mapframe) -> str:
        st = self.link.stats if self.link else None
        net = (f"reseau {st.fps:4.1f} img/s  {st.kbps:6.1f} kB/s  perdues {st.dropped}"
               if st else "reseau --")
        return (f"{net}  |  analyse {self._analysis_fps:4.1f} img/s  |  "
                f"points {result.n_tracked}  |  nuage {mapframe.n_cloud} pts  |  "
                f"assiette {mapframe.tilt_deg:+.1f} deg")

    # -- console -----------------------------------------------------------

    def _report(self, result, mapframe) -> None:
        now = time.time()
        state = result.state

        if state != self._last_state:
            if state == STATE_OBSTACLE:
                self.log(self._obstacle_line(result, mapframe))
                self._last_obstacle_log = now
            elif state == STATE_CLEAR and self._last_state == STATE_OBSTACLE:
                self.log("LIBRE  obstacle leve")
            elif state == STATE_NO_FLOW:
                self.log("PAS DE FLUX  scene immobile, relief en attente de parallaxe")
            self._last_state = state
            return

        if state == STATE_OBSTACLE and now - self._last_obstacle_log > 0.7:
            self.log(self._obstacle_line(result, mapframe))
            self._last_obstacle_log = now

    @staticmethod
    def _obstacle_line(result, mapframe) -> str:
        worst = result.worst_cell
        zone = worst.name if worst else "?"
        ttc = f"{worst.ttc:.2f}s" if worst and worst.ttc else "n/a"
        parts = [f"OBSTACLE  zone={zone}  ttc={ttc}"]
        if mapframe is not None and mapframe.contact is not None:
            c = mapframe.contact
            parts.append(f"contact={c.range_m:.2f}m [{c.band[0]:.2f}-{c.band[1]:.2f}]")
        if mapframe is not None and mapframe.nearest is not None:
            n = mapframe.nearest
            parts.append(f"triang={n.range_m:.2f}m lat={n.lateral_m:+.2f}m "
                         f"haut={n.height_m:+.2f}m n={n.n_points}")
        return "  ".join(parts)

    # -- boucle d'affichage (thread principal) -----------------------------

    def _pump_ui(self) -> None:
        self._flush_log()

        with self._render_lock:
            slot, self._render_slot = self._render_slot, None

        if slot:
            ppm, status = slot
            try:
                self._photo = tk.PhotoImage(data=ppm)
                self.canvas.delete("frame")
                self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW, tags="frame")
            except tk.TclError as exc:
                self.log(f"affichage impossible: {exc}")
            self.var_status.set(status)
        elif self.link is not None and self.link.running and not self.link.stats.connected:
            self.var_status.set(f"connexion... {self.link.stats.last_error}")

        self.after(25, self._pump_ui)

    # -- mise a jour -------------------------------------------------------

    def _poll_update(self) -> None:
        """Affiche la proposition, mais jamais pendant une connexion.

        Changer le logiciel pendant qu'on s'en sert est le meilleur moyen de
        transformer une mise a jour en panne inexpliquee. Le bandeau attend
        donc la deconnexion.
        """
        release = self._update.available
        busy = self.link is not None and self.link.running
        if release and not busy and not self._update_dismissed:
            self.var_update.set(f"Version {release.version} disponible"
                                + (f" -- {release.notes}" if release.notes else ""))
            if not self.update_bar.winfo_ismapped():
                self.update_bar.pack(side=tk.BOTTOM, fill=tk.X, before=self.winfo_children()[-1])
        elif self.update_bar.winfo_ismapped():
            self.update_bar.pack_forget()
        self.after(2000, self._poll_update)

    def _dismiss_update(self) -> None:
        self._update_dismissed = True
        self.update_bar.pack_forget()

    def _do_update(self) -> None:
        release = self._update.available
        if release is None:
            return
        self.btn_update.configure(state=tk.DISABLED)

        def work() -> None:
            try:
                path = install(release, on_progress=lambda m: self._log_queue.put(f"[maj] {m}"))
                self._log_queue.put(f"[maj] installee dans {path}")
                self._log_queue.put("[maj] redemarre l'application pour l'utiliser")
                self._update.available = None
            except Exception as exc:  # noqa: BLE001
                self._log_queue.put(f"[maj] echec: {exc}")
            finally:
                self.btn_update.configure(state=tk.NORMAL)

        threading.Thread(target=work, daemon=True).start()

    def _on_close(self) -> None:
        try:
            if self.link is not None and self.link.running:
                self._disconnect()
        finally:
            self.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Station sol vision ESP32-CAM")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--source", default="stream", choices=("stream", "snapshot", "sim"))
    parser.add_argument("--connect", action="store_true", help="se connecter au demarrage")
    args = parser.parse_args()

    app = VisionApp(host=args.host, source=args.source)
    if args.connect:
        app.after(400, app._connect)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

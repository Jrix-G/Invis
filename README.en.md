# Invis

*[Version française](README.md)*

Ground station for a drone carrying an ESP32-CAM: metric distances, obstacle
detection and live 3D reconstruction of the scene, from a single camera.

## Getting started

### Linux

```bash
wget https://github.com/Jrix-G/Invis/releases/latest/download/Invis-1.0.3-linux-x86_64.tar.gz
tar xzf Invis-1.0.3-linux-x86_64.tar.gz
./Invis/Invis
```

The file to run is called `Invis`, **without `.exe`**. If it carries an `.exe`
extension you have the Windows archive: Linux hands it to Mono, which answers
`does not contain a valid CIL image`.

If the program refuses to start:

```bash
sudo apt install libgl1 libglib2.0-0
```

### Windows

Download
[`Invis-1.0.3-windows.zip`](https://github.com/Jrix-G/Invis/releases/latest),
unzip, run `Invis.exe`.

Windows will show *"Windows protected your PC"* on first launch: click **More
info** then **Run anyway**. The application is not code-signed.

If the program fails to start with a `DLL load failed` error mentioning
`cv2`: this is a Windows N/KN edition (shipped by default in some European
countries), which does not include the media player component Invis needs.
Installing the *Media Feature Pack* for your Windows edition from Microsoft's
site fixes it.

### Then, inside the application

1. Connect the machine to the **drone's Wi-Fi**
2. Host `192.168.4.50`, click **Connect**
3. Enter the **flight height** in the top bar, then *Appliquer*
4. If the image is upside down, toggle **Miroir H** / **Miroir V**

Height is the only metric quantity in the whole system: every distance is
proportional to it. If you don't know it, enter `1.00` — distances are then
expressed in flight heights, which stays exact.

Until the image is upright, **the distances are wrong**: the image row is what
carries distance.

### Without a drone

`Source` -> `sim` -> **Connect**. A complete synthetic flight, nothing to plug in.

## From source

```bash
git clone https://github.com/Jrix-G/Invis.git
cd Invis
sudo apt install python3-tk        # Linux: Tkinter does not always ship with Python
pip install -r requirements.txt
python -m invis.gcs_vision
```

The other tools:

```bash
python -m invis.gcs_vision --source sim --connect # interface, no drone
python -m invis.test_invis                        # 80 tests, no hardware
python -m invis.bench --host 192.168.4.50         # link frame-rate measurement
python -m invis.diag_stream --host 192.168.4.50   # stream drop diagnosis
python -m invis.replay <session> --height 2.4     # replay a recorded flight
```

---

**Read-only with respect to flight** — no `/pilot` endpoint is ever called, no
command is sent to the flight controller. The program observes, it does not
fly. The board's firmware lives in a separate repository: Invis only talks to
it over HTTP.

## The four panels

| | |
|---|---|
| **1. camera** — image, vector field, ground distance marks, contact line | **2. measurements** — distances, time to contact, attitude, frame rates, top view |
| **3. 3D reconstruction** — cloud built up during flight, trajectory, drone | **4. free** |

In the 3D view: **left click** to orbit (−88 to +88 degrees, including from
below), **right click** to pan, **double click** to re-centre, wheel to zoom.
Panning releases drone-follow; without that the view would re-centre on every
frame.

Vector field: **green** = point that follows the ground, **red** = point that
does not, therefore relief. Vector length is amplified, with a gain adapted to
speed so it stays readable.

## How distances are obtained

A single image carries no scale. Here the scale comes from elsewhere: the
camera sits at a known height above a plane. Three measurements follow, in
order of availability.

**1. Ray / ground intersection.** Every pixel is a ray; the ground is a plane
at distance `h` below the camera. `D = h / tan(alpha)`. Exact, immediate,
available even while hovering.

**2. Contact point.** An obstacle standing on the ground touches it somewhere,
and that point belongs to the plane: its distance is directly metric, without
parallax, from the very first frame the obstacle appears in. Mind the bias:
near the foot, the deviation from the ground tends to zero, so those points are
never flagged and the lowest *detected* point always sits too high. The program
extrapolates the deviation to zero to recover the contact line.

**3. Triangulation.** For anything not touching the ground (branch, cable, wall
seen head-on), two viewpoints. The displacement between the two frames is
measured and scaled by `h`; a rolling median per tracked point compensates for
the short baseline.

Accuracy measured against simulator ground truth, over three configurations
(see `test_invis.py`):

| | bias | median error |
|---|---|---|
| contact point | −0.10 to −0.30 m | 0.14 to 0.40 m |
| triangulation | −0.02 to +0.26 m | 0.04 to 0.14 m |
| recovered attitude | — | 0.14 to 0.41 deg |
| odometry over the run | — | 1 to 17 % drift |

## If the height `h` is wrong

This is the question that decides everything, because `h` is the only metric
quantity in the system.

**The error is purely multiplicative.** `D` is proportional to `h`: a height
overestimated by 25 % yields distances overestimated by 25 %, and nothing else
moves. The scene is not distorted, ratios between distances are exact, the
ordering "which one is closest" is exact, time to collision is exact. A test
checks this explicitly (`test_scale_invariance`).

Practical consequences:

- the interface shows an **interval** (`2.08 m [1.66 - 2.50]`) derived from the
  uncertainty you declare, rather than imaginary centimetres;
- a recorded session can be **replayed with a different height**
  (`replay.py --height`), which rescales every distance without recomputing
  anything else;
- with `h = 1`, distances are expressed in flight heights, and that reading is
  exact whatever happens.

**Attitude, however, is not a mere scale factor**: one degree of pitch error
shifts the impact point non-linearly, up to 10 % of the distance at the top of
the image. It is therefore not assumed but **measured on the ground itself**,
frame after frame, through the plane normal. That is the part of the geometry
an image *can* determine without scale. Result: 0.2 to 0.4 degree of typical
deviation, with no sensor at all.

For a truly reliable scale you need a metric input: a rangefinder, MAVLink
`RANGEFINDER`, or ground speed from the FC. All read-only.

## Drift and limits

- **Drift is bounded to three axes.** The ground supplies pitch, roll and
  altitude *without integration*, hence without drift. Only forward motion,
  lateral motion and heading are integrated. Heading drifts (nothing observes
  it without a magnetometer).
- **Range caps at ~2.2 times the flight height.** At 2 m altitude, a camera
  pitched 45 degrees down only sees the ground from 0.9 m to 4.4 m. An obstacle
  separates itself in practice around 1.4 to 1.9 m. That is not an algorithmic
  limit, it is the mounting: the exact value is displayed live (*portee du
  champ*).
- **Nothing for relief while hovering.** With no motion there is no parallax:
  state becomes `NO_FLOW`, triangulation stops, and that is displayed rather
  than hidden. Ground distances remain valid.
- **No action taken.** Nothing is transmitted to the FC.

## Frame rate

Cost measured on this machine, per frame, in QVGA:

```
detector 4.8 ms | map 1.3 ms | overlay 1.4 ms | measurements 1.7 ms
3D view 1.4 ms | composition 1.6 ms | encoding 2.0 ms   => ~14 ms, ceiling ~70 fps
```

The board supplies about 12 per second, a ceiling set by its firmware
(`max_fps=12`): the PC is nowhere near being the limiting factor.

The sensor runs at a 5 MHz clock, below the OV2640's usual range. Project
history reports that 10 and 20 MHz produced corrupted images, including outside
any Wi-Fi load. The cause is not established: the PSRAM cache workaround was
suspected then **disproved** — camera PSRAM DMA only exists on ESP32-S2 and S3,
and is hard-coded to false on the classic ESP32. `bench.py` measures,
`diag_stream.py` diagnoses stream drops; neither touches the clock.

Tracking runs at **full QVGA resolution**, not half. This is not a comfort
detail: for a wall at 2.7 m flown over at 2 m, the displacement difference
between the obstacle and the ground is ~3 px at 320x240 and ~1.5 px at 160x120
— below the plane-fitting threshold. Downsampling removed the very signal being
looked for.

## Files

| File | Role |
|---|---|
| `gcs_vision.py` | 4-panel interface, network / analysis / display threads |
| `mjpeg_client.py` | MJPEG stream split by `Content-Length`, latest frame only, resume |
| `detector.py` | point tracking, two fitted planes, ground selection, cells, hysteresis |
| `geometry.py` | intrinsics, ray/plane, attitude from the normal, uncertainty |
| `mapper.py` | attitude, odometry, ground contact, triangulation, point cloud |
| `render3d.py` | cloud projection and rendering, in numpy |
| `panels.py` | measurement panel, top view, 2x2 composition |
| `overlay.py` | vectors, grid, distance marks, contact line |
| `simulator.py` | synthetic world with ground truth, and video source without a drone |
| `recorder.py` / `replay.py` | CSV + image recording, ground replay |
| `framecheck.py` | frame quality control, degradation reporting |
| `updater.py` / `release.py` | signed updates, packaging and publishing |
| `build_app.py` | standalone executable build |
| `bench.py` / `diag_stream.py` | link frame rate, stream drop diagnosis |
| `test_invis.py` | 80 tests, no hardware |

## Settings

| Setting | Effect |
|---|---|
| `h` and `+/-` (interface) | metric scale and width of the displayed interval |
| *Sensi* slider | 0.5 (cautious) to 2.0 (twitchy) |
| `config.TTC_WARN_S` | time-to-collision warning threshold |
| `config.RESIDUAL_SIGMA_K` | strictness of the "off-ground" criterion, in standard deviations |
| `config.CONFIRM_HITS` | hysteresis strictness |
| `config.CAMERA_TILT_DEG` | nominal tilt, starting point for calibration |

Tune from a recorded session (`replay.py`), not in flight.

## Updates

The application checks for new versions at startup and **offers** to install
them — never while connected to the camera. Archives are Ed25519-signed; any
archive that is unsigned, altered, older, served in the clear, or whose paths
escape the target directory is refused.

Only the code travels (about 70 KB), not the executable (about 150 MB).

## Note on the interface language

The user interface is in French. The measurement panel labels are therefore
`contact sol` (ground contact), `triangulation`, `temps avant contact` (time to
contact), `vitesse estimee` (estimated speed), `hauteur retenue` (height used),
`assiette mesuree` (measured attitude), `portee du champ` (field range).

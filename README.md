<div align="center" style="font-family: 'Segoe UI', Roboto, sans-serif;">

# Invis

**Ground station for an ESP32-CAM drone — metric distances, obstacle detection, live 3D reconstruction, from one camera.**

</div>

<div style="border-radius: 20px; background: #f7ecd8; padding: 22px 26px; box-shadow: 6px 6px 0 #5b4636; border: 3px solid #5b4636; font-family: 'Segoe UI', Roboto, sans-serif;">

**Quick start**

| | |
|---|---|
| 🐧 **Linux** | `wget …/Invis-latest-linux.tar.gz && tar xzf … && ./Invis/Invis` |
| 🪟 **Windows** | Grab `Invis-1.0.3-windows.zip`, unzip, run `Invis.exe` |
| 🛸 **No drone** | `Source → sim → Connect` |

Set the **flight height** before anything else — every distance is proportional to it.

</div>

```

# From source
git clone https://github.com/Jrix-G/Invis.git && cd Invis
pip install -r requirements.txt
python -m invis.gcs_vision

# Tools
python -m invis.test_invis        # 126 checks, no hardware
python -m invis.bench --host 192.168.4.50   # link frame rate
python -m invis.replay <session> --height 2.4   # re-fly a recording
```

<div style="border-radius: 20px; background: #d8f1e8; padding: 18px 26px; box-shadow: 6px 6px 0 #2f6b54; border: 3px solid #2f6b54; font-family: 'Segoe UI', Roboto, sans-serif;">

**What it does**

`camera` knows the ground → **metric distances** · `contact point` + `triangulation` → positions · attitude, odometry & a **live 3D point cloud** come along. Read-only: no `/pilot`, never sends a command.

And it stays out of the way: **at most 2 persistent sockets** to the board, ever. The pilot WebSocket shares that HTTP server on an LRU-purged pool — a chatty ground station gets it purged and the drone latches LAND.

</div>

<div align="center" style="font-family: 'Segoe UI', Roboto, sans-serif; margin-top: 16px;">

*Updates are Ed25519-signed and offered at startup, never in flight.*

</div>
# TrackWatch — Multi-Camera YOLO Tracking with WebSocket Alerts

`trackwatch.py` is a multi-camera, headless-capable object detection and tracking daemon built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [OpenCV](https://opencv.org/). It monitors multiple video sources simultaneously, raises configurable alarms when objects remain visible beyond a threshold, and communicates with a web UI over WebSocket — including support for dynamically updating the detection region of interest (ROI) per camera at runtime.

---

## ✨ Features

### Multi-camera

- Unlimited concurrent video sources via a simple `CAMERAS` list
- Supported source types: RTSP streams, USB/device indices, local video files, YouTube URLs
- Each camera runs in its own thread with an isolated YOLO tracker instance
- RTSP auto-reconnect on stream loss; YouTube sources re-resolved via `yt-dlp`

### Detection & tracking

- YOLO-based object detection ([ByteTrack](https://docs.ultralytics.com/reference/trackers/byte_tracker/) by default)
- Per-camera overrides for confidence, resolution, frame-skip, watched classes, and alarm threshold
- Periodic tracker reset to prevent ever-growing track IDs
- Frame-skip control (`detect_every`) to trade latency for CPU load

### Alarm system

- Per-track timer: alarm triggers after an object is continuously visible for `ALARM_SECONDS`
- Pulsing red overlay on alarmed objects, centred banner with alarm count
- Alarm cleared event fired when a track disappears

### Dynamic ROI

- Optional pixel-space ROI `(x1, y1, x2, y2)` per camera — only the cropped region is sent to the model
- Outside the ROI is dimmed to 35 % brightness with a cyan border drawn on the preview window
- ROI can be set or cleared **at runtime** via WebSocket without restarting the process
- Changing the ROI automatically resets the tracker for that camera

### WebSocket integration

- Single persistent outbound connection per event (fire-and-forget)
- Persistent **inbound listener** coroutine that reconnects automatically on disconnect
- Full protocol documented below

### Display & output

- Optional OpenCV preview windows (disable with `show_video = False` for headless/server use)
- Optional per-source video recording (`save_video = True`)
- FPS counter and confidence score overlays
- GPU acceleration via `enable_gpu = True` (CUDA)

## 🏗️ Project Structure

```
YOLO-Interactive-Tracking-UI/
├── interactive_tracker.py   # Main Python tracking UI script
└── README.md                # You're here!
```

---

## 🏗️ Project Structure

```
YOLO-Interactive-Tracking-UI/
├── trackwatch.py          # Multi-camera tracking daemon (this project)
├── interactive_tracker.py # Single-camera interactive UI (legacy)
└── README.md
```

---

## 🛠️ Installation

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/ersitzt/trackwatch-ai.git
cd trackwatch-ai

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package         | Purpose                             |
| --------------- | ----------------------------------- |
| `ultralytics`   | YOLO detection & ByteTrack tracking |
| `opencv-python` | Video capture and display           |
| `numpy`         | Array operations                    |
| `yt-dlp`        | YouTube source resolution           |
| `websockets`    | WebSocket alarm hub (optional)      |
| `lapx`          | Linear assignment for ByteTrack     |

**GPU acceleration (optional):** install a CUDA-enabled PyTorch build before the above, then set `enable_gpu = True` in the script:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### 1. Configure cameras

Edit the `CAMERAS` list near the top of `trackwatch.py`:

```python
CAMERAS: list[CameraConfig] = [
    CameraConfig(
        source="rtsp://user:pass@192.168.1.100/stream1",
        camera_id=1,
        label="Front Door",
        alarm_seconds=5.0,
        roi=(100, 80, 1180, 640),   # optional static starting ROI
    ),
    CameraConfig(
        source="https://www.youtube.com/watch?v=...",
        camera_id=2,
        label="Test Feed",
        alarm_seconds=1.0,
    ),
]
```

All `CameraConfig` fields:

| Field           | Default                | Description                                               |
| --------------- | ---------------------- | --------------------------------------------------------- |
| `source`        | —                      | RTSP URL, device index, file path, or YouTube URL         |
| `camera_id`     | —                      | Integer ID sent with every WebSocket event                |
| `label`         | `str(camera_id)`       | Display label in the window title                         |
| `conf`          | global `conf`          | Detection confidence threshold                            |
| `imgsz`         | global `imgsz`         | Inference resolution                                      |
| `alarm_seconds` | global `ALARM_SECONDS` | Visibility duration before alarm                          |
| `watch_classes` | global `WATCH_CLASSES` | Set of COCO class IDs to watch                            |
| `detect_every`  | global `detect_every`  | Run detection every N frames                              |
| `roi`           | `None`                 | Initial pixel ROI `(x1, y1, x2, y2)`; `None` = full frame |

### 2. Configure globals

```python
WS_URL = "ws://localhost:3001"   # your WebSocket server; None to disable
ALARM_SECONDS = 5.0
WATCH_CLASSES = {0, 2, 3}        # 0=person, 2=car, 3=motorcycle
show_video = True                 # False for headless/server deployment
enable_gpu = False                # True for CUDA
model_file = "yolo11s.pt"
```

### 3. Run

```bash
python trackwatch.py
```

Press **`q`** in any preview window (or send SIGINT) to stop all cameras cleanly.

---

## 🔌 WebSocket Protocol

`trackwatch` connects to `WS_URL` (a WebSocket **server** you host, typically your web UI backend). The connection is bidirectional.

### Outbound — trackwatch → server

| Event           | Payload fields                                                                  | When sent                                               |
| --------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `camera_info`   | `cameras: [{camera_id, label, frame_width, frame_height, roi}]`                 | Immediately on every (re)connect                        |
| `alarm`         | `camera_id`, `track_id`, `class`, `duration`, `ts`                              | When a track exceeds `alarm_seconds` for the first time |
| `alarm_cleared` | `camera_id`, `track_id`, `ts`                                                   | When an alarmed track disappears                        |
| `roi_updated`   | `camera_id`, `roi` (scaled coords or null), `frame_width`, `frame_height`, `ts` | After every `set_roi` / `clear_roi` is applied          |

`camera_info` example (sent once on connect so the UI knows native resolutions):

```json
{
  "event": "camera_info",
  "cameras": [
    {
      "camera_id": 1,
      "label": "Front Door",
      "frame_width": 1920,
      "frame_height": 1080,
      "roi": null
    },
    {
      "camera_id": 2,
      "label": "YouTube Test",
      "frame_width": 1280,
      "frame_height": 720,
      "roi": [120, 60, 960, 540]
    }
  ]
}
```

`alarm` example:

```json
{
  "event": "alarm",
  "camera_id": 1,
  "track_id": 7,
  "class": "person",
  "duration": 6.2,
  "ts": 1745600000
}
```

### Inbound — server → trackwatch

ROI coordinates use **normalized values (0.0–1.0)** relative to the display dimensions. This removes any dependency on matching pixel resolutions between the UI and the camera.

| Event       | Required fields                                    | Effect                                                                              |
| ----------- | -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `set_roi`   | `camera_id`, `roi: [x1n, y1n, x2n, y2n]` (0.0–1.0) | Set/update ROI; trackwatch multiplies by the actual frame size. Resets the tracker. |
| `clear_roi` | `camera_id`                                        | Remove ROI (full frame); resets tracker                                             |

The web UI should normalize before sending:

```
roi_normalized = [x1/displayWidth, y1/displayHeight, x2/displayWidth, y2/displayHeight]
```

Example:

```json
{"event": "set_roi",   "camera_id": 1, "roi": [0.3516, 0.4792, 0.9531, 0.8194]}
{"event": "clear_roi", "camera_id": 1}
```

`roi_updated` and `camera_info` also return `roi` in normalized form so the UI can draw it back correctly at any display size.

> `trackwatch` maintains **one** persistent WebSocket connection for both outbound events and inbound commands. It reconnects automatically with a 5 s back-off if the server is unavailable.

---

## 🧑‍💻 Web UI — ROI Implementation Prompts

The following prompts are ready to paste into GitHub Copilot (or any AI assistant) inside your web UI project to implement the ROI feature end-to-end. They are intentionally framework-agnostic — adapt the technology names to match your stack before pasting.

---

### Prompt 1 — WebSocket server hub

> Use this for a backend server that sits between `trackwatch` and the browser clients.

```
I have a WebSocket server running on port 3001.
It receives events from a Python process called trackwatch that connects as a client.
Trackwatch sends JSON messages: { event: 'detected' | 'alarm' | 'alarm_cleared', camera_id, track_id, class, duration?, ts }.
The server should:
1. Maintain a registry of connected browser clients (separate from the trackwatch client).
2. Broadcast every inbound trackwatch event to all connected browser clients.
3. Accept inbound messages from browser clients with events 'set_roi' and 'clear_roi':
   - set_roi:   { event: 'set_roi',   camera_id: number, roi: [x1, y1, x2, y2] }
   - clear_roi: { event: 'clear_roi', camera_id: number }
4. Forward these ROI commands directly to the connected trackwatch client.
5. Maintain the last known ROI per camera_id in memory. When a new browser client connects,
   send it the current ROI state for all cameras as a 'roi_state' event:
   { event: 'roi_state', cameras: { [camera_id]: [x1,y1,x2,y2] | null } }
Keep existing message handling intact. Add reconnection handling if trackwatch disconnects.
```

---

### Prompt 2 — ROI canvas component

> Use this to build a drag-to-draw ROI overlay on top of a camera image or MJPEG feed.

```
Create a UI component called RoiEditor that:
- Accepts: cameraId (number), imageUrl (string), initialRoi ([x1,y1,x2,y2] or null), and a callback onRoiChange(roi)
- Renders the camera image inside a positioned container
- Overlays an HTML5 canvas of the same size for drawing
- Allows the user to draw a new ROI rectangle by clicking and dragging on the canvas
- While dragging, shows a semi-transparent cyan rectangle with a 2 px cyan border as a preview
- On mouse/pointer release, calls onRoiChange with **normalized** coordinates [x1n, y1n, x2n, y2n]
  where each value = pixel coordinate / display dimension (all values 0.0–1.0)
- Always renders the current initialRoi (already normalized) as a persistent cyan rectangle,
  multiplying back by display width/height to get draw coordinates
- Provides a "Clear ROI" button that calls onRoiChange(null)
- Uses no external drawing libraries
```

---

### Prompt 3 — WebSocket state manager

> Use this to connect the browser to the server hub and manage ROI + alert state.

```
Create a WebSocket state manager / service called TrackWatchClient that:
- Accepts wsUrl (string) as configuration
- Opens a WebSocket connection, reconnects with exponential back-off (max 10 s) on disconnect
- Parses every inbound JSON message and maintains:
  - alerts: list of active alarm objects { camera_id, track_id, class, duration, ts }
  - rois: map of camera_id → [x1,y1,x2,y2] or null (current ROI per camera)
    populated from the 'roi_state' event on connect and kept up to date by 'set_roi' / 'clear_roi' events
- Exposes:
  - setRoi(cameraId, [x1,y1,x2,y2]) — sends a set_roi command to the server
  - clearRoi(cameraId) — sends a clear_roi command to the server
  - subscribe(listener) / unsubscribe(listener) — notify listeners on any state change
- Cleans up the socket when destroyed
```

---

### Prompt 4 — Camera management view

> Wires the state manager and ROI editor together into a single camera card/view.

```
Create a UI component called CameraCard that:
- Accepts: cameraId (number), label (string), feedUrl (string)
- Retrieves the current ROI and alarm list from the TrackWatchClient state manager
- Shows the camera label and ID in a header
- Embeds the RoiEditor component, passing the current roi for this camera
- When the editor emits a non-null roi, calls setRoi(cameraId, roi)
- When the editor emits null, calls clearRoi(cameraId)
- Shows a status badge: "ROI active [x1, y1, x2, y2]" when a ROI is set, or "Full frame" otherwise
- Lists active alarms for this camera showing class name, track ID, and how long the object has been visible
```

---

### Prompt 5 — ROI persistence (server-side, optional)

> Use this if you want ROI settings to survive a server restart.

```
In the existing WebSocket server, add ROI persistence:
- On every set_roi or clear_roi command, write the full rois map to a local JSON file (e.g. roi_state.json)
  inside a try/catch so a write error never crashes the server
- On startup, read roi_state.json if it exists and pre-populate the in-memory rois map
- When trackwatch reconnects, immediately send the persisted ROI for each camera as a set_roi command
  so the Python process picks up the last-known ROI without any browser interaction
```

---

## 📜 License

AGPL-3.0 — see [Ultralytics Licensing](https://www.ultralytics.com/license).

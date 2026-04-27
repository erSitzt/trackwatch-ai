from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
enable_gpu = False                          # Set True if running with CUDA
model_file = "yolo26s.pt"                   # Path to model file
show_fps = True                             # Show current FPS overlay
show_conf = True                            # Show confidence score in label
show_video = False                           # Set False to disable all display windows (headless mode)
save_video = False                          # Set True to save per-source video
video_output_template = "trackwatch_cam{idx}.avi"  # {idx} replaced per source

conf = 0.60         # Min detection confidence
iou = 0.5           # IoU threshold for NMS
max_det = 30        # Maximum detections per frame
imgsz = 224         # Inference resolution (320/416/640); smaller = faster, less accurate
detect_every = 2    # Run detection every N frames (1 = every frame, 2 = every other, …)

tracker = "bytetrack.yaml"
track_args = {
    "persist": True,
    "verbose": False,
}

ALARM_SECONDS = 5.0         # Continuous visibility before alarm triggers
WATCH_CLASSES = {0, 2, 3}   # COCO class IDs: 0 = person, 2 = car, 3 = motorcycle
LOOP_YOUTUBE = True         # Re-open YouTube sources when they reach the end
TRACKER_RESET_INTERVAL = 300.0  # Seconds between full tracker resets (0 = disabled)
WS_URL: str | None = "ws://localhost:3001"  # WebSocket URL for alarm events (None = disabled)

# ---------------------------------------------------------------------------
# TrackWatch web-app integration
# ---------------------------------------------------------------------------
# Set SERVER_HOST to the IP/hostname the AI service uses to reach TrackWatch.
# Set API_CAMERAS_ENABLED = False to skip the API call entirely.
SERVER_HOST: str = "localhost"
SERVER_PORT: int = 3000
API_CAMERAS_ENABLED: bool = True       # False = use only CAMERAS list below
API_CAMERAS_TIMEOUT: float = 5.0       # Seconds before giving up on the API call
CAMERA_POLL_INTERVAL: float = 30.0    # Seconds between runtime camera-list refreshes (0 = disabled)


# ---------------------------------------------------------------------------
# Per-camera configuration
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    source: int | str           # URL, RTSP address, device index, or file path
    camera_id: int              # Logical ID used when sending alerts (device number)
    label: str = ""             # Human-readable name shown in the window title
    # Per-camera overrides (None = use global value)
    conf: float | None = None
    imgsz: int | None = None
    alarm_seconds: float | None = None
    watch_classes: set[int] | None = None
    detect_every: int | None = None  # per-camera frame-skip override
    roi: tuple[int, int, int, int] | None = None  # Pixel ROI (x1, y1, x2, y2); None = full frame

    def __post_init__(self) -> None:
        if not self.label:
            self.label = str(self.camera_id)


# ---------------------------------------------------------------------------
# Camera list — local overrides / extras
# ---------------------------------------------------------------------------
# These cameras are always started.  When API_CAMERAS_ENABLED is True, cameras
# fetched from the API are merged in at runtime (API cameras that share a
# camera_id with a local entry are skipped so local config wins).
CAMERAS: list[CameraConfig] = [
    #  CameraConfig(
    #      source="rtsp://gsccam:Start123@192.168.2.168/stream2",
    #      camera_id=4,
    #      label="Entrance RTSP",
    #      alarm_seconds=5.0,
    #  ),
    #  CameraConfig(
    #      source="https://www.youtube.com/watch?v=2-tYpDcTcKQ",
    #      camera_id=3,
    #      label="YouTube Test",
    #      alarm_seconds=1.0,
    #  ),
    # CameraConfig(
    #     source="https://www.youtube.com/watch?v=_k-qyGNptY0",
    #     camera_id=2,
    #     label="YouTube Loop",
    #     alarm_seconds=1.0,
    # ),
    # CameraConfig(source=0, camera_id=3, label="Webcam"),
    # CameraConfig(source="/path/to/video.mp4", camera_id=4, label="Recording"),
    # CameraConfig(source=0, camera_id=5, label="Webcam ROI", roi=(100, 80, 540, 400)),  # only analyse this pixel rect
]


# ---------------------------------------------------------------------------
# API camera fetch
# ---------------------------------------------------------------------------
def fetch_cameras() -> list[CameraConfig]:
    """Return the merged camera list: API cameras + local CAMERAS.

    Local entries always win when camera_id collides with an API entry.
    If the API is disabled or unreachable, only CAMERAS is returned.
    """
    merged: list[CameraConfig] = list(CAMERAS)  # start with local list
    local_ids: set[int] = {c.camera_id for c in CAMERAS}

    if not API_CAMERAS_ENABLED:
        return merged

    url = f"http://{SERVER_HOST}:{SERVER_PORT}/api/cameras/ai-config"
    try:
        req = urllib.request.urlopen(url, timeout=API_CAMERAS_TIMEOUT)  # noqa: S310
        data: list[dict] = json.loads(req.read())
    except urllib.error.URLError as exc:
        import warnings
        warnings.warn(f"[config] Could not reach camera API ({url}): {exc} — using local CAMERAS only.")
        return merged
    except Exception as exc:  # malformed JSON, etc.
        import warnings
        warnings.warn(f"[config] Unexpected error fetching camera API: {exc} — using local CAMERAS only.")
        return merged

    for entry in data:
        cam_id = int(entry["camera_id"])
        if cam_id in local_ids:
            continue  # local config takes priority
        merged.append(CameraConfig(
            source=entry["source"],
            camera_id=cam_id,
            label=entry.get("label", ""),
            alarm_seconds=entry.get("alarm_seconds"),
        ))

    return merged

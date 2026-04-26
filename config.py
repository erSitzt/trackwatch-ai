from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
enable_gpu = False                          # Set True if running with CUDA
model_file = "yolo26s.pt"                   # Path to model file
show_fps = True                             # Show current FPS overlay
show_conf = True                            # Show confidence score in label
show_video = True                           # Set False to disable all display windows (headless mode)
save_video = False                          # Set True to save per-source video
video_output_template = "trackwatch_cam{idx}.avi"  # {idx} replaced per source

conf = 0.65         # Min detection confidence
iou = 0.3           # IoU threshold for NMS
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
# Camera list — add / remove entries freely
# ---------------------------------------------------------------------------
CAMERAS: list[CameraConfig] = [
    CameraConfig(
        source="rtsp://gsccam:Start123@192.168.2.101/stream1",
        camera_id=1,
        label="Entrance RTSP",
    ),
    CameraConfig(
        source="https://www.youtube.com/watch?v=2-tYpDcTcKQ",
        camera_id=3,
        label="YouTube Test",
        alarm_seconds=1.0,
    ),
    CameraConfig(
        source="https://www.youtube.com/watch?v=_k-qyGNptY0",
        camera_id=2,
        label="YouTube Loop",
        alarm_seconds=1.0,
    ),
    # CameraConfig(source=0, camera_id=3, label="Webcam"),
    # CameraConfig(source="/path/to/video.mp4", camera_id=4, label="Recording"),
    # CameraConfig(source=0, camera_id=5, label="Webcam ROI", roi=(100, 80, 540, 400)),  # only analyse this pixel rect
]

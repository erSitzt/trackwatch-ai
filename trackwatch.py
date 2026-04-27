# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# trackwatch.py — Auto-tracks people and motorcycles across multiple sources.
# Raises overlay alarm after ALARM_SECONDS of continuous visibility per track.

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

from config import (
    ALARM_SECONDS,
    CAMERA_POLL_INTERVAL,
    CAMERAS,
    LOOP_YOUTUBE,
    TRACKER_RESET_INTERVAL,
    WATCH_CLASSES,
    WS_URL,
    CameraConfig,
    conf,
    detect_every,
    enable_gpu,
    fetch_cameras,
    imgsz,
    iou,
    max_det,
    model_file,
    save_video,
    show_conf,
    show_fps,
    show_video,
    track_args,
    tracker,
    video_output_template,
)

# ---------------------------------------------------------------------------
# WebSocket — shared background event loop + outbound queue
# ---------------------------------------------------------------------------
# A single asyncio loop runs in a daemon thread.  All outbound events are
# put into _ws_send_queue; _ws_hub maintains ONE persistent connection that
# drains the queue (send) and reads inbound ROI commands (receive) in
# parallel.  Workers never open their own connections.
_ws_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_ws_send_queue: asyncio.Queue[str] = asyncio.Queue()

def _start_ws_loop() -> None:
    asyncio.set_event_loop(_ws_loop)
    _ws_loop.run_forever()

_ws_thread = threading.Thread(target=_start_ws_loop, daemon=True, name="ws-loop")
_ws_thread.start()


# ---------------------------------------------------------------------------
# Source resolver (YouTube → direct CDN URL via yt-dlp)
# ---------------------------------------------------------------------------
def resolve_source(src: int | str) -> int | str:
    """Return a URL/index suitable for cv2.VideoCapture."""
    if isinstance(src, int) or (isinstance(src, str) and not src.startswith("http")):
        return src
    yt_domains = ("youtube.com", "youtu.be")
    if any(d in str(src) for d in yt_domains):
        LOGGER.info(f"[resolve] Resolving YouTube URL: {src}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-g", "-f", "best[ext=mp4]/best", src],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
            direct_url = result.stdout.strip().splitlines()[0]
            LOGGER.info(f"[resolve] Resolved to: {direct_url[:80]}...")
            return direct_url
        except FileNotFoundError:
            raise SystemError("yt-dlp not installed. Run: pip install yt-dlp") from None
    return src


# ---------------------------------------------------------------------------
# Per-source worker — runs in its own thread
# ---------------------------------------------------------------------------
class CameraWorker:
    """Captures, tracks, and annotates frames for a single video source."""

    def __init__(self, idx: int, cfg: CameraConfig, loop: bool = False) -> None:
        self.idx = idx
        self.cfg = cfg
        self.camera_id = cfg.camera_id
        self.loop = loop
        self.window_name = f"TrackWatch [{cfg.camera_id}] {cfg.label}"

        # Effective per-camera values (fall back to globals)
        self._conf = cfg.conf if cfg.conf is not None else conf
        self._imgsz = cfg.imgsz if cfg.imgsz is not None else imgsz
        self._alarm_seconds = cfg.alarm_seconds if cfg.alarm_seconds is not None else ALARM_SECONDS
        self._watch_classes = cfg.watch_classes if cfg.watch_classes is not None else WATCH_CLASSES
        self._detect_every = cfg.detect_every if cfg.detect_every is not None else detect_every
        self._roi: tuple[int, int, int, int] | None = cfg.roi  # (x1, y1, x2, y2); updated at runtime via WS
        self._roi_lock = threading.Lock()
        self._ws_url: str | None = WS_URL

        self.resolved = resolve_source(cfg.source)

        # Each worker owns its model so tracker state is fully isolated.
        LOGGER.info(f"[{self.camera_id}] Loading model...")
        self.model = YOLO(model_file, task="detect")
        if enable_gpu:
            self.model.to("cuda")
        self.classes = self.model.names

        self.cap = cv2.VideoCapture(self.resolved)
        if not self.cap.isOpened():
            raise SystemError(f"[{self.camera_id}] Failed to open source: {cfg.source}")

        self.track_first_seen: dict[int, float] = {}
        self._prev_active_ids: set[int] = set()  # IDs seen in the last frame
        self._alarmed_ids: set[int] = set()       # IDs whose alarm was already logged
        self._last_tracker_reset: float = time.time()
        self._frame_count: int = 0  # counts captured frames for skip logic

        self.vw: cv2.VideoWriter | None = None
        self.vw_path = video_output_template.format(idx=idx)

        self.fps_counter = 0
        self.fps_timer = time.time()
        self.fps_display = 0

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._capture_size: tuple[int, int] | None = None  # set from first decoded frame

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam{idx}")

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """Update the active ROI at runtime (thread-safe). Pass None to remove it."""
        with self._roi_lock:
            self._roi = roi
        LOGGER.info(f"[{self.camera_id}] ROI {'cleared' if roi is None else f'set → {roi}'}")
        self._reset_tracker()  # existing tracks outside the new region are no longer valid

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """Return (width, height) of the capture source.

        Prefers the size measured from an actual decoded frame; falls back to
        the VideoCapture property (which may be 0 for RTSP until the stream
        starts).
        """
        if self._capture_size is not None:
            return self._capture_size
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w > 0 and h > 0 else None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)
        self.cap.release()
        if self.vw is not None:
            self.vw.release()

    def get_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _ws_send(self, payload: dict) -> None:
        """Enqueue payload for the shared persistent WebSocket connection."""
        if not self._ws_url:
            return
        LOGGER.info(f"[{self.camera_id}] WS → {payload['event']} {json.dumps({k: v for k, v in payload.items() if k != 'event'})}")
        asyncio.run_coroutine_threadsafe(_ws_send_queue.put(json.dumps(payload)), _ws_loop)

    def _reset_tracker(self) -> None:
        """Reset the YOLO tracker so track IDs restart from 1."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None  # forces re-init on next track() call
        self.track_first_seen.clear()
        self._prev_active_ids.clear()
        self._alarmed_ids.clear()
        self._last_tracker_reset = time.time()
        LOGGER.info(f"[{self.camera_id}] Tracker reset — IDs restarting from 1")

    def _run(self) -> None:
        while not self._stop_event.is_set() and self.cap.isOpened():
            success, im = self.cap.read()
            if not success:
                if isinstance(self.resolved, str) and self.resolved.startswith("rtsp"):
                    LOGGER.warning(f"[{self.camera_id}] Stream lost — retrying in 2 s...")
                    time.sleep(2)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.resolved)
                    continue
                if self.loop:
                    LOGGER.info(f"[{self.camera_id}] End of source — looping...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.resolved)
                    self._reset_tracker()
                    continue
                break

            # Cache the real frame dimensions from the first decoded frame and
            # notify the UI so it can use the correct coordinate space for ROI.
            if self._capture_size is None:
                self._capture_size = (im.shape[1], im.shape[0])
                LOGGER.info(
                    f"[{self.camera_id}] Capture size: "
                    f"{self._capture_size[0]}x{self._capture_size[1]}"
                )
                with self._roi_lock:
                    px_roi = self._roi
                fw, fh = self._capture_size
                norm_roi = _roi_to_norm(px_roi, fw, fh) if px_roi else None
                self._ws_send({
                    "event": "camera_info",
                    "cameras": [{
                        "camera_id": self.camera_id,
                        "label": self.cfg.label,
                        "frame_width": fw,
                        "frame_height": fh,
                        "roi": norm_roi,
                    }],
                })

            annotated = self._process(im) if self._frame_count % self._detect_every == 0 else self._latest_frame if self._latest_frame is not None else im
            self._frame_count += 1

            # Periodic tracker reset to prevent ever-growing IDs
            if TRACKER_RESET_INTERVAL > 0 and (time.time() - self._last_tracker_reset) >= TRACKER_RESET_INTERVAL:
                self._reset_tracker()

            if save_video and self.vw is None:
                h, w = annotated.shape[:2]
                src_fps = self.cap.get(cv2.CAP_PROP_FPS)
                out_fps = float(src_fps) if src_fps and src_fps > 0 else 30.0
                fourcc = cv2.VideoWriter_fourcc(*("MJPG" if self.vw_path.endswith(".avi") else "mp4v"))
                self.vw = cv2.VideoWriter(self.vw_path, fourcc, out_fps, (w, h))

            if save_video and self.vw is not None:
                self.vw.write(annotated)

            with self._frame_lock:
                self._latest_frame = annotated

    def _process(self, im: np.ndarray) -> np.ndarray:
        with self._roi_lock:
            roi = self._roi
        if roi is not None:
            # Clamp to actual frame bounds so a stale or mis-scaled ROI never
            # produces an empty / out-of-bounds numpy slice.
            fh, fw = im.shape[:2]
            rx1 = max(0, min(roi[0], fw - 1))
            ry1 = max(0, min(roi[1], fh - 1))
            rx2 = max(0, min(roi[2], fw))
            ry2 = max(0, min(roi[3], fh))
            if rx2 <= rx1 or ry2 <= ry1:
                LOGGER.warning(
                    f"[{self.camera_id}] ROI {roi} is degenerate after clamping to "
                    f"{fw}x{fh} — ignoring ROI this frame"
                )
                roi = None
            else:
                roi = (rx1, ry1, rx2, ry2)
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            crop = im[ry1:ry2, rx1:rx2]
            results = self.model.track(crop, conf=self._conf, iou=iou, max_det=max_det, imgsz=self._imgsz, tracker=tracker, **track_args)
            ox, oy = rx1, ry1
        else:
            results = self.model.track(im, conf=self._conf, iou=iou, max_det=max_det, imgsz=self._imgsz, tracker=tracker, **track_args)
            ox, oy = 0, 0

        annotator = Annotator(im)
        detections = results[0].boxes.data if results[0].boxes is not None else []
        raw = detections.cpu().tolist() if hasattr(detections, "cpu") else list(detections)

        active_ids: set[int] = set()
        log_parts: list[str] = []
        ws_detections: list[dict] = []
        now = time.time()
        frame_h_full, frame_w_full = im.shape[:2]

        for track in raw:
            if len(track) < 6:
                continue
            x1, y1, x2, y2 = int(track[0]) + ox, int(track[1]) + oy, int(track[2]) + ox, int(track[3]) + oy
            conf_score = float(track[5]) if len(track) >= 7 else 0.0
            class_id = int(track[6]) if len(track) >= 7 else int(track[5])
            track_id = int(track[4]) if len(track) == 7 else -1

            if class_id not in self._watch_classes:
                continue

            class_name = self.classes.get(class_id, str(class_id))
            color = colors(track_id, True)
            txt_color = annotator.get_txt_color(color)

            if track_id not in self.track_first_seen:
                self.track_first_seen[track_id] = now
            duration = now - self.track_first_seen[track_id]
            alarm = duration >= self._alarm_seconds
            active_ids.add(track_id)

            duration_str = f"{duration:.1f}s"
            conf_part = f" ({conf_score:.2f})" if show_conf else ""
            if alarm:
                label = f"!! ALARM !! {class_name} ID {track_id} [{duration_str}]{conf_part}"
            else:
                label = f"{class_name} ID {track_id} [{duration_str}]{conf_part}"

            log_parts.append(f"{class_name}#{track_id}@{duration_str}{'[ALARM]' if alarm else ''}")

            ws_detections.append({
                "track_id": track_id,
                "class": class_name,
                "conf": round(conf_score, 3),
                "alarm": alarm,
                "box": [
                    round(x1 / frame_w_full, 4),
                    round(y1 / frame_h_full, 4),
                    round(x2 / frame_w_full, 4),
                    round(y2 / frame_h_full, 4),
                ],
            })

            if alarm:
                alarm_color = (0, 0, 255)
                pulse = 2 + int(3 * abs(now % 1 - 0.5))
                annotator.box_label([x1, y1, x2, y2], label=label, color=alarm_color)
                cv2.rectangle(im, (x1 - pulse, y1 - pulse), (x2 + pulse, y2 + pulse), alarm_color, pulse)
            else:
                for i in range(x1, x2, 10):
                    cv2.line(im, (i, y1), (i + 5, y1), color, 2)
                    cv2.line(im, (i, y2), (i + 5, y2), color, 2)
                for i in range(y1, y2, 10):
                    cv2.line(im, (x1, i), (x1, i + 5), color, 2)
                    cv2.line(im, (x2, i), (x2, i + 5), color, 2)
                (tw, th), bl = cv2.getTextSize(label, 0, 0.6, 1)
                cv2.rectangle(im, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(im, label, (x1 + 3, y1 - 4), 0, 0.6, txt_color, 1, cv2.LINE_AA)

        _debug = os.environ.get("DEBUG", "").lower() == "true"
        to_send = ws_detections if _debug else [d for d in ws_detections if d["alarm"]]
        if to_send or _debug:
            self._ws_send({
                "event": "detections",
                "camera_id": self.camera_id,
                "ts": now,
                "detections": to_send,
            })

        for stale_id in list(self.track_first_seen.keys()):
            if stale_id not in active_ids:
                del self.track_first_seen[stale_id]
                if stale_id in self._alarmed_ids:
                    self._alarmed_ids.discard(stale_id)
                    LOGGER.info(f"[{self.camera_id}] ALARM CLEARED ID {stale_id} — no longer visible")
                    self._ws_send({
                        "event": "alarm_cleared",
                        "camera_id": self.camera_id,
                        "track_id": stale_id,
                        "ts": now,
                    })
                else:
                    self._alarmed_ids.discard(stale_id)

        # Log only IDs appearing for the first time this run, and alarm transitions.
        new_ids = active_ids - self._prev_active_ids
        for track_id in new_ids:
            ts = self.track_first_seen.get(track_id, now)
            class_name = next(
                (self.classes.get(int(t[6]) if len(t) >= 7 else int(t[5]), str(int(t[6]) if len(t) >= 7 else int(t[5])))
                 for t in raw if len(t) >= 5 and int(t[4]) == track_id),
                str(track_id),
            )
            LOGGER.info(f"[{self.camera_id}] NEW {class_name} ID {track_id} detected")

        # Log alarm transitions (first time threshold is crossed).
        for track_id, ts in self.track_first_seen.items():
            if track_id in active_ids and (now - ts) >= self._alarm_seconds and track_id not in self._alarmed_ids:
                class_name = next(
                    (self.classes.get(int(t[6]) if len(t) >= 7 else int(t[5]), str(int(t[6]) if len(t) >= 7 else int(t[5])))
                     for t in raw if len(t) >= 5 and int(t[4]) == track_id),
                    str(track_id),
                )
                duration = now - ts
                LOGGER.warning(f"[{self.camera_id}] ALARM {class_name} ID {track_id} visible > {self._alarm_seconds:.0f}s")
                self._ws_send({
                    "event": "alarm",
                    "camera_id": self.camera_id,
                    "track_id": track_id,
                    "class": class_name,
                    "duration": round(duration, 2),
                    "ts": now,
                })
                self._alarmed_ids.add(track_id)

        self._prev_active_ids = active_ids

        # Dim the area outside the ROI and draw a border so the active region is obvious
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            for region in [im[:ry1, :], im[ry2:, :], im[ry1:ry2, :rx1], im[ry1:ry2, rx2:]]:
                if region.size:
                    region[:] = (region * 0.35).astype(np.uint8)
            cv2.rectangle(im, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        alarm_count = sum(
            1 for tid, ts in self.track_first_seen.items()
            if tid in active_ids and (now - ts) >= ALARM_SECONDS
        )
        if alarm_count:
            banner = f"  [{self.camera_id}] ALARM: {alarm_count} object(s) > {self._alarm_seconds:.0f}s  "
            (bw, bh), bbl = cv2.getTextSize(banner, 0, 0.8, 2)
            bx = (im.shape[1] - bw) // 2
            cv2.rectangle(im, (bx - 6, 8), (bx + bw + 6, 8 + bh + bbl + 8), (0, 0, 200), -1)
            cv2.putText(im, banner, (bx, 8 + bh + 4), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(im, f"{self.cfg.label} [{self.camera_id}]", (im.shape[1] - 300, im.shape[0] - 10),
                    0, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if show_fps:
            self.fps_counter += 1
            if now - self.fps_timer >= 1.0:
                self.fps_display = self.fps_counter
                self.fps_counter = 0
                self.fps_timer = now
            fps_text = f"FPS: {self.fps_display}"
            (fw, fh), fbl = cv2.getTextSize(fps_text, 0, 0.7, 2)
            cv2.rectangle(im, (5, 5), (5 + fw + 10, 5 + fh + fbl + 6), (255, 255, 255), -1)
            cv2.putText(im, fps_text, (10, 5 + fh + 3), 0, 0.7, (104, 31, 17), 1, cv2.LINE_AA)

        LOGGER.info(f"[{self.camera_id}] No targets.") if not active_ids and self._prev_active_ids else None
        return im


# ---------------------------------------------------------------------------
# Main — start workers, display on main thread (required by most GUI backends)
# ---------------------------------------------------------------------------
workers: list[CameraWorker] = []
_workers_lock = threading.Lock()
yt_domains = ("youtube.com", "youtu.be")

# Registry used by the WebSocket listener to dispatch ROI commands.
_worker_registry: dict[int, CameraWorker] = {}


def _launch_worker(cfg: CameraConfig) -> CameraWorker | None:
    """Instantiate, register, and start a CameraWorker. Returns None on failure.

    Safe to call from any thread. Window creation is intentionally omitted here
    because cv2.namedWindow must run on the main thread — the main loop handles
    it the first time it sees a new worker.
    """
    with _workers_lock:
        if cfg.camera_id in _worker_registry:
            return None  # already running
        idx = len(workers)
    is_yt = isinstance(cfg.source, str) and any(d in cfg.source for d in yt_domains)
    try:
        w = CameraWorker(idx, cfg, loop=LOOP_YOUTUBE and is_yt)
    except SystemError as e:
        LOGGER.error(str(e))
        return None
    with _workers_lock:
        workers.append(w)
        _worker_registry[w.camera_id] = w
    w.start()
    return w


for cfg in fetch_cameras():
    _launch_worker(cfg)

if not workers:
    raise SystemError("No video sources could be opened.")


def _camera_poll_loop() -> None:
    """Background thread: periodically re-fetch the camera list and launch new workers."""
    if not CAMERA_POLL_INTERVAL:
        return
    while True:
        time.sleep(CAMERA_POLL_INTERVAL)
        try:
            for cfg in fetch_cameras():
                with _workers_lock:
                    already = cfg.camera_id in _worker_registry
                if not already:
                    w = _launch_worker(cfg)
                    if w:
                        LOGGER.info(f"[camera-poll] Started new worker: camera_id={cfg.camera_id} ({cfg.label})")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"[camera-poll] Error during poll: {exc}")


_poll_thread = threading.Thread(target=_camera_poll_loop, daemon=True, name="camera-poll")
_poll_thread.start()


# ---------------------------------------------------------------------------
# ROI coordinate helpers
# ---------------------------------------------------------------------------
def _roi_to_pixels(
    norm: list | tuple, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Convert normalized ROI [0.0–1.0] to pixel coordinates."""
    return (
        int(norm[0] * frame_w),
        int(norm[1] * frame_h),
        int(norm[2] * frame_w),
        int(norm[3] * frame_h),
    )


def _roi_to_norm(
    px: tuple | list, frame_w: int, frame_h: int
) -> list[float]:
    """Convert pixel ROI to normalized [0.0–1.0] coordinates (4 decimals)."""
    return [
        round(px[0] / frame_w, 4),
        round(px[1] / frame_h, 4),
        round(px[2] / frame_w, 4),
        round(px[3] / frame_h, 4),
    ]


# ---------------------------------------------------------------------------
# WebSocket hub — single persistent connection for both directions
# ---------------------------------------------------------------------------
# Outbound events are drained from _ws_send_queue and sent on the open socket.
# Inbound messages are dispatched as ROI commands to the appropriate worker.
# Supported inbound events:
#   {"event": "set_roi",   "camera_id": 1, "roi": [x1n, y1n, x2n, y2n]}  ← normalized 0.0–1.0
#   {"event": "clear_roi", "camera_id": 1}
async def _ws_hub() -> None:
    """Open one persistent connection; reconnects automatically on failure."""
    if not WS_URL:
        return
    try:
        import websockets
    except ImportError:
        LOGGER.warning("[ws-hub] websockets not installed — WebSocket disabled")
        return

    async def _sender(ws) -> None:
        while True:
            msg = await _ws_send_queue.get()
            await ws.send(msg)

    async def _receiver(ws) -> None:
        async for raw_msg in ws:
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue
            event = msg.get("event")
            cam_id = msg.get("camera_id")
            worker = _worker_registry.get(cam_id)
            if worker is None:
                if event in ("set_roi", "clear_roi"):
                    LOGGER.warning(
                        f"[ws-hub] {event}: camera_id={cam_id!r} not found in registry "
                        f"(known: {list(_worker_registry.keys())})"
                    )
                continue
            if event == "set_roi":
                LOGGER.info(f"[ws-hub] set_roi received: {json.dumps(msg)}")
                coords = msg.get("roi")
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    fs = worker.frame_size
                    if fs is None:
                        LOGGER.warning(
                            f"[ws-hub] set_roi for camera {cam_id} ignored — "
                            "frame size not yet known (no frame decoded)"
                        )
                        continue
                    # Guard: reject coords that are not genuinely normalized.
                    # [0, 0, 1, 1] as integers is a common default/uninitialized
                    # value — treat it as "clear ROI" rather than a valid selection.
                    coords_f = [float(v) for v in coords]
                    all_zero_one = all(v in (0.0, 1.0) for v in coords_f)
                    if all_zero_one and coords_f[0] == 0.0 and coords_f[1] == 0.0 \
                            and coords_f[2] == 1.0 and coords_f[3] == 1.0:
                        LOGGER.warning(
                            f"[ws-hub] set_roi cam {cam_id}: received [0,0,1,1] "
                            "(full-frame default) — treating as clear_roi. "
                            "Ensure the web UI sends genuinely normalized coordinates."
                        )
                        worker.set_roi(None)
                        fs2 = worker.frame_size
                        payload = json.dumps({
                            "event": "roi_updated",
                            "camera_id": cam_id,
                            "roi": None,
                            "frame_width": fs2[0] if fs2 else None,
                            "frame_height": fs2[1] if fs2 else None,
                            "ts": asyncio.get_event_loop().time(),
                        })
                        LOGGER.info(f"[ws-hub] roi_updated sent: {payload}")
                        await _ws_send_queue.put(payload)
                        continue
                    # Expect normalized coordinates [0.0–1.0]; convert to pixels.
                    roi = _roi_to_pixels(coords_f, fs[0], fs[1])
                    norm_back = _roi_to_norm(roi, fs[0], fs[1])
                    LOGGER.info(
                        f"[ws-hub] set_roi cam {cam_id}: norm={[round(v,3) for v in coords_f]} "
                        f"→ pixels={roi} (frame {fs[0]}x{fs[1]})"
                    )
                    worker.set_roi(roi)
                    payload = json.dumps({
                        "event": "roi_updated",
                        "camera_id": cam_id,
                        "roi": norm_back,
                        "frame_width": fs[0],
                        "frame_height": fs[1],
                        "ts": asyncio.get_event_loop().time(),
                    })
                    LOGGER.info(f"[ws-hub] roi_updated sent: {payload}")
                    await _ws_send_queue.put(payload)
            elif event == "clear_roi":
                worker.set_roi(None)
                fs = worker.frame_size
                await _ws_send_queue.put(json.dumps({
                    "event": "roi_updated",
                    "camera_id": cam_id,
                    "roi": None,
                    "frame_width": fs[0] if fs else None,
                    "frame_height": fs[1] if fs else None,
                    "ts": asyncio.get_event_loop().time(),
                }))

    while True:
        try:
            async with websockets.connect(WS_URL, open_timeout=5) as ws:
                LOGGER.info("[ws-hub] Connected")

                # Announce camera metadata so the UI knows each camera's native
                # resolution and can scale ROI coordinates correctly.
                camera_info = []
                for w in _worker_registry.values():
                    fs = w.frame_size
                    with w._roi_lock:
                        px_roi = w._roi
                    norm_roi = _roi_to_norm(px_roi, fs[0], fs[1]) if (px_roi and fs) else None
                    camera_info.append({
                        "camera_id": w.camera_id,
                        "label": w.cfg.label,
                        "frame_width": fs[0] if fs else None,
                        "frame_height": fs[1] if fs else None,
                        "roi": norm_roi,
                    })
                await ws.send(json.dumps({"event": "camera_info", "cameras": camera_info}))
                sender = asyncio.create_task(_sender(ws))
                receiver = asyncio.create_task(_receiver(ws))
                try:
                    done, pending = await asyncio.wait(
                        [sender, receiver], return_when=asyncio.FIRST_EXCEPTION
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        t.result()  # re-raise any exception
                except Exception:
                    sender.cancel()
                    receiver.cancel()
                    raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(f"[ws-hub] Disconnected ({exc}), reconnecting in 5 s...")
            await asyncio.sleep(5)


if WS_URL:
    asyncio.run_coroutine_threadsafe(_ws_hub(), _ws_loop)

# Initial window creation for workers that were started at launch.
_known_windows: set[str] = set()
if show_video:
    with _workers_lock:
        initial_workers = list(workers)
    for w in initial_workers:
        cv2.namedWindow(w.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w.window_name, 1280, 720)
        _known_windows.add(w.window_name)

LOGGER.info(f"Started {len(workers)} camera worker(s). Press 'q' to quit.")

try:
    while True:
        if show_video:
            with _workers_lock:
                current_workers = list(workers)
            for w in current_workers:
                # Create window the first time we see a dynamically added worker.
                if w.window_name not in _known_windows:
                    cv2.namedWindow(w.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(w.window_name, 1280, 720)
                    _known_windows.add(w.window_name)
                frame = w.get_frame()
                if frame is not None:
                    cv2.imshow(w.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            time.sleep(0.01)
finally:
    with _workers_lock:
        all_workers = list(workers)
    for w in all_workers:
        w.stop()
    if show_video:
        cv2.destroyAllWindows()

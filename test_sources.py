#!/usr/bin/env python3
"""test_sources.py — Probe each configured video source and report its framerate."""

import sys
import time

import cv2

from config import fetch_cameras

def probe_source(cfg) -> None:
    src = cfg.source
    label = cfg.label or str(cfg.camera_id)
    print(f"\n[{cfg.camera_id}] {label}")
    print(f"  Source : {src}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("  Status : ❌ Could not open source")
        return

    # Reported FPS from metadata
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Measure real FPS by reading N frames
    NUM_FRAMES = 30
    grabbed = 0
    start = time.time()
    for _ in range(NUM_FRAMES):
        ok, _ = cap.read()
        if not ok:
            break
        grabbed += 1
    elapsed = time.time() - start
    cap.release()

    if grabbed == 0:
        print("  Status : ❌ Opened but could not read frames")
        return

    real_fps = grabbed / elapsed if elapsed > 0 else 0.0
    resolution = f"{w}x{h}" if w > 0 and h > 0 else "unknown"

    print(f"  Status     : ✅ OK")
    print(f"  Resolution : {resolution}")
    print(f"  Meta FPS   : {meta_fps:.2f}")
    print(f"  Real FPS   : {real_fps:.2f}  ({grabbed}/{NUM_FRAMES} frames in {elapsed:.2f}s)")


def main() -> None:
    cameras = fetch_cameras()
    if not cameras:
        print("No cameras configured.")
        sys.exit(1)

    print(f"Probing {len(cameras)} source(s)...\n{'─' * 50}")
    for cfg in cameras:
        probe_source(cfg)
    print(f"\n{'─' * 50}\nDone.")


if __name__ == "__main__":
    main()

import argparse
import collections
import math
import time
import numpy as np

import torch
from ultralytics import YOLO
import cv2
import supervision as sv
import credentials

def create_tracker(tracker_type: str = "csrt"):
    # Try to create a tracker in a compatible way across OpenCV versions
    tracker = None
    try:
        if tracker_type.lower() == "csrt":
            if hasattr(cv2, "legacy"):
                tracker = cv2.legacy.TrackerCSRT_create()
            else:
                tracker = cv2.TrackerCSRT_create()
        elif tracker_type.lower() == "mosse":
            if hasattr(cv2, "legacy"):
                tracker = cv2.legacy.TrackerMOSSE_create()
            else:
                tracker = cv2.TrackerMOSSE_create()
    except Exception:
        # Fallback: try any available tracker factory
        for name in ("legacy",):
            if hasattr(cv2, name):
                module = getattr(cv2, name)
                if hasattr(module, "TrackerCSRT_create"):
                    tracker = module.TrackerCSRT_create()
                    break
    return tracker


def draw_text(frame, text, pos=(10, 20), color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-fps", type=float, default=2.0, help="Detections per second (default: 2)")
    parser.add_argument("--tracker", type=str, default="csrt", help="Tracker type to use between detections: csrt|mosse")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL override (optional)")
    parser.add_argument("--show-stats", action="store_true", help="Show device and FPS stats")
    parser.add_argument("--display-width", type=int, default=1280, help="Display window width in pixels (default: 1280)")
    parser.add_argument("--display-height", type=int, default=720, help="Display window height in pixels (default: 720)")
    parser.add_argument("--letterbox", action="store_true", help="Preserve aspect ratio by letterboxing to target display size")
    args = parser.parse_args()

    # Initialize YOLO model
    model = YOLO("yolov8x.pt")  # Load YOLOv8x model

    # RTSP stream URL - allow override
    if args.rtsp:
        rtsp_url = args.rtsp
    else:
        rtsp_url = f"rtsp://{credentials.username}:{credentials.password}@{credentials.cam_url}:{credentials.cam_port}/stream1"

    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.show_stats:
        print("Using device:", device)

    # Initialize annotation parameters
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # Inference timing
    INFERENCE_FPS = max(0.1, float(args.inference_fps))
    INFERENCE_INTERVAL = 1.0 / INFERENCE_FPS
    last_inference = 0.0

    # FPS stats
    shown_fps = 0.0
    display_fps = 0.0
    last_frame_time = time.time()
    inference_times = collections.deque(maxlen=20)

    # Trackers list: each item is dict {tracker, class_id, label, confidence}
    trackers = []
    tracker_type = args.tracker

    window_name = "RTSP Stream Object Detection"
    # Target display resolution (configurable via CLI)
    TARGET_WIDTH = int(args.display_width)
    TARGET_HEIGHT = int(args.display_height)
    LETTERBOX = bool(args.letterbox)

    # Create a resizable window and set its initial size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window_name, TARGET_WIDTH, TARGET_HEIGHT)
    except Exception:
        # Some backends may not support resizeWindow before showing; ignore
        pass

    def letterbox_frame(src, target_w, target_h):
        # Resize src to fit into target while preserving aspect ratio and pad with black bars
        h, w = src.shape[:2]
        if w == 0 or h == 0:
            return cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=src.dtype)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        now = time.time()
        # update display FPS (simple)
        dt = now - last_frame_time if last_frame_time else 0.0
        last_frame_time = now
        if dt > 0:
            display_fps = 0.9 * display_fps + 0.1 * (1.0 / dt) if display_fps else (1.0 / dt)

        # Run inference at configurable rate
        if (now - last_inference) >= INFERENCE_INTERVAL:
            t0 = time.time()
            results = model(frame)[0]
            t1 = time.time()
            inference_times.append(t1 - t0)
            # convert to detections
            detections = sv.Detections.from_ultralytics(results)

            # build labels and annotate (for reference)
            labels = [f"{model.names[cid]} {conf:0.2f}" for conf, cid in zip(detections.confidence, detections.class_id)]

            # Recreate trackers from detections
            trackers.clear()
            boxes = getattr(detections, "xyxy", None)
            if boxes is None:
                # fallback if different attr name
                boxes = detections.xyxy

            for box, label, conf, cid in zip(boxes, labels, detections.confidence, detections.class_id):
                x1, y1, x2, y2 = map(int, box)
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                bbox = (x1, y1, w, h)
                tr = create_tracker(tracker_type)
                if tr is not None:
                    try:
                        tr.init(frame, bbox)
                        trackers.append({"tracker": tr, "class_id": int(cid), "label": label, "confidence": float(conf)})
                    except Exception:
                        # ignore tracker init failures
                        pass

            last_inference = now

        # Update trackers and draw them onto a display frame
        display_frame = frame.copy()
        new_trackers = []
        for item in trackers:
            tr = item["tracker"]
            ok, bbox = tr.update(display_frame)
            if ok:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_text(display_frame, item["label"], pos=(x, y - 6))
                new_trackers.append(item)
        trackers = new_trackers

        # Draw FPS info (inference and display)
        inf_fps = 0.0
        if inference_times:
            avg_inf = sum(inference_times) / len(inference_times)
            if avg_inf > 0:
                inf_fps = 1.0 / avg_inf
        draw_text(display_frame, f"Inf FPS: {inf_fps:0.2f}", pos=(10, 80), color=(0, 200, 255))
        draw_text(display_frame, f"Disp FPS: {display_fps:0.1f}", pos=(10, 100), color=(0, 200, 255))
        if args.show_stats:
            draw_text(display_frame, f"Device: {device}", pos=(10, 140), color=(0, 255, 0), scale=0.5)

        # Resize to target display size before showing. Optionally preserve aspect ratio using letterbox.
        try:
            if LETTERBOX:
                display_resized = letterbox_frame(display_frame, TARGET_WIDTH, TARGET_HEIGHT)
            else:
                display_resized = cv2.resize(display_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        except Exception:
            display_resized = display_frame
        cv2.imshow(window_name, display_resized)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
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
    parser.add_argument("--undistort", action="store_true", help="Enable lens undistortion using calibration")
    parser.add_argument("--calib-file", type=str, default=None, help="Path to calibration file (.npz or .yml/.yaml/.xml). Should contain camera_matrix and dist_coeffs")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x (optional manual calib)")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y (optional manual calib)")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (optional manual calib)")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (optional manual calib)")
    parser.add_argument("--k1", type=float, default=0.0, help="Radial distortion k1 (manual)")
    parser.add_argument("--k2", type=float, default=0.0, help="Radial distortion k2 (manual)")
    parser.add_argument("--p1", type=float, default=0.0, help="Tangential distortion p1 (manual)")
    parser.add_argument("--p2", type=float, default=0.0, help="Tangential distortion p2 (manual)")
    parser.add_argument("--k3", type=float, default=0.0, help="Radial distortion k3 (manual)")
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

    # Undistort setup
    undistort_enabled = bool(args.undistort)
    camera_matrix = None
    dist_coeffs = None
    map1 = None
    map2 = None
    new_camera_matrix = None
    if undistort_enabled:
        # Load calibration from file if provided
        if args.calib_file:
            cf = args.calib_file
            try:
                if cf.endswith('.npz'):
                    data = np.load(cf)
                    camera_matrix = data.get('camera_matrix')
                    dist_coeffs = data.get('dist_coeffs')
                else:
                    fs = cv2.FileStorage(cf, cv2.FILE_STORAGE_READ)
                    camera_matrix = fs.getNode('camera_matrix').mat()
                    dist_coeffs = fs.getNode('dist_coeffs').mat()
                    fs.release()
            except Exception as e:
                print(f"Failed to load calib file {cf}: {e}")
        else:
            # try to build from manual params
            if args.fx and args.fy and args.cx and args.cy:
                camera_matrix = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=float)
                dist_coeffs = np.array([args.k1, args.k2, args.p1, args.p2, args.k3], dtype=float)
            else:
                print("Undistort enabled but no calibration provided. Disable --undistort or provide --calib-file or manual fx/fy/cx/cy.")
                undistort_enabled = False

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

        # Apply undistortion if requested. Initialize maps when we know frame size.
        if undistort_enabled:
            if camera_matrix is None or dist_coeffs is None:
                # no calibration available, skip
                undistort_enabled = False
            else:
                h, w = frame.shape[:2]
                if map1 is None or map2 is None or new_camera_matrix is None:
                    try:
                        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
                        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2)
                    except Exception as e:
                        print(f"Failed to init undistort maps: {e}")
                        undistort_enabled = False
                if undistort_enabled and map1 is not None and map2 is not None:
                    try:
                        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        pass

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

            # Annotate using supervision annotators for the inference frame
            try:
                annotated = frame.copy()
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                last_annotated_frame = annotated
            except Exception:
                # If supervision annotation fails, keep last_annotated_frame unchanged
                pass

            last_inference = now

        # Update trackers and render annotations using supervision so the look is consistent
        display_frame = frame.copy()
        new_trackers = []
        xyxy_list = []
        confs = []
        cids = []
        labels_for_annot = []

        for item in trackers:
            tr = item["tracker"]
            ok, bbox = tr.update(display_frame)
            if ok:
                x, y, w, h = map(int, bbox)
                x1, y1, x2, y2 = x, y, x + w, y + h
                xyxy_list.append([x1, y1, x2, y2])
                confs.append(item.get("confidence", 1.0))
                cids.append(item.get("class_id", 0))
                labels_for_annot.append(item.get("label", ""))
                new_trackers.append(item)
        trackers = new_trackers

        # If we have tracker boxes, convert to a supervision Detections object and annotate
        if len(xyxy_list) > 0:
            try:
                dets_from_trackers = sv.Detections(xyxy=np.array(xyxy_list), confidence=np.array(confs), class_id=np.array(cids))
                display_frame = box_annotator.annotate(scene=display_frame, detections=dets_from_trackers)
                display_frame = label_annotator.annotate(scene=display_frame, detections=dets_from_trackers, labels=labels_for_annot)
            except Exception:
                # fallback: draw simple rectangles/labels
                for xy, lab in zip(xyxy_list, labels_for_annot):
                    x1, y1, x2, y2 = map(int, xy)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_text(display_frame, lab, pos=(x1, y1 - 6))
        else:
            # If no trackers, optionally show last annotated inference frame
            if 'last_annotated_frame' in locals() and last_annotated_frame is not None:
                display_frame = last_annotated_frame.copy()

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

        # Resize to the current window size while forcing a 16:9 video area
        try:
            # Try to read the current window size (x,y,w,h)
            win_rect = None
            try:
                win_rect = cv2.getWindowImageRect(window_name)
            except Exception:
                # OpenCV build may not support getWindowImageRect
                win_rect = None

            if win_rect and len(win_rect) >= 4:
                _, _, win_w, win_h = win_rect
            else:
                win_w, win_h = TARGET_WIDTH, TARGET_HEIGHT

            # target video aspect is 16:9
            target_aspect = 16.0 / 9.0

            # compute the largest 16:9 rectangle that fits inside the window
            max_w = win_w
            max_h = int(max_w / target_aspect)
            if max_h > win_h:
                max_h = win_h
                max_w = int(max_h * target_aspect)

            # now fit the source frame into that rectangle preserving its own aspect ratio
            fh, fw = display_frame.shape[:2]
            if fw == 0 or fh == 0:
                display_resized = display_frame
            else:
                scale = min(max_w / fw, max_h / fh)
                new_w = max(1, int(fw * scale))
                new_h = max(1, int(fh * scale))
                resized = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # create black canvas of window size and center the resized frame inside the 16:9 area
                canvas = np.zeros((win_h, win_w, 3), dtype=display_frame.dtype)
                x_off = (win_w - new_w) // 2
                y_off = (win_h - new_h) // 2
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
                display_resized = canvas
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
from ultralytics import YOLO
import cv2
import supervision as sv
import time
import credentials

def main():
    # Initialize YOLO model
    model = YOLO("yolov8x.pt")  # Load YOLOv8x model

    # RTSP stream URL - modify this to your camera's URL
    rtsp_url = f"rtsp://{credentials.username}:{credentials.password}@{credentials.cam_url}:{credentials.cam_port}/stream1"

    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)
    
    # Initialize annotation parameters
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=1,
        text_scale=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        # Run YOLOv8 inference
        results = model(frame, device="cuda")[0]
        
        # Convert results to supervision detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Get labels for detected objects
        labels = [
            f"{model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id
            in zip(detections.confidence, detections.class_id)
        ]
        
        # Annotate frame with detections
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections
        )
        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        
        # Display the frame
        cv2.imshow("RTSP Stream Object Detection", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
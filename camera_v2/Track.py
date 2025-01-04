from ultralytics import YOLO
import os
import cv2

class ObjectTracking:
    def __init__(self):
        dir = os.path.dirname(__file__)
        weights_path = os.path.join(dir, "..", "assets", "yolo11x.pt")
        self.video_path = os.path.join(dir, "..", "assets", "test.mp4")
        self.video_path = os.path.join(dir, "..", "assets", "street_view.mp4")
        self.bytetrack_yaml_path = os.path.join(dir, "..", "assets", "bytetrack.yaml")
        self.model = YOLO(weights_path)

    
    def detect_objects(self):
        results = self.model.predict(source=self.video_path, show=True, line_width=1)

    def track_objects(self):
        frame_count = 0
        n_frames = 1
        image_scale = 1
        cap = cv2.VideoCapture(self.video_path)

        conf_threshold = 0.3
        iou_threshold = 0.3

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            new_height = int(height * image_scale)
            new_width = int(width * image_scale)
            frame = cv2.resize(frame, (new_width, new_height))

            frame_count += 1
            if frame_count % n_frames != 0:
                continue

            results = self.model.track(
                source=frame, 
                persist=True, 
                tracker=self.bytetrack_yaml_path, 
                classes=0,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, id, conf in zip(boxes, ids, confs):
                    color = (0, int(255 * conf), 0)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"Person {id} ({conf:.2f})", 
                              (box[0], box[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)

            cv2.imshow("Person Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def run_detect_object():
    ot = ObjectTracking()
    ot.detect_objects()

def run_track_object():
    ot = ObjectTracking()
    ot.track_objects()

if __name__ == "__main__":
    #run_detect_object()
    run_track_object()  








        

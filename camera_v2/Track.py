from ultralytics import YOLO
import cv2
import numpy as np
from config import *

class ObjectTracking:
    def __init__(self):
        self.model = YOLO(modele_path)
        self.video_proc = None
        self.display = None

    def set_components(self, video_processor, display_manager):
        self.video_proc = video_processor
        self.display = display_manager

    def track_objects(self):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        is_paused = False
        last_frame = None

        while True:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame.copy()
            else:
                frame = last_frame.copy()

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Détection et tracking avec ByteTrack
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=bytetrack_path,
                    classes=0,
                    conf=MIN_CONFIDENCE,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )

                # Affichage des détections
                if results and len(results) > 0 and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy()
                    
                    for box, id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        # Dessiner la box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Afficher l'ID
                        cv2.putText(frame, f"ID: {int(id)}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Affichage
            cv2.imshow("Tracking", frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                is_paused = not is_paused

        cap.release()
        cv2.destroyAllWindows()








        

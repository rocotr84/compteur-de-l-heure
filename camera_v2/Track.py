from ultralytics import YOLO
import cv2
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
        # Initialisation de la capture vidéo
        cap = self.video_proc.setup_video_capture(video_path)
        frame_count = 0
        
        while True:
            # Gestion de la pause via le DisplayManager
            if not self.display.is_paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Traitement de la frame
                frame = self.video_proc.process_frame(frame)
                
                frame_count += 1
                
                # Détection et tracking
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=bytetrack_path,
                    classes=0,
                    conf=MIN_CONFIDENCE,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )
                
                # Traitement des résultats
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for box, id, conf in zip(boxes, ids, confs):
                        # Vérification de la taille minimale
                        box_width = box[2] - box[0]
                        box_height = box[3] - box[1]
                        
                        if box_width >= MIN_WIDTH and box_height >= MIN_HEIGHT:
                            self.display.draw_detection(frame, box, id, conf)
                
                # Dessin de la ligne de comptage
                self.display.draw_crossing_line(frame, line_start, line_end)
            
            # Affichage et gestion des événements
            if self.display.show_frame(frame):
                break
        
        # Nettoyage
        cap.release()
        cv2.destroyAllWindows()

    def detect_objects(self):
        """
        Méthode de détection simple sans tracking
        """
        results = self.model.predict(source=video_path, show=True, line_width=1)








        

import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Configuration des chemins
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "..", "assets", "fixe_line.mp4")
modele_path = os.path.join(current_dir, "..", "assets", "yolo11x.pt")
mask_path = os.path.join(current_dir, "..", "assets", "fixe_line_mask.jpg")

# Paramètres
MAX_DISAPPEAR_FRAMES = 15
MAX_DISTANCE = 50
MIN_CONFIDENCE = 0.6
MIN_IOU_THRESHOLD = 0.3
output_width = 1280
output_height = 720
desired_fps = 30
line_start = (640, 720)
line_end = (1280, 360)

class TrackedPerson:
    def __init__(self, track_id, bbox, color=None):
        self.id = track_id
        self.bbox = bbox
        self.color = color
        self.disappeared = 0
        self.crossed_line = False
        self.center = self.calculate_center()
        self.trajectory = [self.center]
        self.last_confidence = 1.0
        
    def calculate_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def calculate_iou(self, bbox):
        x1 = max(self.bbox[0], bbox[0])
        y1 = max(self.bbox[1], bbox[1])
        x2 = min(self.bbox[2], bbox[2])
        y2 = min(self.bbox[3], bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def update(self, bbox, color=None, confidence=1.0):
        self.bbox = bbox
        if color:
            self.color = color
        self.disappeared = 0
        self.last_confidence = confidence
        self.center = self.calculate_center()
        self.trajectory.append(self.center)
        if len(self.trajectory) > 10:
            self.trajectory.pop(0)
            
    def check_line_crossing(self, line_start, line_end):
        if len(self.trajectory) < 2:
            return False
            
        prev_pos = self.trajectory[-2]
        curr_pos = self.trajectory[-1]
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            
        if intersect(prev_pos, curr_pos, line_start, line_end):
            if not self.crossed_line:
                self.crossed_line = True
                return True
        return False

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_persons = []
        self.counter = defaultdict(int)
        
    def get_distance(self, center1, center2):
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
    def update(self, detections, colors, confidences):
        # Marquer toutes les personnes comme non mises à jour
        for person in self.tracked_persons:
            person.disappeared += 1
            if person.last_confidence < 0.7:
                person.disappeared += 1
            
        # Mise à jour ou création de nouvelles personnes
        matched_indices = set()
        
        for bbox, color, conf in zip(detections, colors, confidences):
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            matched = False
            
            # Chercher la personne la plus proche
            min_dist = float('inf')
            best_match = None
            best_iou = 0
            
            for i, person in enumerate(self.tracked_persons):
                if i in matched_indices:
                    continue
                    
                dist = self.get_distance(center, person.center)
                iou = person.calculate_iou(bbox)
                
                # Utiliser à la fois la distance et l'IoU pour le matching
                if dist < MAX_DISTANCE and iou > MIN_IOU_THRESHOLD:
                    if dist < min_dist:
                        min_dist = dist
                        best_match = i
                        best_iou = iou
                    
            if best_match is not None:
                self.tracked_persons[best_match].update(bbox, color, conf)
                matched_indices.add(best_match)
                matched = True
                
            if not matched and conf > MIN_CONFIDENCE:
                self.tracked_persons.append(TrackedPerson(self.next_id, bbox, color))
                self.next_id += 1
                
        # Supprimer les personnes disparues
        self.tracked_persons = [p for p in self.tracked_persons 
                              if p.disappeared < MAX_DISAPPEAR_FRAMES]
        
        return self.tracked_persons

def main():
    # Initialisation
    model = YOLO(modele_path)
    tracker = PersonTracker()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la vidéo.")
        return
        
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (output_width, output_height))
        
        # Détection avec YOLO
        results = model(frame, stream=True)
        detections = []
        colors = []
        confidences = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                if conf < MIN_CONFIDENCE:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2))
                colors.append(None)
                confidences.append(conf)
                
        # Mise à jour du tracker avec les confiances
        tracked_persons = tracker.update(detections, colors, confidences)
        
        # Affichage
        for person in tracked_persons:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person.id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Dessiner la trajectoire
            for i in range(len(person.trajectory)-1):
                cv2.line(frame, person.trajectory[i], person.trajectory[i+1],
                        (0, 0, 255), 2)
                        
            if person.check_line_crossing(line_start, line_end):
                tracker.counter[person.color] += 1
                
        # Afficher la ligne et les compteurs
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
        y_offset = 30
        for color, count in tracker.counter.items():
            cv2.putText(frame, f"{color}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 30
            
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
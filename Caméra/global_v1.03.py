import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Configuration des chemins
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "..", "assets", "video.mp4")
modele_path = os.path.join(current_dir, "..", "assets", "yolo11x.pt")
mask_path = os.path.join(current_dir, "..", "assets", "fixe_line_mask1.jpg")

# Paramètres
MAX_DISAPPEAR_FRAMES = 15
MAX_DISTANCE = 50
MIN_CONFIDENCE = 0.60
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
        self.color_history = []  # Liste de tuples (couleur)
        
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
        
    def update_color(self, frame):
        x1, y1, x2, y2 = map(int, self.bbox)
        
        # Calcul de la région du t-shirt proportionnelle à la box
        roi_x1 = int(x1 + (x2 - x1) * 0.30)
        roi_x2 = int(x1 + (x2 - x1) * 0.70)
        roi_y1 = int(y1 + (y2 - y1) * 0.2)
        roi_y2 = int(y1 + (y2 - y1) * 0.4)

        if roi_x1 >= 0 and roi_y1 >= 0 and roi_x2 <= frame.shape[1] and roi_y2 <= frame.shape[0]:
            colors_to_detect = ["noir", "blanc", "rouge_fonce", "bleu_fonce", "bleu_clair", 
                              "vert_fonce", "rose", "jaune", "vert_clair"]
            color = detect_color_in_roi(frame, roi_x1, roi_y1, roi_x2, roi_y2, colors_to_detect)
            
            if color:
                self.color_history.append(color)
                # Garder un historique limité
                if len(self.color_history) > 5:
                    self.color_history.pop(0)
                # Mettre à jour la couleur si une couleur est détectée de manière consistante
                if len(self.color_history) >= 3:
                    # Prendre la couleur la plus fréquente des dernières détections
                    self.color = max(set(self.color_history), key=self.color_history.count)
    
    def update(self, bbox, frame, confidence=1.0):
        self.bbox = bbox
        self.disappeared = 0
        self.last_confidence = confidence
        self.center = self.calculate_center()
        self.trajectory.append(self.center)
        if len(self.trajectory) > 10:
            self.trajectory.pop(0)
        # Mettre à jour la couleur
        self.update_color(frame)
        
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
        
    def update(self, detections, frame, confidences):
        # Marquer toutes les personnes comme non mises à jour
        for person in self.tracked_persons:
            person.disappeared += 1
            if person.last_confidence < 0.7:
                person.disappeared += 1
            
        # Mise à jour ou création de nouvelles personnes
        matched_indices = set()
        
        for bbox, conf in zip(detections, confidences):
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
                
                if dist < MAX_DISTANCE and iou > MIN_IOU_THRESHOLD:
                    if dist < min_dist:
                        min_dist = dist
                        best_match = i
                        best_iou = iou
                    
            if best_match is not None:
                self.tracked_persons[best_match].update(bbox, frame, conf)
                matched_indices.add(best_match)
                matched = True
                
            if not matched and conf > MIN_CONFIDENCE:
                new_person = TrackedPerson(self.next_id, bbox)
                new_person.update_color(frame)  # Détection initiale de la couleur
                self.tracked_persons.append(new_person)
                self.next_id += 1
                
        # Supprimer les personnes disparues
        self.tracked_persons = [p for p in self.tracked_persons 
                              if p.disappeared < MAX_DISAPPEAR_FRAMES]
        
        return self.tracked_persons

def adjust_brightness(img, factor=1.2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_limits(color):
    colors = {
        "noir": ((0, 0, 0), (180, 255, 50)),
        "blanc": ((0, 0, 200), (180, 30, 255)),
        "rouge_fonce": ((0, 50, 50), (10, 255, 255)),
        "bleu_fonce": ((100, 50, 50), (130, 255, 120)),
        "bleu_clair": ((100, 50, 121), (130, 255, 255)),
        "vert_fonce": ((35, 50, 50), (85, 255, 255)),
        "rose": ((140, 50, 50), (170, 255, 255)),
        "jaune": ((20, 100, 100), (40, 255, 255)),
        "vert_clair": ((40, 50, 50), (80, 255, 255)),
    }
    
    color = color.lower()
    if color in colors:
        lower_limit, upper_limit = colors[color]
        return np.array(lower_limit, dtype=np.uint8), np.array(upper_limit, dtype=np.uint8)
    return None, None

def detect_color_in_roi(frame, x1, y1, x2, y2, color_list):
    # Ajuster la luminosité de la ROI
    roi_frame = frame[y1:y2, x1:x2]
    brightened_roi = adjust_brightness(roi_frame, 1.5)
    
    hsv_image = cv2.cvtColor(brightened_roi, cv2.COLOR_BGR2HSV)
    color_counts = {color: 0 for color in color_list}
    
    # Ajouter un seuil minimum de pixels pour la détection
    min_pixel_threshold = 100  # À ajuster selon vos besoins

    for color_name in color_list:
        lower_limit, upper_limit = get_limits(color_name)
        if lower_limit is not None and upper_limit is not None:
            mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
            color_counts[color_name] = cv2.countNonZero(mask)

    return max(color_counts, key=color_counts.get)

def main():
    # Initialisation
    model = YOLO(modele_path)
    tracker = PersonTracker()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la vidéo.")
        return
        
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    
    # Charger le masque
    try:
        mask = cv2.imread(mask_path, 0)  # Charger en niveaux de gris
        if mask is None:
            print(f"Impossible de charger le masque: {mask_path}")
            mask = np.ones((output_height, output_width), dtype=np.uint8) * 255
        else:
            # Redimensionner le masque pour correspondre à la taille de sortie
            mask = cv2.resize(mask, (output_width, output_height))
            print(f"Masque chargé avec succès. Taille: {mask.shape}")
    except Exception as e:
        print(f"Erreur lors du chargement du masque: {e}")
        mask = np.ones((output_height, output_width), dtype=np.uint8) * 255
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (output_width, output_height))
        
        # Vérifier que le masque a la même taille que la frame
        if mask.shape[:2] != frame.shape[:2]:
            print("Redimensionnement du masque pour correspondre à la frame")
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Appliquer le masque à la frame
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Détection avec YOLO
        results = model(frame, stream=True)
        detections = []
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
                confidences.append(conf)
                
        # Mise à jour du tracker avec la frame pour la détection de couleur
        tracked_persons = tracker.update(detections, frame, confidences)
        
        # Affichage
        for person in tracked_persons:
            x1, y1, x2, y2 = map(int, person.bbox)
            # Rectangle vert pour la détection de la personne
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calcul de la région du t-shirt proportionnelle à la box
            roi_x1 = int(x1 + (x2 - x1) * 0.30)
            roi_x2 = int(x1 + (x2 - x1) * 0.70)
            roi_y1 = int(y1 + (y2 - y1) * 0.2)
            roi_y2 = int(y1 + (y2 - y1) * 0.4)

            # Vérifier que le ROI est dans les limites de l'image
            if roi_x1 >= 0 and roi_y1 >= 0 and roi_x2 <= frame.shape[1] and roi_y2 <= frame.shape[0]:
                # Rectangle rouge pour la ROI de détection de couleur
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            
            # Afficher l'ID et la couleur détectée
            label = f"ID: {person.id}"
            if person.color:
                label += f" {person.color}"
            cv2.putText(frame, label, (x1, y1-10),
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
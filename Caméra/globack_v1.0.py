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
MAX_DISAPPEAR_FRAMES = 30
output_width = 1280
output_height = 720
desired_fps = 30
line_start = (640, 720)
line_end = (1280, 360)

def adjust_brightness(img, factor=1.2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_box_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

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

class Person:
    def __init__(self, pid, cx, cy, color=None):
        self.id = pid
        self.tracks = [(cx, cy)]
        self.done = False
        self.dir = None
        self.last_seen = 0
        self.color = color
        self.color_history = []  # Liste de tuples (position, couleur)
        if color:
            self.color_history.append((cx, cy, color))

    def update_coords(self, cx, cy, color=None):
        self.tracks.append((cx, cy))
        self.last_seen = 0
        if color:
            # Stocker la position et la couleur
            self.color_history.append((cx, cy, color))
            # Utiliser les 3 dernières détections pour stabiliser la couleur
            if len(self.color_history) >= 3:
                # Extraire seulement les couleurs des 3 derniers enregistrements
                recent_colors = [c[2] for c in self.color_history[-3:]]
                # Prendre la couleur la plus fréquente des 3 dernières détections
                self.color = max(set(recent_colors), key=recent_colors.count)

    def get_last_position(self):
        return self.tracks[-1]

    def check_crossing(self, line_start, line_end):
        if len(self.tracks) < 2:
            return False
        prev_x, prev_y = self.tracks[-2]
        curr_x, curr_y = self.tracks[-1]

        m = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
        c = line_start[1] - m * line_start[0]

        prev_pos = prev_y - (m * prev_x + c)
        curr_pos = curr_y - (m * curr_x + c)

        if prev_pos * curr_pos < 0:
            self.dir = "up" if prev_y > curr_y else "down"
            return True
        return False

def main():
    model = YOLO(modele_path)
    persons = []
    next_id = 1
    counter = defaultdict(int)
    
    # Charger le masque si le fichier existe
    mask = None
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print("Masque chargé avec succès")
    else:
        print("Aucun masque trouvé, le programme continuera sans masque")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la vidéo.")
        return

    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    colors_to_detect = ["noir", "blanc", "rouge_fonce", "bleu_fonce", "bleu_clair", 
                       "vert_fonce", "rose", "jaune", "vert_clair"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (output_width, output_height))
        
        # Appliquer le masque seulement s'il existe
        if mask is not None:
            # Redimensionner le masque si nécessaire
            if mask.shape[:2] != (output_height, output_width):
                mask = cv2.resize(mask, (output_width, output_height))
            working_frame = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            working_frame = frame.copy()

        # Utiliser working_frame pour toutes les opérations
        results = model(working_frame, stream=True)
        
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            class_ids = result.boxes.cls

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if int(cls_id) == 0:  # Classe "person"
                    x1, y1, x2, y2 = map(int, box)
                    center = get_box_center((x1, y1, x2, y2))
                    
                    # Calcul de la région du t-shirt proportionnelle à la box
                    roi_x1 = int(x1 + (x2 - x1) * 0.30)
                    roi_x2 = int(x1 + (x2 - x1) * 0.70)
                    roi_y1 = int(y1 + (y2 - y1) * 0.2)
                    roi_y2 = int(y1 + (y2 - y1) * 0.4)

                    if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > working_frame.shape[1] or roi_y2 > working_frame.shape[0]:
                        continue
                    
                    # Utiliser working_frame pour la détection de couleur
                    detected_color = detect_color_in_roi(working_frame, roi_x1, roi_y1, roi_x2, roi_y2, colors_to_detect)

                    new_person = True
                    current_id = next_id
                    for person in persons:
                        if not person.done and abs(center[0] - person.get_last_position()[0]) <= 50 and \
                           abs(center[1] - person.get_last_position()[1]) <= 50:
                            person.update_coords(*center, detected_color)
                            if person.check_crossing(line_start, line_end):
                                counter[person.color] += 1
                                person.done = True
                            new_person = False
                            current_id = person.id
                            break

                    if new_person:
                        persons.append(Person(next_id, *center, detected_color))
                        next_id += 1

                    # Dessiner sur working_frame
                    cv2.rectangle(working_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(working_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
                    cv2.putText(working_frame, f"ID{current_id}: {detected_color}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Mise à jour et nettoyage des personnes
        for person in persons:
            person.last_seen += 1
        persons = [p for p in persons if p.last_seen < MAX_DISAPPEAR_FRAMES]

        # Affichage sur working_frame
        cv2.line(working_frame, line_start, line_end, (0, 0, 255), 2)
        y_offset = 50
        for color, count in counter.items():
            cv2.putText(working_frame, f"{color}: {count}", (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 40

        cv2.imshow("Detection avec ligne diagonale", working_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Analyse finale des couleurs par personne
    print("\nAnalyse finale des couleurs par personne :")
    print("-" * 40)
    
    for person in persons:
        if person.color_history:  # Vérifier si l'historique n'est pas vide
            # Extraire toutes les couleurs de l'historique
            all_colors = [color for _, _, color in person.color_history]
            # Compter les occurrences de chaque couleur
            color_counts = {}
            for color in all_colors:
                color_counts[color] = color_counts.get(color, 0) + 1
            
            # Trouver la couleur la plus fréquente
            dominant_color = max(color_counts.items(), key=lambda x: x[1])
            percentage = (dominant_color[1] / len(all_colors)) * 100
            
            print(f"ID {person.id}:")
            print(f"  Couleur dominante: {dominant_color[0]}")
            print(f"  Pourcentage: {percentage:.1f}%")
            print(f"  Nombre total de détections: {len(all_colors)}")
            print(f"  Répartition des couleurs:")
            for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
                color_percentage = (count / len(all_colors)) * 100
                print(f"    - {color}: {count} ({color_percentage:.1f}%)")
            print("-" * 40)

if __name__ == "__main__":
    main()

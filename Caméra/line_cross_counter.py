# chemin_du_script.py

import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Récupérer le chemin du dossier contenant le script
current_dir = os.path.dirname(__file__)
# Construire le chemin vers la vidéo dans 'assets'
video_path = os.path.join(current_dir, "..", "assets", "test_video_2.mp4")

# Construire le chemin vers le modèle dans 'assets'
modele_path = os.path.join(current_dir, "..", "assets", "yolo11x.pt")

# Charger le modèle YOLO
model = YOLO(modele_path)

# Paramètres utilisateur
MAX_DISAPPEAR_FRAMES = 30  # Nombre de frames avant de supprimer une personne inactive
output_width = 1280
output_height = 720
desired_fps = 30
line_start = (640, 720)  # Milieu en bas
line_end = (1280, 360)  # Milieu sur le côté droit


# Fonction pour calculer le centre d'une boîte englobante
# Appel : center = get_box_center((x1, y1, x2, y2))
def get_box_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_limits(color):
    # HSV Values not RGB
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
        lower_limit = np.array(lower_limit, dtype=np.uint8)
        upper_limit = np.array(upper_limit, dtype=np.uint8)
        return lower_limit, upper_limit
    else:
        return None, None

def detect_color_in_roi(frame, x1, y1, x2, y2, color_list):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_frame = frame[y1:y2, x1:x2]

    color_counts = {color: 0 for color in color_list}

    for color_name in color_list:
        lower_limit, upper_limit = get_limits(color_name)
        if lower_limit is not None and upper_limit is not None:
            mask = cv2.inRange(hsv_image[y1:y2, x1:x2], lower_limit, upper_limit)
            color_counts[color_name] = cv2.countNonZero(mask)

    detected_color = max(color_counts, key=color_counts.get)
    return detected_color

# Classe pour suivre les personnes classées
class Person:
    def __init__(self, pid, cx, cy, color=None):
        self.id = pid
        self.tracks = [(cx, cy)]
        self.done = False
        self.dir = None
        self.last_seen = 0
        self.color = color
        self.color_history = []  # Pour suivre les changements de couleur

    # Appel : person.update_coords(cx, cy)
    def update_coords(self, cx, cy, color=None):
        self.tracks.append((cx, cy))
        self.last_seen = 0
        if color:
            self.color_history.append(color)
            if len(self.color_history) >= 3:  # Utiliser les 3 dernières détections pour stabiliser la couleur
                # Prendre la couleur la plus fréquente des 3 dernières détections
                self.color = max(set(self.color_history[-3:]), key=self.color_history[-3:].count)

    # Appel : last_position = person.get_last_position()
    def get_last_position(self):
        return self.tracks[-1]

    # Appel : crossed = person.check_crossing(line_start, line_end)
    def check_crossing(self, line_start, line_end):
        if len(self.tracks) < 2:
            return False
        prev_x, prev_y = self.tracks[-2]
        curr_x, curr_y = self.tracks[-1]

        m = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
        c = line_start[1] - m * line_start[0]

        prev_pos = prev_y - (m * prev_x + c)
        curr_pos = curr_y - (m * curr_x + c)

        if prev_pos * curr_pos < 0:  # Sign change means crossing
            self.dir = "up" if prev_y > curr_y else "down"
            return True
        return False

# Initialiser les variables
persons = []
next_id = 1
counter = defaultdict(int)  # Utiliser un defaultdict pour compter par couleur
frame_count = 0  # Pour suivre le nombre de frames

# Charger la vidéo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

cap.set(cv2.CAP_PROP_FPS, desired_fps)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    frame_count += 1  # Incrémenter le compteur de frames

    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    results = model(frame, stream=True)
    for result in results:
        # S'assurer que result.boxes n'est pas None
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf if result.boxes.conf is not None else []
        class_ids = result.boxes.cls if result.boxes.cls is not None else []

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if int(cls_id) == 0:  # Classe "person"
                x1, y1, x2, y2 = map(int, box)
                center = get_box_center((x1, y1, x2, y2))
                
                # Détecter la couleur dans la boîte englobante
                colors_to_detect = ["noir", "blanc", "rouge_fonce", "bleu_fonce", "bleu_clair", 
                                  "vert_fonce", "rose", "jaune", "vert_clair"]
                detected_color = detect_color_in_roi(frame, x1, y1, x2, y2, colors_to_detect)

                # Mise à jour ou ajout des personnes
                new_person = True
                for person in persons:
                    if not person.done and abs(center[0] - person.get_last_position()[0]) <= 50 and abs(center[1] - person.get_last_position()[1]) <= 50:
                        person.update_coords(*center, detected_color)
                        if person.check_crossing(line_start, line_end):
                            counter[person.color] += 1  # Incrémenter le compteur pour cette couleur
                            person.done = True
                        new_person = False
                        break

                if new_person:
                    persons.append(Person(next_id, *center, detected_color))
                    next_id += 1

    # Mettre à jour le compteur last_seen pour toutes les personnes
    for person in persons:
        person.last_seen += 1

    # Nettoyer les personnes qui n'ont pas été vues depuis longtemps
    persons = [p for p in persons if p.last_seen < MAX_DISAPPEAR_FRAMES]

    # Dessiner la ligne diagonale sur l'image
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Afficher le compteur de passages par couleur
    y_offset = 50
    for color, count in counter.items():
        cv2.putText(frame, f"{color}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 40

    # Afficher le flux vidéo
    cv2.imshow("Detection avec ligne diagonale", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

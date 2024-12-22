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

# Charger le modèle YOLOv8
model = YOLO(modele_path)

# Paramètres utilisateur
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

# Classe pour suivre les personnes classées
class Person:
    def __init__(self, pid, cx, cy):
        self.id = pid
        self.tracks = [(cx, cy)]
        self.done = False
        self.dir = None

    # Appel : person.update_coords(cx, cy)
    def update_coords(self, cx, cy):
        self.tracks.append((cx, cy))

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
counter = 0

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

                # Mise à jour ou ajout des personnes
                new_person = True
                for person in persons:
                    if not person.done and abs(center[0] - person.get_last_position()[0]) <= 50 and abs(center[1] - person.get_last_position()[1]) <= 50:
                        person.update_coords(*center)
                        if person.check_crossing(line_start, line_end):
                            counter += 1
                            person.done = True
                        new_person = False
                        break

                if new_person:
                    persons.append(Person(next_id, *center))
                    next_id += 1

                # Dessiner la boîte englobante principale (personne)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {person.id if not new_person else next_id - 1}"  # Afficher l'ID
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dessiner la ligne diagonale sur l'image
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Afficher le compteur de passages
    cv2.putText(frame, f"Compteur: {counter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Afficher le flux vidéo
    cv2.imshow("Detection avec ligne diagonale", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

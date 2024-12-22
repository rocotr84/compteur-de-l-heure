# chemin_du_script.py

import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from ultralytics import YOLO
import os



# Récupérer le chemin du dossier contenant le script
current_dir = os.path.dirname(__file__)
# Construire le chemin vers la vidéo dans 'assets'
video_path = os.path.join(current_dir, "..", "assets", "fixe_line.mp4")

# Construire le chemin vers le modèle dans 'assets'
modele_path = os.path.join(current_dir, "..", "assets", "yolo11n.pt")

# Charger le modèle YOLOv8
model = YOLO(modele_path)

# Paramètres utilisateur
output_width = 1280
output_height = 720
desired_fps = 30
line_position = 700
line_start = (0, line_position)
line_end = (1280, line_position)

# Charger les couleurs depuis un fichier JSON
def load_colors(filename=None):
    if filename is None:
        # Construire le chemin par défaut vers assets/colors.json
        current_dir = os.path.dirname(__file__)
        filename = os.path.join(current_dir, "..", "assets", "colors.json")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas.")
    
    with open(filename, 'r') as json_file:
        return json.load(json_file)

# Fonction pour calculer la distance euclidienne
def euclidean_distance(rgb1, rgb2):
    return math.sqrt((rgb1[0] - rgb2[0]) ** 2 + (rgb1[1] - rgb2[1]) ** 2 + (rgb1[2] - rgb2[2]) ** 2)

# Fonction pour trouver la couleur la plus proche (en tenant compte du format BGR d'OpenCV)
def find_closest_color(input_bgr, color_list):
    # Convertir la couleur BGR en RGB
    input_rgb = (input_bgr[2], input_bgr[1], input_bgr[0])
    
    closest_color = None
    min_distance = float('inf')
    for color in color_list:
        distance = euclidean_distance(input_rgb, color["rgb"])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

# Fonction pour détecter la couleur dominante
def get_dominant_color(roi):
    roi_small = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_AREA)
    roi_small = roi_small.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(roi_small)
    dominant_color = kmeans.cluster_centers_[0]
    return tuple(map(int, dominant_color))

# Vérifie si une boîte croise la ligne fixe
def is_box_crossing_line(box, line_y):
    _, y1, _, y2 = box
    return y1 <= line_y <= y2

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

cap.set(cv2.CAP_PROP_FPS, desired_fps)
color_list = load_colors()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if int(cls_id) == 0:  # Classe "person"
                x1, y1, x2, y2 = map(int, box)

                # Vérification si la boîte croise la ligne
                if is_box_crossing_line((x1, y1, x2, y2), line_position):
                    cv2.putText(frame, "Passage detecte", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Calcul de la région du t-shirt
                roi_x1 = int(x1 + (x2 - x1) * 0.30)
                roi_x2 = int(x1 + (x2 - x1) * 0.70)
                roi_y1 = int(y1 + (y2 - y1) * 0.2)
                roi_y2 = int(y1 + (y2 - y1) * 0.4)

                if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > frame.shape[1] or roi_y2 > frame.shape[0]:
                    continue

                # Extraire le ROI du t-shirt
                roi_teeshirt = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

                # Obtenir la couleur dominante du t-shirt
                if roi_teeshirt.size > 0:
                    dominant_color = get_dominant_color(roi_teeshirt)
                    closest_color = find_closest_color(dominant_color, color_list)

                    if closest_color:
                        color_name = closest_color["name"]
                        cv2.putText(frame, color_name, (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Dessiner la boîte englobante principale (personne)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dessiner la ligne fixe sur l'image
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
    cv2.imshow("Detection avec ligne fixe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# chemin_du_script.py

import json
import cv2
import numpy as np
import math
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO("yolo11n.pt")

# Base des plages HSV
color_ranges = {
    "Red": ([0, 100, 100], [10, 255, 255]),        # Rouge (Teinte basse)
    "Red2": ([160, 100, 100], [180, 255, 255]),    # Rouge (Teinte haute)
    "Green": ([40, 50, 50], [80, 255, 255]),       # Vert
    "Blue": ([100, 50, 50], [140, 255, 255]),      # Bleu
    "Yellow": ([20, 100, 100], [30, 255, 255]),    # Jaune
    "Cyan": ([85, 50, 50], [95, 255, 255]),        # Cyan
    "Magenta": ([145, 50, 50], [155, 255, 255]),   # Magenta
    "White": ([0, 0, 200], [180, 30, 255]),        # Blanc
    "Black": ([0, 0, 0], [180, 255, 30])           # Noir
}

# Paramètres utilisateur
output_width = 1280
output_height = 720
desired_fps = 10
line_position = 700
line_start = (0, line_position)
line_end = (1280, line_position)

# Charger les couleurs depuis un fichier JSON
def load_colors(filename='colors.json'):
    with open(filename, 'r') as json_file:
        return json.load(json_file)

# Fonction pour calculer la distance euclidienne
def euclidean_distance(rgb1, rgb2):
    return math.sqrt((rgb1[0] - rgb2[0]) ** 2 + (rgb1[1] - rgb2[1]) ** 2 + (rgb1[2] - rgb2[2]) ** 2)

# Fonction pour trouver la couleur la plus proche
def find_closest_color(input_rgb, color_list):
    closest_color = None
    min_distance = float('inf')
    for color in color_list:
        distance = euclidean_distance(input_rgb, color["rgb"])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

# Fonction pour détecter la couleur dominante avec un histogramme# Fonction pour trouver la couleur prédominante dans toute la zone ROI avec des masques HSV
def get_dominant_color_hsv(roi, color_ranges):
    """
    Detecte la couleur prédominante dans un ROI en utilisant les masques HSV sur toute la zone.
    Args:
        roi: L'image ou la région d'intérêt (ROI) en BGR.
        color_ranges: Dictionnaire des plages HSV pour chaque couleur.
    Returns:
        Le nom de la couleur dominante ou None.
    """
    # Convertir le ROI en HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    max_pixels = 0
    dominant_color_name = None

    # Parcourir les plages de couleurs HSV définies
    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        # Appliquer le masque HSV sur l'ensemble du ROI
        mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)

        # Compter les pixels correspondant à la couleur
        pixel_count = cv2.countNonZero(mask)

        # Garder la couleur avec le plus grand nombre de pixels
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            dominant_color_name = color_name
            

    return dominant_color_name


# Vérifie si une boîte croise la ligne fixe
def is_box_crossing_line(box, line_y):
    _, y1, _, y2 = box
    return y1 <= line_y <= y2

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

cap.set(cv2.CAP_PROP_FPS, desired_fps)
color_list = load_colors('colors.json')

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

                # Calcul de la région du t-shirt
                roi_x1 = int(x1 + (x2 - x1) * 0.30)
                roi_x2 = int(x1 + (x2 - x1) * 0.70)
                roi_y1 = int(y1 + (y2 - y1) * 0.2)
                roi_y2 = int(y1 + (y2 - y1) * 0.4)

                if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > frame.shape[1] or roi_y2 > frame.shape[0]:
                    continue

                # Extraire le ROI du t-shirt
                roi_teeshirt = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                # Convertir le ROI en espace HSV
                if roi_teeshirt.size > 0:
                    hsv_roi = cv2.cvtColor(roi_teeshirt, cv2.COLOR_BGR2HSV)

                    # Afficher uniquement le ROI en HSV
                    cv2.imshow("ROI en HSV", hsv_roi)

    # Sortir de la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



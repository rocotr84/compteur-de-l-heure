import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from math import sqrt
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO("yolo11n.pt")

# Paramètres utilisateur
output_width = 1280
output_height = 720
desired_fps = 30
line_position = 700
line_start = (0, line_position)
line_end = (1280, line_position)

# Charger les couleurs depuis un fichier JSON
def load_colors_from_json(filename='colors_hsv.json'):
    with open(filename, 'r') as json_file:
        return json.load(json_file)

# Fonction pour vérifier si une couleur est dans une plage HSV donnée
def is_color_in_range(hsv_color, hsv_range):
    h, s, v = hsv_color
    h_min, s_min, v_min, h_max, s_max, v_max = hsv_range
    return h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max

# Fonction pour détecter la couleur dominante dans un ROI
def get_dominant_color(roi_hsv):
    roi_small = cv2.resize(roi_hsv, (50, 50), interpolation=cv2.INTER_AREA)
    roi_small = roi_small.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(roi_small)
    dominant_color = kmeans.cluster_centers_[0]
    return tuple(map(int, dominant_color))

# Vérifie si une boîte croise la ligne fixe
def is_box_crossing_line(box, line_y):
    _, y1, _, y2 = box
    return y1 <= line_y <= y2

# Charger la vidéo
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erreur : Impossible d'ouvrir la vidéo '{video_path}'.")
    exit()

cap.set(cv2.CAP_PROP_FPS, desired_fps)
color_list = load_colors_from_json('colors.json')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur de lecture.")
        break

    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Convertir l'image en HSV directement ici
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

                # Extraire le ROI du t-shirt dans l'espace HSV
                roi_teeshirt_hsv = frame_hsv[roi_y1:roi_y2, roi_x1:roi_x2]
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

                # Obtenir la couleur dominante du t-shirt en HSV
                if roi_teeshirt_hsv.size > 0:
                    dominant_color = get_dominant_color(roi_teeshirt_hsv)
                    
                    # Vérifier si la couleur dominante correspond à une couleur dans le fichier JSON
                    color_name = None
                    for color in color_list:
                        hsv_range = color["hsv_range"]  # Plage HSV
                        if is_color_in_range(dominant_color, hsv_range):
                            color_name = color["name"]
                            break

                    if color_name:
                        print(f"Couleur dominante détectée: {color_name}")
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

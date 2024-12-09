import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from ultralytics import YOLO

#couleur la plus proche v1, ligne, fps, taille image

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Modèle léger pour la détection

# Paramètres utilisateur
output_width = 1280  # Largeur de la sortie
output_height = 720  # Hauteur de la sortie
desired_fps = 30     # Nombre de FPS désiré

# Définir la ligne fixe (exemple : ligne horizontale au centre de l'image)
line_position = 700  # Position verticale de la ligne (pixels)
line_start = (0, line_position)  # Début de la ligne (x=0)
line_end = (1280, line_position)  # Fin de la ligne (x=800, largeur de l'image)

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

# Fonction pour détecter la couleur dominante
def get_dominant_color(roi):
    # Redimensionner pour accélérer le clustering
    roi_small = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_AREA)
    roi_small = roi_small.reshape((-1, 3))
    
    # Utiliser KMeans pour trouver la couleur dominante
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(roi_small)
    dominant_color = kmeans.cluster_centers_[0]
    return tuple(map(int, dominant_color))

# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

# Appliquer le FPS désiré
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Charger la liste des couleurs
color_list = load_colors('colors.json')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    # Redimensionner le frame à la taille de sortie
    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Détection des objets dans le frame
    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes.xyxy  # Coordonnées des boîtes (xmin, ymin, xmax, ymax)
        confidences = result.boxes.conf  # Confiances
        class_ids = result.boxes.cls  # IDs des classes

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if int(cls_id) == 0:  # Classe "person"
                x1, y1, x2, y2 = map(int, box)
                
                # Calcul de la région du t-shirt
                roi_x1 = int(x1 + (x2 - x1) * 0.30)  # Décalage horizontal de 15%
                roi_x2 = int(x1 + (x2 - x1) * 0.70)  # Décalage horizontal de 85%
                roi_y1 = int(y1 + (y2 - y1) * 0.2)   # Décalage vertical de 40%
                roi_y2 = int(y1 + (y2 - y1) * 0.4)   # Décalage vertical de 70%

                # Vérification des limites
                if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > frame.shape[1] or roi_y2 > frame.shape[0]:
                    continue
                
                # Extraire le ROI du t-shirt
                roi_teeshirt = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Dessiner le rectangle pour visualiser le ROI
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)  # Rectangle bleu pour le t-shirt

                # Obtenir la couleur dominante du t-shirt
                if roi_teeshirt.size > 0:
                    dominant_color = get_dominant_color(roi_teeshirt)
                    closest_color = find_closest_color(dominant_color, color_list)
                    
                    if closest_color:
                        color_name = closest_color["name"]
                        # Afficher le nom de la couleur dominante
                        cv2.putText(frame, color_name, (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Dessiner la boîte englobante principale (personne)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dessiner la ligne fixe sur l'image
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Afficher le flux vidéo avec les annotations
    cv2.imshow("Detection avec ligne fixe", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

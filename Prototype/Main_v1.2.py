import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from ultralytics import YOLO
#couleur la plus proche et ligne
# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Modèle léger pour la détection

# Définir la ligne fixe (exemple : ligne horizontale au centre de l'image)
line_position = 450  # Position verticale de la ligne (pixels)
line_start = (0, line_position)  # Début de la ligne (x=0)
line_end = (800, line_position)  # Fin de la ligne (x=800, largeur de l'image)

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
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

# Charger la liste des couleurs
color_list = load_colors('colors.json')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    # Détection des objets dans le frame
    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes.xyxy  # Coordonnées des boîtes (xmin, ymin, xmax, ymax)
        confidences = result.boxes.conf  # Confiances
        class_ids = result.boxes.cls  # IDs des classes

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            # Filtrer uniquement les personnes (classe "person", ID=0 dans COCO)
            if int(cls_id) == 0:  # Classe "person"
                x1, y1, x2, y2 = map(int, box)
                
                # Extraire la région d'intérêt (ROI)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue  # Éviter les erreurs si la boîte dépasse les limites

                # Obtenir la couleur dominante
                dominant_color = get_dominant_color(roi)
                closest_color = find_closest_color(dominant_color, color_list)
                
                if closest_color:
                    color_name = closest_color["name"]  # Nom de la couleur trouvée
                
                # Dessiner la boîte englobante
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Afficher le nom de la couleur dominante
                cv2.putText(frame, color_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Vérifier si la boîte croise la ligne
                box_center_y = (y1 + y2) // 2  # Centre vertical de la boîte
                if y1 < line_position < y2:  # La ligne est dans les limites de la boîte
                    cv2.putText(frame, "PASSAGE DETECTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"Passage détecté ! {x1, y1, x2, y2}")

    # Dessiner la ligne fixe sur l'image
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Afficher le flux vidéo avec les annotations
    cv2.imshow("Detection avec ligne fixe", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

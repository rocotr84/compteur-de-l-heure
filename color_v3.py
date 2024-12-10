import cv2
import numpy as np
from math import sqrt
from ultralytics import YOLO

# Fonction pour trouver la couleur la plus proche avec un seuil de tolérance dans l'espace HSV
def closest_color_in_hsv(hsv_color, color_range):
    """Cette fonction détecte la couleur la plus proche dans l'espace HSV"""
    hue, saturation, value = hsv_color
    min_diff = float('inf')
    closest_color = None

    for color_name, (h_min, h_max, s_min, s_max, v_min, v_max) in color_range.items():
        # Vérifier si la couleur se trouve dans la plage HSV définie
        if h_min <= hue <= h_max and s_min <= saturation <= s_max and v_min <= value <= v_max:
            closest_color = color_name
            break

    return closest_color

# Définir les plages de couleurs en HSV pour chaque couleur d'intérêt
color_range = {
    'Red': (0, 10, 100, 255, 100, 255),
    'Green': (35, 85, 50, 255, 50, 255),
    'Blue': (90, 130, 50, 255, 50, 255),
    'Yellow': (20, 40, 100, 255, 100, 255),
    # Ajouter d'autres couleurs si nécessaire
}

# Charger le modèle YOLOv8 pour la détection des personnes
model = YOLO("yolo11n.pt")

# Charger la vidéo
video_path = "test_tee-shirt.mp4"  # Remplacez par le chemin de votre vidéo
cap = cv2.VideoCapture(video_path)

# Vérifiez si la vidéo a été ouverte correctement
if not cap.isOpened():
    print(f"Erreur : Impossible d'ouvrir la vidéo '{video_path}'.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur de lecture.")
        break

    # Appliquer YOLO pour détecter les personnes dans l'image
    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes.xyxy  # Coordonnées des boîtes
        confidences = result.boxes.conf  # Confidences des boîtes
        class_ids = result.boxes.cls  # Classes des boîtes

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if int(cls_id) == 0:  # Classe "person" (identifiant 0 pour personne)
                x1, y1, x2, y2 = map(int, box)

                # Définir la zone d'intérêt (ROI) autour du t-shirt
                roi_x1 = int(x1 + (x2 - x1) * 0.30)  # 30% à partir de la gauche
                roi_x2 = int(x1 + (x2 - x1) * 0.70)  # 70% à partir de la gauche
                roi_y1 = int(y1 + (y2 - y1) * 0.2)   # 20% à partir du haut
                roi_y2 = int(y1 + (y2 - y1) * 0.4)   # 40% à partir du haut

                # Vérifier si la ROI est valide dans l'image
                if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > frame.shape[1] or roi_y2 > frame.shape[0]:
                    continue

                # Extraire la ROI du t-shirt
                roi_teeshirt = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                # Convertir la ROI en HSV et calculer la couleur dominante
                hsv_roi = cv2.cvtColor(roi_teeshirt, cv2.COLOR_BGR2HSV)
                mean_hue = np.mean(hsv_roi[:, :, 0])  # Teinte (Hue)
                mean_saturation = np.mean(hsv_roi[:, :, 1])  # Saturation
                mean_value = np.mean(hsv_roi[:, :, 2])  # Valeur

                # Déterminer la couleur dominante
                dominant_color = closest_color_in_hsv((mean_hue, mean_saturation, mean_value), color_range)

                # Afficher la couleur dominante sur la vidéo
                cv2.putText(frame, f"Couleur: {dominant_color}", (roi_x1, roi_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Dessiner la boîte englobante autour de la personne
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher la vidéo avec la couleur dominante détectée
    cv2.imshow("Vidéo avec détection de couleur", frame)

    # Appuyez sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture de la vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

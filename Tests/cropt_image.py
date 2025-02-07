import os
import cv2
from ultralytics import YOLO

# Définition des chemins
current_dir = os.path.dirname(__file__)
modele_path = os.path.join(current_dir, "..", "assets", "yolo11x.pt")
base_folder = r'C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\camera4K2'

# Charger le modèle YOLOv8
model = YOLO(modele_path)

def detect_and_crop(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger {image_path}")
        return
    
    # Détection des objets avec YOLO
    results = model(image)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordonnées [x_min, y_min, x_max, y_max]
        confidences = result.boxes.conf.cpu().numpy()  # Scores de confiance
        classes = result.boxes.cls.cpu().numpy()  # Classes détectées

        # Filtrer les détections pour ne garder que les personnes (classe 0 pour YOLO)
        persons = [boxes[i] for i in range(len(classes)) if classes[i] == 0]

        if not persons:
            print(f"Aucune personne détectée dans {image_path}")
            return
        
        # Sélectionner la personne la plus grande (probablement la plus proche)
        x_min, y_min, x_max, y_max = max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        # Calcul du centre de la personne
        person_center_x = (x_min + x_max) / 2
        img_center_x = image.shape[1] / 2

        # Si la personne est trop à gauche ou droite, rogner pour la centrer
        if person_center_x < img_center_x - 50:  # Trop à gauche
            new_x_min = max(0, int(person_center_x - (x_max - x_min) / 2))
            new_x_max = min(image.shape[1], new_x_min + (x_max - x_min))
        elif person_center_x > img_center_x + 50:  # Trop à droite
            new_x_max = min(image.shape[1], int(person_center_x + (x_max - x_min) / 2))
            new_x_min = max(0, new_x_max - (x_max - x_min))
        else:
            new_x_min, new_x_max = 0, image.shape[1]  # Garder l'image entière

        # Recadrage de l'image
        cropped_image = image[:, int(new_x_min):int(new_x_max)]


        # Écraser l'image originale
        cv2.imwrite(image_path, cropped_image)
        print(f"Image mise à jour : {image_path}")

# Scanner récursivement chaque sous-dossier et analyser les images
for root, _, files in os.walk(base_folder):
    for file_name in files:
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            detect_and_crop(os.path.join(root, file_name))

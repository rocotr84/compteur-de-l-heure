import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import colors as mcolors
import os

# Récupérer le chemin du dossier contenant le script
current_dir = os.path.dirname(__file__)
# Construire le chemin vers la vidéo dans 'assets'
video_path = os.path.join(current_dir, "..", "assets", "video.mp4")

# Construire le chemin vers le modèle dans 'assets'
modele_path = os.path.join(current_dir, "..", "assets", "yolo11n.pt")

# Liste des couleurs prédéfinies en HSV
COLOR_NAMES = ["Rouge", "Vert", "Bleu", "Jaune", "Orange", "Violet", "Cyan", "Rose", "Gris", "Noir"]
COLOR_RANGES_HSV = {
    "noir": ((0, 0, 0), (180, 255, 50)),
    "blanc": ((0, 0, 200), (180, 30, 255)),
    "rouge_fonce": ((0, 50, 50), (10, 255, 255)),
    "bleu_fonce": ((100, 50, 50), (130, 255, 100)),  # Teinte basse, Valeur faible
    "bleu_clair": ((100, 50, 101), (130, 255, 255)),  # Teinte identique, Valeur élevée
    "vert_fonce": ((35, 50, 50), (85, 255, 255)),
    "rose": ((140, 50, 50), (170, 255, 255)),
    "jaune": ((20, 100, 100), (40, 255, 255)),
    "vert_clair": ((40, 50, 50), (80, 255, 255)),
}

# Fonction pour trouver la couleur la plus proche en HSV
def closest_color_hsv(hsv_value):
    closest_color = None
    min_distance = float("inf")
    for color_name, (lower, upper) in COLOR_RANGES_HSV.items():
        # Calculer la distance euclidienne entre la couleur détectée et la plage HSV
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        distance = np.linalg.norm(hsv_value - (lower_bound + upper_bound) / 2)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    return closest_color

# Fonction pour extraire la couleur dominante dans une zone donnée en utilisant HSV
def dominant_color_hsv(image, x, y, w, h):
    # Extraire la zone d'intérêt (ROI)
    roi = image[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculer la couleur dominante en utilisant la moyenne des pixels
    mean_hsv = np.mean(hsv_roi, axis=(0, 1))
    
    return mean_hsv

# Programme principal
def main():
    # Paramètres de la zone fixe (x, y, largeur, hauteur) - à ajuster
    x, y, w, h = 250, 600, 50, 50  # Exemple de zone

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)  # Remplacez par 0 pour la webcam

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo ou erreur.")
            break

        # Trouver la couleur dominante dans la zone
        mean_hsv = dominant_color_hsv(frame, x, y, w, h)
        
        # Trouver la couleur la plus proche dans les plages HSV prédéfinies
        closest_color = closest_color_hsv(mean_hsv)

        # Afficher la zone de détection
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Couleur: {closest_color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Afficher la vidéo
        cv2.imshow("Video - Detection Couleur", frame)

        # Quitter avec 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

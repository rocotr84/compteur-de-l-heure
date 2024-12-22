import cv2
import numpy as np
import os


# Récupérer le chemin du dossier contenant le script
current_dir = os.path.dirname(__file__)
# Construire le chemin vers la vidéo dans 'assets'
video_path = os.path.join(current_dir, "..", "assets", "video.mp4")

# Construire le chemin vers le modèle dans 'assets'
modele_path = os.path.join(current_dir, "..", "assets", "yolo11n.pt")


# Définir les plages HSV pour chaque couleur
color_ranges = {
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

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convertir la frame en espace de couleurs HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Appliquer les masques pour chaque couleur
    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        
        # Créer un masque pour chaque plage de couleur
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        # Appliquer le masque à l'image d'origine pour afficher la couleur détectée
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Afficher le résultat
        cv2.imshow(f'{color_name}', result)

    # Afficher la vidéo originale
    cv2.imshow('Vidéo originale', frame)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()


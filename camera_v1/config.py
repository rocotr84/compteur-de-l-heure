import os

# Configuration des chemins d'accès aux ressources
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "..", "assets", "street_view.mp4")
#video_path = os.path.join(current_dir, "..", "assets", "man_alone.mp4")
#video_path = os.path.join(current_dir, "..", "assets", "video.mp4")
modele_path = os.path.join(current_dir, "..", "assets", "yolo11x.pt")
mask_path = os.path.join(current_dir, "..", "assets", "fixe_line_mask1.jpg")


# Configuration des paramètres globaux
MAX_DISAPPEAR_FRAMES = 15    # Nombre maximum de frames avant de considérer une personne comme disparue
MAX_DISTANCE = 50           # Distance maximale pour l'association des détections
MIN_CONFIDENCE = 0.60      # Seuil minimal de confiance pour les détections
MIN_IOU_THRESHOLD = 0.3    # Seuil minimal pour l'IoU (Intersection over Union)

# Paramètres d'affichage
output_width = 1280        # Largeur de sortie de la vidéo
output_height = 720        # Hauteur de sortie de la vidéo
desired_fps = 30          # Images par seconde souhaitées
frame_delay = 10           # Délai entre les frames en millisecondes (1 = temps réel, augmentez pour ralentir)

# Points définissant la ligne de comptage
line_start = (640, 720)   # Point de début de la ligne
line_end = (1280, 360)    # Point de fin de la ligne 
import os

# Configuration des chemins d'accès aux ressources
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "..", "assets", "p3.mp4")
modele_path = os.path.join(current_dir, "..", "assets", "yolo11x.pt")
bytetrack_path = os.path.join(current_dir, "..", "assets", "bytetrack.yaml")
mask_path = os.path.join(current_dir, "..", "assets", "fixe_line_mask1.jpg")

# Configuration des paramètres de détection
MIN_CONFIDENCE = 0.3
IOU_THRESHOLD = 0.3
MIN_WIDTH = 50
MIN_HEIGHT = 50

# Paramètres d'affichage
output_width = 1280
output_height = 720
desired_fps = 30
frame_delay = 10

# Points définissant la ligne de comptage
line_start = (640, 720)
line_end = (1280, 360)

# Ajouter ces paramètres
COUNTING_ENABLED = True
# Zone de comptage (pourcentage de la hauteur de l'image)
COUNTING_ZONE_TOP = 0.4    # 40% depuis le haut
COUNTING_ZONE_BOTTOM = 0.6 # 60% depuis le haut 

# Paramètre pour le traitement des frames
PROCESS_EVERY_N_FRAMES = 1  # Traite une frame toutes les N frames 
import os

# Configuration des chemins d'accès aux ressources
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "..", "assets", "street_view.mp4")
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
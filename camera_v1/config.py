import os

# Configuration des chemins d'accès aux ressources
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "..", "assets", "test.mp4")
#video_path = os.path.join(current_dir, "..", "assets", "man_alone.mp4")
#video_path = os.path.join(current_dir, "..", "assets", "video.mp4")
modele_path = os.path.join(current_dir, "..", "assets", "yolo11x.pt")
mask_path = os.path.join(current_dir, "..", "assets", "p3_mask1.png")


# Configuration des paramètres globaux
MAX_DISAPPEAR_FRAMES = 15    # Nombre maximum de frames avant de considérer une personne comme disparue
MAX_DISTANCE = 70           # Distance maximale pour l'association des détections
MIN_CONFIDENCE = 0.50      # Seuil minimal de confiance pour les détections
MIN_IOU_THRESHOLD = 0.3    # Seuil minimal pour l'IoU (Intersection over Union)

# Paramètres d'affichage
output_width = 1280        # Largeur de sortie de la vidéo
output_height = 720        # Hauteur de sortie de la vidéo
desired_fps = 60          # Images par seconde souhaitées
frame_delay = 1           # Délai entre les frames en millisecondes (1 = temps réel, augmentez pour ralentir)

# Points définissant la ligne de comptage
line_start = (0, output_height - 50)  # Point de début de la ligne (bord gauche, 50px du bas)
line_end = (output_width, output_height - 50)  # Point de fin de la ligne (bord droit, 50px du bas)

# Ajouter ces nouvelles constantes
PROCESS_EVERY_N_FRAMES = 1  # Traiter une frame sur deux pour optimiser les performances
IOU_THRESHOLD = 0.3        # Seuil IOU pour ByteTrack
bytetrack_path = "bytetrack.yaml"  # Chemin vers la config ByteTrack q

# Options d'affichage
SHOW_TRAJECTORIES = False  # Option pour activer/désactiver l'affichage des trajectoires

# Options d'enregistrement vidéo
SAVE_VIDEO = False  # Option pour activer l'enregistrement au lieu de l'affichage
VIDEO_OUTPUT_PATH = "output.mp4"  # Chemin du fichier de sortie
VIDEO_FPS = 60  # FPS pour la vidéo de sortie
VIDEO_CODEC = 'mp4v'  # Codec vidéo (peut aussi être 'XVID' pour .avi)
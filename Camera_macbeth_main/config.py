import os

# Configuration des chemins d'accès aux ressources
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "..", "assets", "video", "p3_macbeth.mp4")
#video_path = os.path.join(current_dir, "..", "assets", "photos","Macbeth", "IMG_3673.JPG")
#video_path = os.path.join(current_dir, "..", "assets", "man_alone.mp4")
#video_path = os.path.join(current_dir, "..", "assets", "video.mp4")
modele_path = os.path.join(current_dir, "..", "assets", "models", "yolo11n.pt")
mask_path = os.path.join(current_dir, "..", "assets", "mask", "p3_macbeth_mask.jpg")
output_file = os.path.join(current_dir, "..", "Camera_macbeth_main", "detections.csv")
detection_tracker = None
cache_file = r"D:\Windsuft programme\compteur-de-l-heure\Camera_macbeth_main\macbeth_cache.json"
detect_squares = False

# Configuration des paramètres globaux
MAX_DISAPPEAR_FRAMES = 15    # Nombre maximum de frames avant de considérer une personne comme disparue
MAX_DISTANCE = 70           # Distance maximale pour l'association des détections
MIN_CONFIDENCE = 0.50      # Seuil minimal de confiance pour les détections
MIN_IOU_THRESHOLD = 0.3    # Seuil minimal pour l'IoU (Intersection over Union)

# Paramètres de pondération des couleurs
MIN_TIME_BETWEEN_PASSES = 50.0  # Temps minimal entre deux passages de la même couleur (en secondes)
PENALTY_DURATION = 50.0         # Durée de la pénalité de pondération (en secondes)
MIN_COLOR_WEIGHT = 0.1         # Poids minimal pour une couleur (même si elle vient de passer)
MIN_PIXEL_RATIO = 0.15         # Poids minimal pour une couleur (même si elle vient de passer)
MIN_PIXEL_COUNT = 100         # Poids minimal pour une couleur (même si elle vient de passer)
COLOR_HISTORY_SIZE = 2         # Poids minimal pour une couleur (même si elle vient de passer)

# Paramètres d'affichage
output_width = 1280        # Largeur de sortie de la vidéo
output_height = 720        # Hauteur de sortie de la vidéo
desired_fps = 60          # Images par seconde souhaitées
frame_delay = 1           # Délai entre les frames en millisecondes (1 = temps réel, augmentez pour ralentir)

# Points définissant la ligne de comptage
line_start = (0, output_height - 10)  # Point de début de la ligne (bord gauche, 10px du bas)
line_end = (output_width, output_height - 10)  # Point de fin de la ligne (bord droit, 10px du bas)

# Ajouter ces nouvelles constantes
PROCESS_EVERY_N_FRAMES = 1  # Traiter une frame sur deux pour optimiser les performances
IOU_THRESHOLD = 0.3        # Seuil IOU pour ByteTrack
bytetrack_path = "bytetrack.yaml"  # Chemin vers la config ByteTrack q

# Options d'affichage
SHOW_TRAJECTORIES = False  # Option pour activer/désactiver l'affichage des trajectoires

# Options d'enregistrement vidéo
SAVE_VIDEO = True  # Option pour activer l'enregistrement au lieu de l'affichage
VIDEO_OUTPUT_PATH = "output.mp4"  # Chemin du fichier de sortie
VIDEO_FPS = 15  # FPS pour la vidéo de sortie
VIDEO_CODEC = 'mp4v'  # Codec vidéo (peut aussi être 'XVID' pour .avi)

# Configuration du mode de détection
DETECTION_MODE = "color"  # "color" ou "number"

# Configuration de la détection des couleurs
COLOR_MIN_PIXEL_RATIO = 0.15
COLOR_MIN_PIXEL_COUNT = 100
COLOR_HISTORY_SIZE = 2

# Configuration de la détection des numéros
NUMBER_MIN_CONFIDENCE = 0.6
NUMBER_ROI_SCALE = 0.4  # Taille relative de la ROI par rapport à la bbox
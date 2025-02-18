import os
from collections import defaultdict
import numpy as np

# Définition du répertoire courant
current_dir = os.path.dirname(__file__)

# Chemins des ressources
VIDEO_INPUT_PATH = os.path.join(current_dir, "..", "assets", "video", "p3_macbeth.mp4")
MODEL_PATH = os.path.join(current_dir, "..", "assets", "models", "yolo11n.pt")
DETECTION_MASK_PATH = os.path.join(current_dir, "..", "assets", "mask", "p3_macbeth_mask.jpg")
CSV_OUTPUT_PATH = os.path.join(current_dir, "..", "Camera_macbeth_main", "detections.csv")
CACHE_FILE_PATH = os.path.join(current_dir, "macbeth_cache.json")

# Configuration du système
DETECT_SQUARES = False
DETECTION_MODE = "color"  # "color" ou "number"

# Paramètres d'initialisation
detection_tracker = None

# Configuration des paramètres globaux
MAX_DISAPPEAR_FRAMES = 15    # Nombre de frames avant de considérer une personne disparue
MAX_TRACKING_DISTANCE = 70   # Distance maximale pour suivre une même personne
MIN_DETECTION_CONFIDENCE = 0.50  # Seuil de confiance pour valider une détection
MIN_IOU_THRESHOLD = 0.3    # Seuil minimal de chevauchement entre détections

# Paramètres de pondération des couleurs
MIN_TIME_BETWEEN_PASSES = 50.0  # Temps minimal entre deux passages de la même couleur (en secondes)
PENALTY_DURATION = 50.0         # Durée de la pénalité de pondération (en secondes)
MIN_COLOR_WEIGHT = 0.1         # Poids minimal pour une couleur (même si elle vient de passer)
MIN_PIXEL_RATIO = 0.15         # Poids minimal pour une couleur (même si elle vient de passer)
MIN_PIXEL_COUNT = 100         # Poids minimal pour une couleur (même si elle vient de passer)
COLOR_HISTORY_SIZE = 2        # Nombre de détections conservées dans l'historique

# Paramètres d'affichage
output_width = 1280        # Largeur de sortie de la vidéo
output_height = 720        # Hauteur de sortie de la vidéo
desired_fps = 30          # Images par seconde souhaitées
frame_delay = 1           # Délai entre les frames en millisecondes (1 = temps réel, augmentez pour ralentir)

# Points définissant la ligne de comptage
line_start = (0, output_height - 10)  # Point de début de la ligne (bord gauche, 10px du bas)
line_end = (output_width, output_height - 10)  # Point de fin de la ligne (bord droit, 10px du bas)


IOU_THRESHOLD = 0.5        # Seuil IOU pour ByteTrack
bytetrack_path = "bytetrack.yaml"  # Chemin vers la config ByteTrack q

# Options d'affichage
SHOW_TRAJECTORIES = False  # Option pour activer/désactiver l'affichage des trajectoires

# Options d'enregistrement vidéo
SAVE_VIDEO = True  # Option pour activer l'enregistrement au lieu de l'affichage
VIDEO_OUTPUT_PATH = "output.mp4"  # Chemin du fichier de sortie
VIDEO_FPS = 30  # FPS pour la vidéo de sortie
VIDEO_CODEC = 'mp4v'  # Codec vidéo (peut aussi être 'XVID' pour .avi)

# Configuration de la détection des couleurs
COLOR_MIN_PIXEL_RATIO = 0.15
COLOR_MIN_PIXEL_COUNT = 100
COLOR_HISTORY_SIZE = 2

# Renommer pour plus de cohérence
VIDEO_OUTPUT_WRITER = None
CSV_OUTPUT_FILE = None

DETECTION_HISTORY = defaultdict(list)

# Paramètres de détection et de suivi
MIN_CONFIDENCE = 0.5
MIN_NUMBER_CONFIDENCE = 0.4

# Paramètres de détection des numéros
CONTRAST_CLIP_LIMIT = 2.0
CONTRAST_GRID_SIZE = (8, 8)
BINARY_BLOCK_SIZE = 11
BINARY_CONSTANT = 2
MORPHOLOGY_KERNEL_SIZE = (3, 3)
ROI_EXPANSION_RATIO = 0.2


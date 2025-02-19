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
MAX_DISAPPEAR_FRAMES = 30    # Nombre de frames avant de considérer une personne disparue
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
SHOW_ROI_AND_COLOR = False    # Désactive l'affichage du ROI et de la couleur détectée
SHOW_TRAJECTORIES = False     # Affichage des trajectoires (renommer pour cohérence)
SHOW_CENTER = False           # Affichage du centre
SHOW_LABELS = True           # Affichage des labels (ID, etc.)

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

# Paramètres d'optimisation de la correction des couleurs
COLOR_CORRECTION_INTERVAL = 60  # Effectue la correction toutes les 30 frames

# Couleurs de référence de la charte Macbeth en BGR
MACBETH_REFERENCE_COLORS = np.array([
        [68, 82, 115], [130, 150, 194], [157, 122, 98], [67, 108, 87], [177, 128, 133],
        [170, 189, 103], [44, 126, 214], [166, 91, 80], [99, 90, 193], [108, 60, 94],
        [64, 188, 157], [46, 163, 224], [150, 61, 56], [73, 148, 70], [60, 54, 175],
        [31, 199, 231], [149, 86, 187], [161, 133, 8], [242, 243, 243], [200, 200, 200],
        [160, 160, 160], [121, 122, 122], [85, 85, 85], [52, 52, 52]
    ], dtype=np.float32)

# Configuration des plages de couleurs HSV
COLOR_RANGES = {
    'noir': ((0, 0, 0), (180, 255, 50)),
    'blanc': ((0, 0, 200), (180, 30, 255)),
    'rouge_fonce': ((0, 50, 50), (10, 255, 255)),
    'rouge2': ((160, 50, 50), (180, 255, 255)),
    'bleu_fonce': ((100, 50, 50), (130, 255, 120)),
    'bleu_clair': ((100, 50, 121), (130, 255, 255)),
    'vert_fonce': ((35, 50, 50), (85, 255, 255)),
    'rose': ((140, 50, 50), (170, 255, 255)),
    'jaune': ((20, 100, 100), (40, 255, 255)),
    'vert_clair': ((40, 50, 50), (80, 255, 255))
}

# Variable globale pour stocker les masques pré-calculés
COLOR_MASKS = {}


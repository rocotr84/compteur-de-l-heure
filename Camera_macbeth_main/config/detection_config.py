"""
Configuration des paramètres de détection et de suivi.

Ce module définit les seuils et paramètres utilisés pour:
- La détection des personnes
- Le suivi des trajectoires
- La détection des franchissements de ligne
"""

# Configuration des paramètres globaux
MAX_DISAPPEAR_FRAMES = 30    # Nombre de frames avant de considérer une personne disparue
MAX_TRACKING_DISTANCE = 70   # Distance maximale pour suivre une même personne
MIN_DETECTION_CONFIDENCE = 0.50  # Seuil de confiance pour valider une détection
IOU_THRESHOLD = 0.5     # Seuil minimal de chevauchement entre détections

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
COLOR_CORRECTION_INTERVAL = 300  # Effectue la correction toutes les 30 frames

# Configuration du système
DETECT_SQUARES = False
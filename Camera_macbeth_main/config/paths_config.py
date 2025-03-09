"""
Configuration des chemins de fichiers pour l'application.

Ce module centralise tous les chemins d'accès aux ressources externes:
- Vidéos d'entrée
- Modèles
- Masques
- Fichiers de sortie
"""

import os

# Définition des répertoires principaux
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
DB_DIR = os.path.join(DATA_DIR, "db")


# Chemins des ressources
VIDEO_INPUT_PATH = os.path.join(ASSETS_DIR, "video", "p3_macbeth.mp4")
MODEL_PATH = os.path.join(ASSETS_DIR, "models", "yolo11n.pt")
DETECTION_MASK_PATH = os.path.join(ASSETS_DIR, "mask", "p3_macbeth_mask.jpg")
BYTETRACK_PATH = os.path.join(ASSETS_DIR, "models", "bytetrack.yaml")

# Chemins des fichiers de sortie
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "detections.csv")
SQL_DB_PATH = os.path.join(DB_DIR, "detections.db")
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.mp4")

# Chemins des fichiers de cache
CACHE_FILE_PATH = os.path.join(CACHE_DIR, "macbeth_cache.json")
WARPED_IMAGE_PATH = os.path.join(CACHE_DIR, "macbeth_cache_warped.png")
WARPED_WITH_SQUARES_PATH = os.path.join(CACHE_DIR, "macbeth_cache_warped_with_squares.png")
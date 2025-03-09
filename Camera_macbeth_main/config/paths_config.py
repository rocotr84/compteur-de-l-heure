"""
Configuration des chemins de fichiers pour l'application.

Ce module centralise tous les chemins d'accès aux ressources externes:
- Vidéos d'entrée
- Modèles
- Masques
- Fichiers de sortie
"""

import os

# Définition du répertoire courant
current_dir = os.path.dirname(os.path.dirname(__file__))

# Chemins des ressources
VIDEO_INPUT_PATH = os.path.join(current_dir, "..", "assets", "video", "p3_macbeth.mp4")
MODEL_PATH = os.path.join(current_dir, "..", "assets", "models", "yolo11n.pt")
DETECTION_MASK_PATH = os.path.join(current_dir, "..", "assets", "mask", "p3_macbeth_mask.jpg")
CSV_OUTPUT_PATH = os.path.join(current_dir, "..", "Camera_macbeth_main", "detections.csv")
CACHE_FILE_PATH = os.path.join(current_dir, "macbeth_cache.json")
SQL_DB_PATH = os.path.join(current_dir, "..", "DDB", "detections.db")
BYTETRACK_PATH = os.path.join(current_dir, "..","assets", "bytetrack.yaml")
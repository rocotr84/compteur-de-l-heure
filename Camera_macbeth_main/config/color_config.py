"""
Configuration des paramètres de couleur.

Ce module définit:
- Les plages de couleurs HSV pour la détection
- Les couleurs de référence de la charte Macbeth
- Les paramètres de pondération des couleurs
"""

import numpy as np

# Paramètres de pondération des couleurs
MIN_TIME_BETWEEN_PASSES = 50.0  # Temps minimal entre deux passages de la même couleur (en secondes)
PENALTY_DURATION = 50.0         # Durée de la pénalité de pondération (en secondes)
MIN_COLOR_WEIGHT = 0.1         # Poids minimal pour une couleur (même si elle vient de passer)
MIN_PIXEL_RATIO = 0.15         # Ratio minimal de pixels pour considérer une couleur
MIN_PIXEL_COUNT = 100         # Nombre minimal de pixels pour considérer une couleur
COLOR_HISTORY_SIZE = 2        # Nombre de détections conservées dans l'historique

# Configuration de la détection des couleurs
COLOR_MIN_PIXEL_RATIO = 0.15
COLOR_MIN_PIXEL_COUNT = 100
COLOR_HISTORY_SIZE = 2

# Paramètres d'optimisation de la correction des couleurs
COLOR_CORRECTION_INTERVAL = 300  # Effectue la correction toutes les 300 frames

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
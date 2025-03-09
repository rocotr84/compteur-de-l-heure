"""
Configuration des paramètres d'affichage.

Ce module définit les paramètres liés à:
- L'affichage à l'écran
- L'enregistrement vidéo
- Les options de visualisation
"""

# Paramètres d'affichage
output_width = 1280        # Largeur de sortie de la vidéo
output_height = 720        # Hauteur de sortie de la vidéo
desired_fps = 30          # Images par seconde souhaitées

# Points définissant la ligne de comptage
line_start = (0, output_height - 10)  # Point de début de la ligne (bord gauche, 10px du bas)
line_end = (output_width, output_height - 10)  # Point de fin de la ligne (bord droit, 10px du bas)

# Options d'affichage
SHOW_ROI_AND_COLOR = False    # Désactive l'affichage du ROI et de la couleur détectée
SHOW_TRAJECTORIES = False     # Affichage des trajectoires
SHOW_CENTER = False           # Affichage du centre
SHOW_LABELS = True           # Affichage des labels (ID, etc.)

# Options d'enregistrement vidéo
SAVE_VIDEO = False  # Option pour activer l'enregistrement au lieu de l'affichage
VIDEO_OUTPUT_PATH = "output.mp4"  # Chemin du fichier de sortie
VIDEO_FPS = 30  # FPS pour la vidéo de sortie
VIDEO_CODEC = 'mp4v'  # Codec vidéo (peut aussi être 'XVID' pour .avi)
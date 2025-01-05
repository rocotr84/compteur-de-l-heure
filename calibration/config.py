import os

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "assets", "calibration_samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Chemin vers la vidéo
VIDEO_SOURCE = os.path.join(PROJECT_ROOT, "assets", "video.mp4")  # Ajustez le nom du fichier selon votre vidéo

# Vérification de l'existence du fichier vidéo au démarrage
if not os.path.exists(VIDEO_SOURCE):
    print(f"ATTENTION: Le fichier vidéo {VIDEO_SOURCE} n'existe pas!")

# Couleurs à calibrer
COLORS = {
    "rouge_fonce": {
        "name": "Rouge foncé",
        "key": "r",
        "display_color": (0, 0, 180)  # BGR
    },
    "bleu_fonce": {
        "name": "Bleu foncé",
        "key": "b",
        "display_color": (180, 0, 0)
    },
    "bleu_clair": {
        "name": "Bleu clair",
        "key": "l",  # l pour 'light blue'
        "display_color": (255, 128, 0)
    },
    "vert_fonce": {
        "name": "Vert foncé",
        "key": "g",  # g pour 'green'
        "display_color": (0, 180, 0)
    },
    "vert_clair": {
        "name": "Vert clair",
        "key": "v",
        "display_color": (0, 255, 0)
    },
    "rose": {
        "name": "Rose",
        "key": "p",  # p pour 'pink'
        "display_color": (180, 105, 255)
    },
    "jaune": {
        "name": "Jaune",
        "key": "j",
        "display_color": (0, 255, 255)
    },
    "blanc": {
        "name": "Blanc",
        "key": "w",  # w pour 'white'
        "display_color": (255, 255, 255)
    },
    "noir": {
        "name": "Noir",
        "key": "n",
        "display_color": (0, 0, 0)
    }
}

# Paramètres de l'interface
WINDOW_NAME = "Calibration des couleurs"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720 
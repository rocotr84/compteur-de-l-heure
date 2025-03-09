# Guide de configuration

## Structure des configurations

Le système utilise une approche modulaire pour la configuration, avec des fichiers thématiques dans le dossier `config/`:

- `paths_config.py`: Chemins des fichiers
- `detection_config.py`: Paramètres de détection
- `display_config.py`: Options d'affichage
- `color_config.py`: Configuration des couleurs
- `storage_config.py`: Options de stockage

## Configuration des chemins (paths_config.py)

Ce fichier définit tous les chemins utilisés par l'application:

```python
# Chemins des ressources
VIDEO_INPUT_PATH = "chemin/vers/video.mp4"  # Vidéo d'entrée
MODEL_PATH = "chemin/vers/modele.pt"        # Modèle YOLO
DETECTION_MASK_PATH = "chemin/vers/masque.jpg"  # Masque de détection
CSV_OUTPUT_PATH = "chemin/vers/detections.csv"  # Fichier CSV de sortie
CACHE_FILE_PATH = "chemin/vers/cache.json"      # Cache Macbeth
SQL_DB_PATH = "chemin/vers/detections.db"       # Base SQLite
BYTETRACK_PATH = "chemin/vers/bytetrack.yaml"   # Config ByteTrack
```

## Configuration de la détection (detection_config.py)

Paramètres liés à la détection et au suivi des personnes:

```python
# Paramètres de suivi
MAX_DISAPPEAR_FRAMES = 30    # Frames avant de considérer une personne disparue
MAX_TRACKING_DISTANCE = 70   # Distance max pour suivre une personne
MIN_DETECTION_CONFIDENCE = 0.50  # Seuil de confiance pour la détection
IOU_THRESHOLD = 0.5     # Seuil de chevauchement

# Correction des couleurs
COLOR_CORRECTION_INTERVAL = 300  # Frames entre les corrections
DETECT_SQUARES = False  # Détecter les carrés Macbeth à chaque fois
```

## Configuration de l'affichage (display_config.py)

Options liées à l'affichage et à la visualisation:

```python
# Dimensions de sortie
output_width = 1280
output_height = 720
desired_fps = 30

# Ligne de comptage
line_start = (0, output_height - 10)
line_end = (output_width, output_height - 10)

# Options d'affichage
SHOW_ROI_AND_COLOR = False  # Afficher ROI et couleur
SHOW_TRAJECTORIES = False   # Afficher trajectoires
SHOW_CENTER = False         # Afficher centre
SHOW_LABELS = True          # Afficher labels

# Enregistrement vidéo
SAVE_VIDEO = False          # Enregistrer la vidéo
VIDEO_OUTPUT_PATH = "output.mp4"
VIDEO_FPS = 30
VIDEO_CODEC = 'mp4v'
```

## Configuration des couleurs (color_config.py)

Paramètres liés à la détection et au traitement des couleurs:

```python
# Pondération des couleurs
MIN_TIME_BETWEEN_PASSES = 50.0  # Temps min entre passages (secondes)
MIN_COLOR_WEIGHT = 0.1         # Poids min pour une couleur
MIN_PIXEL_RATIO = 0.15         # Ratio min de pixels
MIN_PIXEL_COUNT = 100          # Nombre min de pixels
COLOR_HISTORY_SIZE = 2         # Taille de l'historique

# Plages de couleurs HSV
COLOR_RANGES = {
    'noir': ((0, 0, 0), (180, 255, 50)),
    'blanc': ((0, 0, 200), (180, 30, 255)),
    # ... autres couleurs ...
}
```

## Configuration du stockage (storage_config.py)

Options liées au stockage des données:

```python
# Mode de stockage
SAVE_SQL = False  # Si True, utilise SQLite au lieu de CSV
```

## Modification des configurations

Pour modifier les configurations, vous pouvez:

1. Éditer directement les fichiers de configuration
2. Modifier les variables en cours d'exécution:

```python
from config.display_config import SHOW_TRAJECTORIES
SHOW_TRAJECTORIES = True
```

## Bonnes pratiques

- Créez des profils de configuration pour différents scénarios
- Documentez les modifications importantes
- Testez après chaque changement significatif
- Utilisez des valeurs par défaut raisonnables

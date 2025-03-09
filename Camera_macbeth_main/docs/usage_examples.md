# Exemples d'utilisation

## Démarrage simple

Le moyen le plus simple d'utiliser le système est de créer une instance de la classe `Application` et d'appeler sa méthode `run()`:

```python
from application import Application

# Création et démarrage de l'application
app = Application()
app.run()
```

## Configuration personnalisée

Vous pouvez modifier les paramètres de configuration avant de démarrer l'application:

```python
from application import Application
from config.display_config import SHOW_TRAJECTORIES, SHOW_CENTER
from config.detection_config import MIN_DETECTION_CONFIDENCE

# Modification des paramètres de configuration
SHOW_TRAJECTORIES = True  # Afficher les trajectoires
SHOW_CENTER = True        # Afficher les centres
MIN_DETECTION_CONFIDENCE = 0.6  # Augmenter le seuil de confiance

# Création et démarrage de l'application
app = Application()
app.run()
```

## Traitement d'une vidéo spécifique

Pour traiter une vidéo spécifique, modifiez le chemin dans `paths_config.py`:

```python
from application import Application
from config.paths_config import VIDEO_INPUT_PATH

# Spécification d'une vidéo personnalisée
VIDEO_INPUT_PATH = "chemin/vers/ma_video.mp4"

# Création et démarrage de l'application
app = Application()
app.run()
```

## Enregistrement des détections en SQLite

Pour activer l'enregistrement en SQLite au lieu de CSV:

```python
from application import Application
from config.storage_config import SAVE_SQL

# Activation de l'enregistrement SQLite
SAVE_SQL = True

# Création et démarrage de l'application
app = Application()
app.run()
```

## Utilisation avancée: traitement frame par frame

Si vous avez besoin d'un contrôle plus fin sur le traitement:

```python
import cv2
from application import Application

# Création de l'application
app = Application()
app.initialize()

# Capture vidéo personnalisée
cap = cv2.VideoCapture(0)  # Utiliser la webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Traitement de la frame
    processed_frame, should_exit, _ = app.process_frame_with_tracking(frame)

    # Affichage personnalisé
    cv2.imshow("Résultat", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or should_exit:
        break

# Nettoyage
cap.release()
app.cleanup()
cv2.destroyAllWindows()
```

## Intégration avec d'autres systèmes

Vous pouvez facilement intégrer ce système avec d'autres applications:

```python
from application import Application
import time

# Création de l'application
app = Application()
app.initialize()

# Fonction pour traiter une image et obtenir les résultats
def process_image(image):
    processed_frame, _, _ = app.process_frame_with_tracking(image)
    return {
        'processed_image': processed_frame,
        'counters': app.tracker_state['line_crossing_counter'].copy()
    }

# Exemple d'utilisation dans une autre application
def main():
    # ... votre code ...

    result = process_image(my_image)
    print(f"Compteurs: {result['counters']}")

    # ... votre code ...

    # Nettoyage à la fin
    app.cleanup()
```

````

### configuration.md

```markdown
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
````

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

````

### api/application.md

```markdown
# Documentation de la classe Application

La classe `Application` est le composant central qui orchestre tous les autres modules du système.

## Méthodes principales

### `__init__()`

Initialise l'application et configure les gestionnaires de signaux.

```python
app = Application()
````

### `initialize()`

Initialise tous les composants nécessaires au fonctionnement du programme.

```python
success = app.initialize()
if success:
    print("Initialisation réussie")
```

**Retourne**: `bool` - True si l'initialisation a réussi, False sinon

### `run()`

Exécute la boucle principale de l'application.

```python
app.run()
```

### `process_frame_with_tracking(current_frame)`

Traite une frame avec détection et suivi des personnes.

```python
frame = cv2.imread("image.jpg")
processed_frame, should_exit, formatted_time = app.process_frame_with_tracking(frame)
```

**Paramètres**:

- `current_frame` (np.ndarray): Frame brute à traiter

**Retourne**: `tuple` - (processed_frame, should_exit, formatted_time)

### `cleanup()`

Nettoie les ressources utilisées par l'application.

```python
app.cleanup()
```

## Méthodes internes

### `_signal_handler(signal_received, frame)`

Gestionnaire pour l'arrêt propre du programme.

### `setup_device()`

Configure le dispositif de calcul (GPU/CPU) pour le traitement.

**Retourne**: `torch.device` - Dispositif à utiliser pour les calculs

## Attributs

- `video_capture`: Objet de capture vidéo
- `tracker_state`: État du tracker de personnes
- `running` (bool): Indique si l'application est en cours d'exécution

## Exemple complet

```python
from application import Application

# Création de l'application
app = Application()

# Initialisation
if app.initialize():
    try:
        # Exécution de la boucle principale
        app.run()
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur")
    finally:
        # Nettoyage des ressources
        app.cleanup()
else:
    print("Échec de l'initialisation")
```

```

## 3. Création des diagrammes

Pour les diagrammes, vous pouvez utiliser des outils comme [draw.io](https://app.diagrams.net/), [PlantUML](https://plantuml.com/) ou [Mermaid](https://mermaid-js.github.io/).

### Diagramme des composants (component_diagram.png)

Créez un diagramme montrant les relations entre les différents modules:
- Application au centre
- Flèches montrant les dépendances entre les modules
- Regroupement des modules par fonction (traitement vidéo, détection, affichage, etc.)

### Diagramme de séquence (sequence_diagram.png)

Créez un diagramme montrant le flux d'exécution:
1. Initialisation de l'application
2. Lecture d'une frame
3. Traitement de la frame
4. Détection et suivi
5. Vérification des franchissements
6. Enregistrement des données
7. Affichage des résultats
8. Nettoyage final

## 4. Intégration dans votre projet

Pour intégrer cette documentation dans votre projet:

1. Créez le dossier `docs` à la racine de votre projet
2. Ajoutez les fichiers Markdown décrits ci-dessus
3. Créez les diagrammes et placez-les dans `docs/diagrams/`
4. Ajoutez un lien vers la documentation dans votre README principal

Cette documentation améliorée aidera les utilisateurs et les développeurs à comprendre et à utiliser efficacement votre système, tout en mettant en valeur les améliorations architecturales que vous avez apportées.
```

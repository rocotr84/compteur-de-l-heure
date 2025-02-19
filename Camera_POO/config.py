import os
import numpy as np

class Config:
    """
    Classe de configuration centralisant tous les paramètres du système.
    """
    def __init__(self):
        # Définition du répertoire courant
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # Chemins des ressources
        self.VIDEO_INPUT_PATH = os.path.join(self.current_dir, "..", "assets", "video", "p3_macbeth.mp4")
        self.MODEL_PATH = os.path.join(self.current_dir, "..", "assets", "models", "yolo11n.pt")
        self.DETECTION_MASK_PATH = os.path.join(self.current_dir, "..", "assets", "mask", "p3_macbeth_mask.jpg")
        self.CSV_OUTPUT_PATH = os.path.join(self.current_dir, "..", "Camera_POO", "detections.csv")
        self.CACHE_FILE_PATH = os.path.join(self.current_dir, "macbeth_cache.json")

        # Paramètres d'affichage
        self.output_width = 1280
        self.output_height = 720
        self.desired_fps = 30
        self._show_roi_and_color = True
        self._show_trajectories = True
        self._show_center = True
        self._show_labels = True
        self._save_video = False
        self.VIDEO_OUTPUT_PATH = os.path.join(self.current_dir, "output.mp4")
        self.VIDEO_FPS = 30
        self.VIDEO_CODEC = 'mp4v'

        # Paramètres de détection
        self._detection_mode = "color"  # "color" ou "number"
        self.DETECT_SQUARES = True
        self.MAX_DISAPPEAR_FRAMES = 30
        self.MIN_CONFIDENCE = 0.5
        self.MIN_NUMBER_CONFIDENCE = 0.4

        # Paramètres de couleur
        self.COLOR_RANGES = {
            'rouge_fonce': (np.array([0, 100, 100]), np.array([10, 255, 255])),
            'rouge2': (np.array([170, 100, 100]), np.array([180, 255, 255])),
            'vert': (np.array([35, 100, 100]), np.array([85, 255, 255])),
            'bleu': (np.array([100, 100, 100]), np.array([130, 255, 255])),
            'jaune': (np.array([20, 100, 100]), np.array([35, 255, 255])),
            'violet': (np.array([130, 100, 100]), np.array([170, 255, 255]))
        }

        # Couleurs de référence Macbeth
        self.MACBETH_REFERENCE_COLORS = np.array([
            [255, 255, 255],  # Blanc
            [128, 128, 128],  # Gris moyen
            [0, 0, 0],       # Noir
            [255, 0, 0],     # Rouge
            [0, 255, 0],     # Vert
            [0, 0, 255],     # Bleu
            [255, 255, 0],   # Jaune
            [255, 0, 255],   # Magenta
            [0, 255, 255]    # Cyan
        ], dtype=np.uint8)

        # Paramètres de détection de couleur
        self.MIN_COLOR_WEIGHT = 0.3
        self.ROI_EXPANSION_RATIO = 0.2
        self.COLOR_CORRECTION_INTERVAL = 30  # Nombre de frames entre chaque correction de couleur

        # Points de la ligne de comptage
        self.line_start = (0, self.output_height // 2)
        self.line_end = (self.output_width, self.output_height // 2)

    # Propriétés avec getters et setters
    @property
    def SHOW_ROI_AND_COLOR(self):
        return self._show_roi_and_color

    @SHOW_ROI_AND_COLOR.setter
    def SHOW_ROI_AND_COLOR(self, value):
        self._show_roi_and_color = bool(value)

    @property
    def SHOW_TRAJECTORIES(self):
        return self._show_trajectories

    @SHOW_TRAJECTORIES.setter
    def SHOW_TRAJECTORIES(self, value):
        self._show_trajectories = bool(value)

    @property
    def SHOW_CENTER(self):
        return self._show_center

    @SHOW_CENTER.setter
    def SHOW_CENTER(self, value):
        self._show_center = bool(value)

    @property
    def SHOW_LABELS(self):
        return self._show_labels

    @SHOW_LABELS.setter
    def SHOW_LABELS(self, value):
        self._show_labels = bool(value)

    @property
    def SAVE_VIDEO(self):
        return self._save_video

    @SAVE_VIDEO.setter
    def SAVE_VIDEO(self, value):
        self._save_video = bool(value)

    @property
    def DETECTION_MODE(self):
        return self._detection_mode

    @DETECTION_MODE.setter
    def DETECTION_MODE(self, value):
        if value in ["color", "number"]:
            self._detection_mode = value
        else:
            raise ValueError("Le mode de détection doit être 'color' ou 'number'")

# Création d'une instance globale de la configuration
config = Config()
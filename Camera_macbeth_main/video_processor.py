import cv2
import numpy as np
from macbeth_nonlinear_color_correction import corriger_image
import os
from config import (output_width, output_height, desired_fps, 
                   COLOR_RANGES, COLOR_MASKS, DETECTION_MASK_PATH, 
                   CACHE_FILE_PATH)
from numba import njit
from functools import lru_cache

# Variables globales
mask: np.ndarray | None = None
resized_mask: np.ndarray | None = None
frame_count = 0
last_correction_coefficients = None

def load_mask():
    """
    Charge et prépare le masque pour le traitement vidéo.
    
    Cette fonction tente de charger un masque depuis un fichier. En cas d'échec,
    elle crée un masque blanc par défaut. Le masque est automatiquement redimensionné
    aux dimensions de sortie configurées.
    
    Notes:
        Le masque est stocké dans la variable globale 'mask'
        En cas d'erreur, un masque blanc est créé par défaut
    """
    global mask, resized_mask
    print(f"Tentative de chargement du masque depuis: {DETECTION_MASK_PATH}")

    def create_default_mask() -> None:
        """Crée un masque blanc par défaut aux dimensions de sortie."""
        global mask, resized_mask
        if output_width is None or output_height is None:
            raise ValueError("Les dimensions de sortie doivent être initialisées")
        mask = np.ones((output_height, output_width), dtype=np.uint8) * 255
        resized_mask = mask.copy()
        print("Création d'un masque blanc par défaut")

    try:
        if not os.path.exists(DETECTION_MASK_PATH):
            print(f"ERREUR: Le fichier de masque n'existe pas: {DETECTION_MASK_PATH}")
            create_default_mask()
            return

        # Chargement du masque en niveaux de gris
        mask = cv2.imread(DETECTION_MASK_PATH, 0)
        
        if mask is None:
            print(f"ERREUR: Impossible de charger le masque: {DETECTION_MASK_PATH}")
            create_default_mask()
        else:
            # Pré-calcul du masque redimensionné une seule fois
            original_shape = mask.shape
            resized_mask = cv2.resize(mask, (output_width, output_height))
            print(f"Masque chargé avec succès: {original_shape} et redimensionné à {(output_height, output_width)}")
            
    except Exception as e:
        print(f"ERREUR lors du chargement du masque: {str(e)}")
        create_default_mask()

def setup_video_capture(video_path):
    """
    Configure la capture vidéo avec les paramètres souhaités.
    
    Args:
        video_path (str): Chemin vers le fichier vidéo
    
    Returns:
        cv2.VideoCapture: Objet de capture vidéo configuré
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Erreur : Impossible d'accéder à la vidéo.")
    
    # Configuration du FPS souhaité
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    return cap

@njit
def apply_mask(frame, mask):
    return cv2.bitwise_and(frame, frame, mask=mask)

@lru_cache(maxsize=1)
def get_resize_matrix(input_shape):
    """Cache la matrice de transformation pour une taille d'entrée donnée"""
    width, height = int(input_shape[1]), int(input_shape[0])
    
    # Vérifier si le redimensionnement est nécessaire
    if (width, height) == (output_width, output_height):
        return None
    
    # Convertir explicitement en float32 pour OpenCV
    src_points = np.array([
        [0, 0],
        [width-1, 0],
        [0, height-1]
    ], dtype=np.float32)
    
    dst_points = np.array([
        [0, 0],
        [output_width-1, 0],
        [0, output_height-1]
    ], dtype=np.float32)
    
    return cv2.getAffineTransform(src_points, dst_points)

@njit
def _apply_single_mask(frame, mask):
    """
    Applique un masque unique à une frame en utilisant Numba pour l'optimisation.
    
    Args:
        frame (np.ndarray): Image à masquer
        mask (np.ndarray): Masque binaire à appliquer
    
    Returns:
        np.ndarray: Image masquée
    """
    return frame * (mask[:, :, None] / 255.0)

def apply_masks_batch(frames, masks):
    """
    Applique un lot de masques à un lot de frames.
    
    Args:
        frames (np.ndarray): Lot d'images à masquer
        masks (np.ndarray): Lot de masques binaires à appliquer
    
    Returns:
        np.ndarray: Lot d'images masquées
    """
    return np.array([_apply_single_mask(frame, mask) 
                    for frame, mask in zip(frames, masks)])

def process_frame(frame_raw, detect_squares):
    """
    Traite une frame individuelle de la vidéo.
    
    Le traitement inclut :
    1. Redimensionnement aux dimensions configurées
    2. Application du masque si disponible
    3. Correction des couleurs via l'algorithme Macbeth
    
    Args:
        frame_raw (np.array): Image brute à traiter (format BGR)
        detect_squares (bool): Si True, détecte les carrés Macbeth, sinon utilise le cache
    
    Returns:
        np.array: Image traitée avec les couleurs corrigées et le masque appliqué
    """
    try:
        # Éviter le redimensionnement si les dimensions sont déjà correctes
        matrix = get_resize_matrix(frame_raw.shape)
        if matrix is None:
            frame_resized = frame_raw
        else:
            frame_resized = cv2.warpAffine(frame_raw, matrix, (output_width, output_height))
        
        # Application vectorisée du masque
        if resized_mask is not None:
            frame_masked = apply_masks_batch(
                np.array([frame_resized]), 
                np.array([resized_mask])
            )[0]
        else:
            frame_masked = frame_resized
        
        # Correction des couleurs avec gestion des erreurs
        frame_corrected = corriger_image(frame_masked, CACHE_FILE_PATH, detect_squares)
        if frame_corrected is None:
            print("Erreur: La correction des couleurs a échoué")
            return frame_masked
            
        return frame_corrected
        
    except Exception as e:
        print(f"Erreur lors du traitement de la frame: {str(e)}")
        return frame_raw

def initialize_color_masks():
    """
    Initialise les masques de couleur avec vérification vectorisée.
    """
    try:
        # Conversion des plages en tableaux NumPy pour un traitement vectorisé
        hsv_ranges = np.array([(hsv_min, hsv_max) for color_name, (hsv_min, hsv_max) in COLOR_RANGES.items()])
        
        # Vérification vectorisée des valeurs HSV
        if not np.all((hsv_ranges[:, 0, 0] >= 0) & (hsv_ranges[:, 1, 0] <= 180)):
            raise ValueError("Valeurs H invalides détectées")
        if not np.all((hsv_ranges[:, :, 1:] >= 0) & (hsv_ranges[:, :, 1:] <= 255)):
            raise ValueError("Valeurs S/V invalides détectées")
        
        # Création vectorisée des masques
        for color_name, (hsv_min, hsv_max) in COLOR_RANGES.items():
            COLOR_MASKS[color_name] = {
                'min': np.array(hsv_min, dtype=np.uint8),
                'max': np.array(hsv_max, dtype=np.uint8)
            }
        print(f"Masques de couleurs initialisés pour {len(COLOR_MASKS)} couleurs")
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation des masques de couleurs: {str(e)}")
        raise

def get_color_mask(color_name):
    """
    Récupère les seuils min et max pour une couleur donnée.
    """
    if color_name not in COLOR_MASKS:
        raise ValueError(f"Couleur non reconnue : {color_name}")
    return COLOR_MASKS[color_name]['min'], COLOR_MASKS[color_name]['max']

@njit
def process_color_ranges():
    """Pré-calcule les plages de couleurs"""
    return np.array([
        (hsv_min, hsv_max) 
        for _, (hsv_min, hsv_max) in COLOR_RANGES.items()
    ])

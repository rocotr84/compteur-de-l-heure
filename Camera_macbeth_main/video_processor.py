import cv2
import numpy as np
from macbeth_nonlinear_color_correction import corriger_image
import os
# Variables globales pour stocker la configuration
output_width: int = None  # type: ignore
output_height: int = None  # type: ignore
desired_fps: int = None  # type: ignore
mask: np.ndarray | None = None

def init_video_processor(width: int, height: int, fps: int) -> None:
    """
    Initialise le processeur vidéo avec les paramètres de sortie souhaités.
    
    Cette fonction configure les dimensions et la fréquence d'images pour le traitement vidéo.
    Elle doit être appelée avant toute autre opération de traitement.
    
    Args:
        width (int): Largeur souhaitée pour les frames de sortie en pixels
        height (int): Hauteur souhaitée pour les frames de sortie en pixels
        fps (int): Nombre d'images par seconde souhaité pour la sortie
    """
    global output_width, output_height, desired_fps
    output_width = width
    output_height = height
    desired_fps = fps

def load_mask(mask_path):
    """
    Charge et prépare le masque pour le traitement vidéo.
    
    Cette fonction tente de charger un masque depuis un fichier. En cas d'échec,
    elle crée un masque blanc par défaut. Le masque est automatiquement redimensionné
    aux dimensions de sortie configurées.
    
    Args:
        mask_path (str): Chemin vers le fichier de masque (format image)
    
    Notes:
        Le masque est stocké dans la variable globale 'mask'
        En cas d'erreur, un masque blanc est créé par défaut
    """
    global mask
    print(f"Tentative de chargement du masque depuis: {mask_path}")

    def create_default_mask() -> None:
        """
        Crée un masque blanc par défaut.
        """
        global mask
        if output_width is None or output_height is None:
            raise ValueError("Les dimensions de sortie doivent être initialisées avant de créer le masque")
        mask = np.ones((output_height, output_width), dtype=np.uint8) * 255
        print("Création d'un masque blanc par défaut")

    try:
        if not os.path.exists(mask_path):
            print(f"ERREUR: Le fichier de masque n'existe pas: {mask_path}")
            create_default_mask()
            return

        # Chargement du masque en niveaux de gris
        mask = cv2.imread(mask_path, 0)
        
        if mask is None:
            print(f"ERREUR: Impossible de charger le masque: {mask_path}")
            create_default_mask()
        else:
            # Redimensionnement du masque aux dimensions souhaitées
            original_shape = mask.shape
            mask = cv2.resize(mask, (output_width, output_height))
            print(f"Masque chargé avec succès: {original_shape} redimensionné à {(output_height, output_width)}")
            
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

def process_frame(frame_raw, cache_file, detect_squares):
    """
    Traite une frame individuelle de la vidéo.
    
    Le traitement inclut :
    1. Redimensionnement aux dimensions configurées
    2. Application du masque si disponible
    3. Correction des couleurs via l'algorithme Macbeth
    
    Args:
        frame_raw (np.array): Image brute à traiter (format BGR)
        cache_file (str): Chemin vers le fichier de cache pour la correction des couleurs
        detect_squares (bool): Si True, détecte les carrés Macbeth, sinon utilise le cache
    
    Returns:
        np.array: Image traitée avec les couleurs corrigées et le masque appliqué
    
    Notes:
        Le masque est redimensionné automatiquement si ses dimensions ne correspondent
        pas à celles de la frame
    """
    # Redimensionnement de la frame
    frame_resized = cv2.resize(frame_raw, (output_width, output_height))
    
    # Application du masque si disponible
    frame_masked = frame_resized
    if mask is not None:
        # Vérification de la compatibilité des dimensions
        if mask.shape[:2] != frame_resized.shape[:2]:
            mask_resized = cv2.resize(mask, (frame_resized.shape[1], frame_resized.shape[0]))
            frame_masked = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_resized)
            cv2.imshow('Masque', frame_masked)
    
    # Appel à la fonction corriger_image pour ajuster les couleurs
    frame_corrected = corriger_image(frame_masked, cache_file, detect_squares)
    
    return frame_corrected

import cv2
import numpy as np
from macbeth_nonlinear_color_correction import corriger_image
import os

# Variables globales pour stocker la configuration
output_width = None
output_height = None
desired_fps = None
mask = None

def init_video_processor(width, height, fps):
    """
    Initialise le processeur vidéo avec les paramètres de sortie souhaités
    Args:
        width (int): Largeur souhaitée pour les frames de sortie
        height (int): Hauteur souhaitée pour les frames de sortie
        fps (int): Nombre d'images par seconde souhaité
    """
    global output_width, output_height, desired_fps
    output_width = width
    output_height = height
    desired_fps = fps

def load_mask(mask_path):
    """
    Charge et prépare le masque pour le traitement vidéo
    Args:
        mask_path (str): Chemin vers le fichier de masque
    """
    global mask
    print(f"Tentative de chargement du masque depuis: {mask_path}")

    def create_default_mask():
        global mask
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
            print(f"Masque chargé avec succès:")
            
    except Exception as e:
        print(f"ERREUR lors du chargement du masque: {str(e)}")
        create_default_mask()

def setup_video_capture(video_path):
    """
    Configure la capture vidéo avec les paramètres souhaités
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

def process_frame(frame, cache_file, detect_squares):
    """
    Traite une frame individuelle
    Args:
        frame (np.array): Image à traiter
        cache_file (str): Chemin vers le fichier de cache
        detect_squares (bool): Si True, détecte les carrés, sinon utilise le cache
    Returns:
        np.array: Image traitée
    """
    # Redimensionnement de la frame
    frame = cv2.resize(frame, (output_width, output_height))
    
    # Application du masque si disponible
    if mask is not None:
        # Vérification de la compatibilité des dimensions
        if mask.shape[:2] != frame.shape[:2]:
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            frame = cv2.bitwise_and(frame, frame, mask=resized_mask)
            cv2.imshow('Masque', frame)
    
    # Appel à la fonction corriger_image pour ajuster les couleurs
    frame = corriger_image(frame, cache_file, detect_squares)
    
    return frame

import cv2
import numpy as np
from src.color_weighting import get_weighted_color_probabilities, update_color_timestamp
from src.video_processor import get_color_mask
from config.color_config import COLOR_RANGES, COLOR_MASKS

def get_dominant_color(frame_raw, detection_zone_coords):
    """
    Détecte la couleur dominante dans une région d'intérêt (ROI) de l'image.
    
    Le processus comprend :
    1. Extraction de la zone de détection
    2. Conversion en espace colorimétrique HSV
    3. Détection vectorisée des pixels dans chaque plage de couleur
    4. Application des pondérations pour déterminer la couleur dominante
    
    Args:
        frame_raw (np.array): Image complète au format BGR
        detection_zone_coords (tuple): Coordonnées de la zone (x1, y1, x2, y2)
    
    Returns:
        str: Nom de la couleur dominante ou "inconnu" en cas d'échec
    
    Notes:
        La fonction met à jour l'horodatage de la couleur détectée via update_color_timestamp
    """
    try:
        zone_x1, zone_y1, zone_x2, zone_y2 = detection_zone_coords
        frame_detection_zone = frame_raw[zone_y1:zone_y2, zone_x1:zone_x2]
        
        if frame_detection_zone.size == 0:
            return "inconnu"

        frame_detection_zone_hsv = cv2.cvtColor(frame_detection_zone, cv2.COLOR_BGR2HSV)
        
        # Approche vectorisée pour la détection des couleurs
        detected_pixels_per_color = {}
        
        # Préparation des masques en une seule opération
        color_names = list(COLOR_MASKS.keys())
        
        for color_name in color_names:
            hsv_min, hsv_max = get_color_mask(color_name)
            mask = cv2.inRange(frame_detection_zone_hsv, hsv_min, hsv_max)
            
            # Gestion spéciale pour le rouge (qui traverse 0° en HSV)
            if color_name == 'rouge_fonce':
                hsv_min2, hsv_max2 = get_color_mask('rouge2')
                mask2 = cv2.inRange(frame_detection_zone_hsv, hsv_min2, hsv_max2)
                mask = cv2.bitwise_or(mask, mask2)
            
            detected_pixels_per_color[color_name] = cv2.countNonZero(mask)

        weighted_color_probabilities = get_weighted_color_probabilities(detected_pixels_per_color)
        
        # Utilisation de max() avec une fonction lambda pour trouver la couleur dominante
        if weighted_color_probabilities:
            max_item = max(weighted_color_probabilities.items(), key=lambda x: x[1], default=(None, 0))
            if max_item[0] is not None and max_item[1] > 0:
                dominant_color_name = max_item[0]
                update_color_timestamp(dominant_color_name)
                return dominant_color_name
            
        return "inconnu"

    except Exception as e:
        print(f"Erreur lors de la détection de couleur: {str(e)}")
        return "inconnu"

def visualize_color(frame_raw, detection_zone_coords, detected_color_name):
    """
    Visualise la couleur détectée en dessinant un rectangle sur l'image.
    
    Args:
        frame_raw (np.array): Image sur laquelle dessiner (format BGR)
        detection_zone_coords (tuple): Coordonnées de la zone (x1, y1, x2, y2)
        detected_color_name (str): Nom de la couleur détectée
    
    Notes:
        Les couleurs de visualisation sont définies en BGR :
        - Couleurs spécifiques pour chaque couleur détectée
        - Gris (128, 128, 128) pour une couleur inconnue
    """
    zone_x1, zone_y1, zone_x2, zone_y2 = detection_zone_coords
    
    visualization_colors = {
        "rouge_fonce": (0, 0, 255),
        "bleu_fonce": (255, 0, 0),
        "bleu_clair": (255, 128, 0),
        "vert_fonce": (0, 255, 0),
        "vert_clair": (0, 255, 128),
        "rose": (255, 0, 255),
        "jaune": (0, 255, 255),
        "blanc": (255, 255, 255),
        "noir": (0, 0, 0),
        "inconnu": (128, 128, 128)
    }

    rectangle_color = visualization_colors.get(detected_color_name, (128, 128, 128))
    cv2.rectangle(frame_raw, 
                 (zone_x1, zone_y1), 
                 (zone_x2, zone_y2), 
                 rectangle_color, 
                 2)

def detect_dominant_color(frame_roi):
    """
    Détecte la couleur dominante dans une région d'intérêt.
    """
    try:
        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        total_pixels = frame_roi.shape[0] * frame_roi.shape[1]
        detected_pixels = {}

        for color_name in COLOR_RANGES.keys():
            hsv_min, hsv_max = get_color_mask(color_name)
            mask = cv2.inRange(hsv, hsv_min, hsv_max)

            # Gestion spéciale pour le rouge (qui traverse 0° en HSV)
            if color_name == 'rouge':
                hsv_min2, hsv_max2 = get_color_mask('rouge2')
                mask2 = cv2.inRange(hsv, hsv_min2, hsv_max2)
                mask = cv2.bitwise_or(mask, mask2)

            pixel_count = cv2.countNonZero(mask)
            pixel_ratio = pixel_count / total_pixels
            detected_pixels[color_name] = (pixel_ratio, pixel_count)

        # Trouver la couleur dominante
        dominant_color = max(detected_pixels.items(), 
                           key=lambda x: x[1][0])

        return (dominant_color[0],           # nom de la couleur
               dominant_color[1][0],         # ratio de pixels
               dominant_color[1][1])         # nombre de pixels

    except Exception as e:
        print(f"Erreur lors de la détection des couleurs: {str(e)}")
        return None, 0, 0 
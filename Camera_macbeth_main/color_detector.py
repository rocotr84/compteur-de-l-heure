import cv2
import numpy as np
from color_weighting import get_weighted_color_probabilities, update_color_timestamp

# Définition des plages de couleurs HSV pour chaque couleur reconnue
# Format: ((hue_min, saturation_min, value_min), (hue_max, saturation_max, value_max))
color_detection_ranges = {
    "noir": ((0, 0, 0), (180, 255, 50)),
    "blanc": ((0, 0, 200), (180, 30, 255)),
    "rouge_fonce": ((0, 50, 50), (10, 255, 255)),
    "bleu_fonce": ((100, 50, 50), (130, 255, 120)),
    "bleu_clair": ((100, 50, 121), (130, 255, 255)),
    "vert_fonce": ((35, 50, 50), (85, 255, 255)),
    "rose": ((140, 50, 50), (170, 255, 255)),
    "jaune": ((20, 100, 100), (40, 255, 255)),
    "vert_clair": ((40, 50, 50), (80, 255, 255))
}

def get_dominant_color(frame_raw, detection_zone_coords):
    """
    Détecte la couleur dominante dans une région d'intérêt (ROI) de l'image.
    
    Le processus comprend :
    1. Extraction de la zone de détection
    2. Conversion en espace colorimétrique HSV
    3. Détection des pixels dans chaque plage de couleur
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
        
        detected_pixels_per_color = {}
        for color_name, (range_min, range_max) in color_detection_ranges.items():
            hsv_min_threshold = np.array(range_min, dtype=np.uint8)
            hsv_max_threshold = np.array(range_max, dtype=np.uint8)
            color_detection_mask = cv2.inRange(frame_detection_zone_hsv, 
                                            hsv_min_threshold, 
                                            hsv_max_threshold)
            detected_pixels_per_color[color_name] = cv2.countNonZero(color_detection_mask)

        weighted_color_probabilities = get_weighted_color_probabilities(detected_pixels_per_color)
        
        dominant_color_name, dominant_color_weight = max(weighted_color_probabilities.items(), 
                                                       key=lambda x: x[1])
        
        if dominant_color_weight > 0:
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
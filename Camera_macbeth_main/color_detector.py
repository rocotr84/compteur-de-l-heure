import cv2
import numpy as np
from color_weighting import get_weighted_color_probabilities, update_color_timestamp

# Définition des plages de couleurs HSV
color_ranges = {
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

def get_dominant_color(frame, roi_coords):
    """
    Détecte la couleur dominante dans la ROI
    Args:
        frame (np.array): Image complète
        roi_coords (tuple): Coordonnées de la ROI (x1, y1, x2, y2)
    Returns:
        str: Nom de la couleur dominante
    """
    try:
        x1, y1, x2, y2 = roi_coords
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "inconnu"

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        color_counts = {}
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower, upper)
            color_counts[color_name] = cv2.countNonZero(mask)

        weighted_counts = get_weighted_color_probabilities(color_counts)
        
        dominant_color = max(weighted_counts.items(), key=lambda x: x[1])
        
        if dominant_color[1] > 0:
            update_color_timestamp(dominant_color[0])
            return dominant_color[0]
            
        return "inconnu"

    except Exception as e:
        print(f"Erreur lors de la détection de couleur: {str(e)}")
        return "inconnu"

def visualize_color(frame, roi_coords, color_name):
    """
    Visualise la couleur détectée sur l'image
    Args:
        frame (np.array): Image sur laquelle dessiner
        roi_coords (tuple): Coordonnées de la ROI (x1, y1, x2, y2)
        color_name (str): Nom de la couleur détectée
    """
    x1, y1, x2, y2 = roi_coords
    
    color_bgr = {
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

    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                 color_bgr.get(color_name, (128, 128, 128)), 2) 
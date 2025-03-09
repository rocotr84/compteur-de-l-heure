import time
from collections import defaultdict
from config.color_config import (
    MIN_TIME_BETWEEN_PASSES,
    MIN_COLOR_WEIGHT,
)

# Variables globales pour le suivi temporel des couleurs
color_detection_history = defaultdict(float)

def get_color_weight(detected_color_name, current_timestamp):
    """
    Calcule le poids à appliquer pour une couleur en fonction du temps écoulé.
    
    Cette fonction implémente une pénalisation temporelle pour éviter les 
    détections multiples d'une même couleur dans un court intervalle.
    
    Args:
        detected_color_name (str): Identifiant de la couleur
        current_timestamp (float): Timestamp actuel en secondes
    
    Returns:
        float: Poids entre MIN_COLOR_WEIGHT et 1.0
               - 1.0 si la couleur n'a pas été vue récemment
               - Valeur réduite si la couleur a été vue récemment
    """
    previous_detection_time = color_detection_history.get(detected_color_name, 0)
    time_since_last_detection = current_timestamp - previous_detection_time
    
    if time_since_last_detection < MIN_TIME_BETWEEN_PASSES:
        # Calcul d'une pénalité progressive basée sur le temps écoulé
        detection_penalty = 1.0 - (time_since_last_detection / MIN_TIME_BETWEEN_PASSES)
        return max(MIN_COLOR_WEIGHT, 1.0 - detection_penalty)
    
    return 1.0

def update_color_timestamp(detected_color_name, detection_timestamp=None):
    """
    Met à jour le timestamp du dernier passage pour une couleur donnée.
    
    Args:
        detected_color_name (str): Identifiant de la couleur à mettre à jour
        detection_timestamp (float, optional): Timestamp spécifique. Si None, utilise le temps actuel
    """
    color_detection_history[detected_color_name] = detection_timestamp if detection_timestamp is not None else time.time()

def get_weighted_color_probabilities(detected_color_pixels, current_timestamp=None):
    """
    Applique une pondération temporelle aux comptages de couleurs détectées.
    
    Cette fonction ajuste les comptages bruts en fonction du temps écoulé
    depuis la dernière détection de chaque couleur.
    
    Args:
        detected_color_pixels (dict): Dictionnaire {couleur: nombre_de_pixels}
        current_timestamp (float, optional): Timestamp pour le calcul. Si None, utilise le temps actuel
    
    Returns:
        dict: Dictionnaire {couleur: compte_pondéré} avec les comptages ajustés
              selon la pondération temporelle
    """
    if current_timestamp is None:
        current_timestamp = time.time()
        
    weighted_detection_counts = {}
    for detected_color_name, pixel_count in detected_color_pixels.items():
        color_temporal_weight = get_color_weight(detected_color_name, current_timestamp)
        weighted_detection_counts[detected_color_name] = pixel_count * color_temporal_weight
        
    return weighted_detection_counts 
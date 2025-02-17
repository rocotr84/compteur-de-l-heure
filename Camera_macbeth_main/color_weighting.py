import time
from collections import defaultdict
from config import (
    MIN_TIME_BETWEEN_PASSES,
    MIN_COLOR_WEIGHT,
)

# Variables globales pour le suivi temporel des couleurs
color_last_seen = defaultdict(float)

def get_color_weight(color, current_time):
    """
    Calcule le poids à appliquer pour une couleur en fonction du temps écoulé.
    
    Cette fonction implémente une pénalisation temporelle pour éviter les 
    détections multiples d'une même couleur dans un court intervalle.
    
    Args:
        color (str): Identifiant de la couleur
        current_time (float): Timestamp actuel en secondes
    
    Returns:
        float: Poids entre MIN_COLOR_WEIGHT et 1.0
               - 1.0 si la couleur n'a pas été vue récemment
               - Valeur réduite si la couleur a été vue récemment
    """
    last_seen = color_last_seen.get(color, 0)
    time_since_last_seen = current_time - last_seen
    
    if time_since_last_seen < MIN_TIME_BETWEEN_PASSES:
        # Calcul d'une pénalité progressive basée sur le temps écoulé
        penalty = 1.0 - (time_since_last_seen / MIN_TIME_BETWEEN_PASSES)
        return max(MIN_COLOR_WEIGHT, 1.0 - penalty)
    
    return 1.0

def update_color_timestamp(color, timestamp=None):
    """
    Met à jour le timestamp du dernier passage pour une couleur donnée.
    
    Args:
        color (str): Identifiant de la couleur à mettre à jour
        timestamp (float, optional): Timestamp spécifique. Si None, utilise le temps actuel
    """
    color_last_seen[color] = timestamp if timestamp is not None else time.time()

def get_weighted_color_probabilities(color_counts, current_time=None):
    """
    Applique une pondération temporelle aux comptages de couleurs détectées.
    
    Cette fonction ajuste les comptages bruts en fonction du temps écoulé
    depuis la dernière détection de chaque couleur.
    
    Args:
        color_counts (dict): Dictionnaire {couleur: nombre_de_pixels}
        current_time (float, optional): Timestamp pour le calcul. Si None, utilise le temps actuel
    
    Returns:
        dict: Dictionnaire {couleur: compte_pondéré} avec les comptages ajustés
              selon la pondération temporelle
    """
    if current_time is None:
        current_time = time.time()
        
    weighted_counts = {}
    for color, count in color_counts.items():
        weight = get_color_weight(color, current_time)
        weighted_counts[color] = count * weight
        
    return weighted_counts 
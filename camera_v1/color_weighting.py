import time
from collections import defaultdict
from config import MIN_TIME_BETWEEN_PASSES, PENALTY_DURATION, MIN_COLOR_WEIGHT, MIN_PIXEL_RATIO, MIN_PIXEL_COUNT, COLOR_HISTORY_SIZE

class ColorWeightManager:
    """
    Gestionnaire de pondération des couleurs basé sur le temps.
    """
    
    def __init__(self):
        self.color_last_seen = defaultdict(float)
        
    def get_color_weight(self, color, current_time):
        """
        Calcule le poids à appliquer pour une couleur donnée.
        """
        last_seen = self.color_last_seen.get(color, 0)
        time_since_last_seen = current_time - last_seen
        
        if time_since_last_seen < MIN_TIME_BETWEEN_PASSES:
            # Calcul d'une pénalité progressive
            penalty = 1.0 - (time_since_last_seen / MIN_TIME_BETWEEN_PASSES)
            return max(MIN_COLOR_WEIGHT, 1.0 - penalty)
        
        return 1.0
    
    def update_color_timestamp(self, color, timestamp=None):
        """
        Met à jour le timestamp du dernier passage pour une couleur.
        """
        self.color_last_seen[color] = timestamp if timestamp is not None else time.time()
    
    def get_weighted_color_probabilities(self, color_counts, current_time=None):
        """
        Applique la pondération temporelle aux comptages de couleurs.
        """
        if current_time is None:
            current_time = time.time()
            
        weighted_counts = {}
        for color, count in color_counts.items():
            weight = self.get_color_weight(color, current_time)
            weighted_counts[color] = count * weight
            
        return weighted_counts 
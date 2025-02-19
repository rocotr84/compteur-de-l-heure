import cv2
import numpy as np
from config import config

class ColorDetector:
    """
    Classe responsable de la détection et de l'analyse des couleurs dans les images
    """
    def __init__(self):
        self.color_masks = {}
        self._initialize_color_masks()

    def _initialize_color_masks(self):
        """Initialise les masques de couleur pré-calculés"""
        for color_name, (lower, upper) in config.COLOR_RANGES.items():
            lower = np.array(lower)
            upper = np.array(upper)
            self.color_masks[color_name] = (lower, upper)

    def get_dominant_color(self, image, mask=None):
        """
        Détermine la couleur dominante dans une région d'image
        
        Args:
            image: Image en format BGR
            mask: Masque binaire optionnel pour la région d'intérêt
        
        Returns:
            tuple: (nom_couleur, pourcentage)
        """
        if image is None:
            return None, 0
        
        if mask is None:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total_pixels = cv2.countNonZero(mask)
        
        if total_pixels == 0:
            return None, 0

        best_color = None
        max_percentage = 0

        for color_name, (lower, upper) in self.color_masks.items():
            color_mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, mask)
            matching_pixels = cv2.countNonZero(color_mask)
            percentage = matching_pixels / total_pixels

            if percentage > max_percentage:
                max_percentage = percentage
                best_color = color_name

            # Cas spécial pour le rouge (qui traverse la limite de teinte)
            if color_name == 'rouge_fonce':
                rouge2_mask = cv2.inRange(hsv, *self.color_masks['rouge2'])
                rouge2_mask = cv2.bitwise_and(rouge2_mask, mask)
                rouge2_pixels = cv2.countNonZero(rouge2_mask)
                total_rouge = (matching_pixels + rouge2_pixels) / total_pixels
                
                if total_rouge > max_percentage:
                    max_percentage = total_rouge
                    best_color = 'rouge'

        return best_color, max_percentage

    def apply_color_correction(self, image, reference_colors=None):
        """
        Applique la correction des couleurs à l'image
        
        Args:
            image: Image en format BGR
            reference_colors: Couleurs de référence (optionnel)
        
        Returns:
            np.ndarray: Image corrigée
        """
        if reference_colors is None:
            reference_colors = config.MACBETH_REFERENCE_COLORS

        # Conversion en float32 pour les calculs
        image_float = image.astype(np.float32)
        
        # Calcul des moyennes pour chaque canal
        src_means = np.mean(image_float, axis=(0, 1))
        ref_means = np.mean(reference_colors, axis=0)
        
        # Calcul des facteurs de correction
        correction_factors = ref_means / (src_means + 1e-6)  # Évite la division par zéro
        
        # Application de la correction
        corrected_image = np.clip(image_float * correction_factors, 0, 255)
        
        return corrected_image.astype(np.uint8)

    def get_color_mask(self, color_name):
        """
        Récupère les seuils de couleur pour une couleur donnée
        
        Args:
            color_name: Nom de la couleur
            
        Returns:
            tuple: (seuil_bas, seuil_haut)
        """
        return self.color_masks.get(color_name)
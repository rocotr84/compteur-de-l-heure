import numpy as np
from scipy.optimize import least_squares
from numba import njit, prange
from config import config

class ColorCorrection:
    """Classe responsable de la correction non linéaire des couleurs"""
    
    def __init__(self):
        self.last_correction_params = None
        self.frame_count = 0

    @staticmethod
    @njit(parallel=True)
    def _apply_color_correction(pixels_raw, correction_coefficients):
        """Version optimisée avec Numba de la correction des couleurs"""
        result = np.empty_like(pixels_raw)
        
        # Extraction des coefficients
        a_b, b_b, c_b, d_b, gamma_b = correction_coefficients[0:5]
        a_g, b_g, c_g, d_g, gamma_g = correction_coefficients[5:10]
        a_r, b_r, c_r, d_r, gamma_r = correction_coefficients[10:15]
        
        for i in prange(len(pixels_raw)):
            B, G, R = pixels_raw[i]
            
            # Calcul des combinaisons linéaires
            B_lin = max(a_b * B + b_b * G + c_b * R + d_b, 1e-6)
            G_lin = max(a_g * B + b_g * G + c_g * R + d_g, 1e-6)
            R_lin = max(a_r * B + b_r * G + c_r * R + d_r, 1e-6)
            
            # Application de la correction gamma
            result[i, 0] = B_lin ** gamma_b
            result[i, 1] = G_lin ** gamma_g
            result[i, 2] = R_lin ** gamma_r
        
        return result

    def calibrate_transformation(self, colors_measured, colors_target):
        """Optimise les paramètres de la transformation non linéaire"""
        colors_measured_norm = colors_measured / 255.0
        colors_target_norm = colors_target / 255.0
        
        def residuals(correction_coefficients):
            pred = self._apply_color_correction(colors_measured_norm, correction_coefficients)
            return (pred - colors_target_norm).ravel()
        
        # Initialisation : transformation identitaire
        coefficients_init = np.array([1, 0, 0, 0, 1] * 3, dtype=np.float32)
        
        # Bornes sur les paramètres gamma
        bounds_lower = [-np.inf] * 15
        bounds_upper = [np.inf] * 15
        for i in [4, 9, 14]:  # Indices des gammas
            bounds_lower[i] = 0.1
            bounds_upper[i] = 5.0
        
        optimization_result = least_squares(residuals, coefficients_init,
                                         bounds=(bounds_lower, bounds_upper),
                                         method="trf")
        
        self.last_correction_params = optimization_result.x
        return self.last_correction_params

    def apply_correction(self, frame):
        """Applique la correction non linéaire à une image"""
        if self.last_correction_params is None:
            return frame
            
        h, w, _ = frame.shape
        frame_normalized = frame.astype(np.float32) / 255.0
        pixels_raw = frame_normalized.reshape(-1, 3)
        
        try:
            pixels_corrected = self._apply_color_correction(pixels_raw, self.last_correction_params)
            pixels_corrected = np.clip(pixels_corrected, 0, 1)
            frame_corrected = (pixels_corrected.reshape(h, w, 3) * 255).astype(np.uint8)
            return frame_corrected
        except Exception as e:
            print(f"Erreur lors de la correction des couleurs: {str(e)}")
            return frame
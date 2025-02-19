"""
Module de correction non linéaire des couleurs utilisant une charte Macbeth.

Ce module implémente une correction des couleurs en espace BGR via une transformation
non linéaire calibrée sur une charte de couleur Macbeth.

Principe de la correction :
    Pour chaque canal (B, G, R), la correction applique la transformation :
    couleur_corrigée_j = (a_j * B + b_j * G + c_j * R + d_j) ** gamma_j

    Les 15 paramètres sont optimisés pour minimiser l'erreur entre les couleurs
    mesurées et les couleurs cibles de la charte Macbeth.
"""


import numpy as np
from scipy.optimize import least_squares
from numba import njit, prange
from macbeth_color_and_rectangle_detector import get_average_colors
from config import COLOR_CORRECTION_INTERVAL, MACBETH_REFERENCE_COLORS
# Variables globales pour la mise en cache des coefficients
last_correction_params = None
frame_count = 0

@njit(parallel=True)
def _apply_color_correction(pixels_raw, correction_coefficients):
    """
    Version optimisée avec Numba de la correction des couleurs
    """
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

def modele_non_lineaire(correction_coefficients, colors_input):
    """
    Applique le modèle non linéaire de correction des couleurs.

    Pour chaque canal (B, G, R), applique la transformation :
    f_j(x) = (a_j * B + b_j * G + c_j * R + d_j) ** gamma_j

    Args:
        correction_coefficients (np.array): Vecteur de 15 coefficients organisés comme suit :
            - Canal B : a_b, b_b, c_b, d_b, gamma_b   (coefficients[0:5])
            - Canal G : a_g, b_g, c_g, d_g, gamma_g   (coefficients[5:10])
            - Canal R : a_r, b_r, c_r, d_r, gamma_r   (coefficients[10:15])
        colors_input (np.array): Matrice (n,3) des couleurs d'entrée en BGR

    Returns:
        np.array: Matrice (n,3) des couleurs transformées en BGR

    Notes:
        Les valeurs négatives sont évitées en appliquant un seuil minimum de 1e-6
    """
    return _apply_color_correction(colors_input, correction_coefficients)

def calibrer_transformation_non_lineaire(colors_measured, colors_target):
    """
    Optimise les paramètres de la transformation non linéaire.

    Cette fonction détermine les 15 paramètres optimaux qui minimisent
    l'erreur entre les couleurs mesurées et les couleurs cibles.

    Args:
        colors_measured (np.array): Couleurs mesurées (n,3) en BGR, normalisées [0,1]
        colors_target (np.array): Couleurs cibles (n,3) en BGR, normalisées [0,1]

    Returns:
        np.array: Vecteur des 15 paramètres optimaux

    Notes:
        - Initialisation avec une transformation identitaire
        - Les paramètres gamma sont contraints entre 0.1 et 5
    """
    def residuals(correction_coefficients):
        pred = modele_non_lineaire(correction_coefficients, colors_measured)
        return (pred - colors_target).ravel()
    
    # Initialisation : transformation identitaire pour chaque canal
    coefficients_init = np.array([1, 0, 0, 0, 1,
                                   0, 1, 0, 0, 1,
                                   0, 0, 1, 0, 1], dtype=np.float32)
    
    # Bornes sur les paramètres gamma pour éviter des valeurs non physiques
    bounds_lower = [-np.inf] * 15
    bounds_upper = [np.inf] * 15
    bounds_lower[4] = 0.1; bounds_lower[9] = 0.1; bounds_lower[14] = 0.1  # gamma >= 0.1
    bounds_upper[4] = 5;   bounds_upper[9] = 5;   bounds_upper[14] = 5    # gamma <= 5
    
    optimization_result = least_squares(residuals, coefficients_init, 
                                          bounds=(bounds_lower, bounds_upper), 
                                          method="trf")
    return optimization_result.x

def appliquer_correction_non_lineaire(frame_masked, correction_coefficients):
    """
    Applique la correction non linéaire à une image complète.

    Args:
        frame_masked (np.array): Image d'entrée en BGR avec masque appliqué (uint8)
        correction_coefficients (np.array): Vecteur des 15 paramètres de correction

    Returns:
        np.array: Image corrigée en BGR (uint8)

    Notes:
        Les valeurs sont automatiquement clippées dans [0,255]
    """
    h, w, _ = frame_masked.shape
    frame_normalized = frame_masked.astype(np.float32) / 255.0
    pixels_raw = frame_normalized.reshape(-1, 3)
    
    try:
        pixels_corrected = _apply_color_correction(pixels_raw, correction_coefficients)
        pixels_corrected = np.clip(pixels_corrected, 0, 1)
        frame_corrected = (pixels_corrected.reshape(h, w, 3) * 255).astype(np.uint8)
        return frame_corrected
    except Exception as e:
        print(f"Erreur lors de la correction des couleurs: {str(e)}")
        return frame_masked

def corriger_image(frame_masked, cache_file, detect_squares):
    """
    Corrige les couleurs d'une image via la charte Macbeth.
    """
    global last_correction_params, frame_count
    
    try:
        # Incrémenter le compteur avant la vérification
        frame_count += 1
        
        # Vérifier si on doit recalculer les paramètres
        if frame_count % COLOR_CORRECTION_INTERVAL == 0 or last_correction_params is None or detect_squares:
            # Si detect_squares est True, force le recalcul
            colors_measured = np.array(get_average_colors(frame_masked, detect_squares))
            colors_target = np.array(MACBETH_REFERENCE_COLORS)
            
            if colors_measured.shape[0] != colors_target.shape[0]:
                raise ValueError("Erreur : Le nombre de patchs mesurés ne correspond pas au nombre de couleurs cibles (24).")
            
            # Normalisation des couleurs dans l'intervalle [0,1]
            colors_measured_norm = colors_measured / 255.0
            colors_target_norm = colors_target / 255.0
            
            # Calibration non linéaire pour obtenir les paramètres optimaux
            last_correction_params = calibrer_transformation_non_lineaire(colors_measured_norm, colors_target_norm)
            print(f"Recalcul des paramètres de correction (frame {frame_count}, detect_squares={detect_squares})")
        
        # Application de la correction à l'image complète avec les derniers paramètres
        frame_corrected = appliquer_correction_non_lineaire(frame_masked, last_correction_params)
        
        return frame_corrected
        
    except Exception as e:
        print(f"Erreur lors de la correction des couleurs: {str(e)}")
        return frame_masked

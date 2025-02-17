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

import cv2
import numpy as np
import sys
from scipy.optimize import least_squares
from macbeth_color_and_rectangle_detector import get_average_colors

def modele_non_lineaire(params, x):
    """
    Applique le modèle non linéaire de correction des couleurs.
    
    Pour chaque canal (B, G, R), applique la transformation :
    f_j(x) = (a_j * B + b_j * G + c_j * R + d_j) ** gamma_j
    
    Args:
        params (np.array): Vecteur de 15 paramètres organisés comme suit :
            - Canal B : a_b, b_b, c_b, d_b, gamma_b   (params[0:5])
            - Canal G : a_g, b_g, c_g, d_g, gamma_g   (params[5:10])
            - Canal R : a_r, b_r, c_r, d_r, gamma_r   (params[10:15])
        x (np.array): Matrice (n,3) des couleurs d'entrée en BGR
    
    Returns:
        np.array: Matrice (n,3) des couleurs transformées en BGR
    
    Notes:
        Les valeurs négatives sont évitées en appliquant un seuil minimum de 1e-6
    """
    # Paramètres pour le canal B (Blue)
    a_b, b_b, c_b, d_b, gamma_b = params[0:5]
    # Paramètres pour le canal G (Green)
    a_g, b_g, c_g, d_g, gamma_g = params[5:10]
    # Paramètres pour le canal R (Red)
    a_r, b_r, c_r, d_r, gamma_r = params[10:15]
    
    # Calcul des combinaisons linéaires pour chaque canal en BGR
    B_lin = a_b * x[:, 0] + b_b * x[:, 1] + c_b * x[:, 2] + d_b
    G_lin = a_g * x[:, 0] + b_g * x[:, 1] + c_g * x[:, 2] + d_g
    R_lin = a_r * x[:, 0] + b_r * x[:, 1] + c_r * x[:, 2] + d_r
    
    # Éviter les valeurs négatives
    B_lin = np.maximum(B_lin, 1e-6)
    G_lin = np.maximum(G_lin, 1e-6)
    R_lin = np.maximum(R_lin, 1e-6)
    
    # Application des exponents (gamma)
    B_pred = np.power(B_lin, gamma_b)
    G_pred = np.power(G_lin, gamma_g)
    R_pred = np.power(R_lin, gamma_r)
    
    return np.stack([B_pred, G_pred, R_pred], axis=1)

def calibrer_transformation_non_lineaire(measured, target):
    """
    Optimise les paramètres de la transformation non linéaire.
    
    Cette fonction détermine les 15 paramètres optimaux qui minimisent
    l'erreur entre les couleurs mesurées et les couleurs cibles.
    
    Args:
        measured (np.array): Couleurs mesurées (n,3) en BGR, normalisées [0,1]
        target (np.array): Couleurs cibles (n,3) en BGR, normalisées [0,1]
    
    Returns:
        np.array: Vecteur des 15 paramètres optimaux
    
    Notes:
        - Initialisation avec une transformation identitaire
        - Les paramètres gamma sont contraints entre 0.1 et 5
    """
    def residuals(params):
        pred = modele_non_lineaire(params, measured)
        return (pred - target).ravel()
    
    # Initialisation : transformation identitaire pour chaque canal
    # Pour le canal B : [1, 0, 0, 0, 1], pour G : [0, 1, 0, 0, 1], pour R : [0, 0, 1, 0, 1]
    x0 = np.array([1, 0, 0, 0, 1,
                   0, 1, 0, 0, 1,
                   0, 0, 1, 0, 1], dtype=np.float32)
    
    # Bornes sur les paramètres gamma pour éviter des valeurs non physiques
    lb = [-np.inf] * 15
    ub = [np.inf] * 15
    lb[4] = 0.1; lb[9] = 0.1; lb[14] = 0.1  # gamma >= 0.1
    ub[4] = 5;   ub[9] = 5;   ub[14] = 5      # gamma <= 5
    
    res = least_squares(residuals, x0, bounds=(lb, ub), method="trf")
    return res.x

def appliquer_correction_non_lineaire(image, params):
    """
    Applique la correction non linéaire à une image complète.
    
    Args:
        image (np.array): Image d'entrée en BGR (uint8)
        params (np.array): Vecteur des 15 paramètres de correction
    
    Returns:
        np.array: Image corrigée en BGR (uint8)
    
    Notes:
        Les valeurs sont automatiquement clippées dans [0,255]
    """
    h, w, _ = image.shape
    image_norm = image.astype(np.float32) / 255.0
    pixels = image_norm.reshape(-1, 3)
    pixels_corriges = modele_non_lineaire(params, pixels)
    pixels_corriges = np.clip(pixels_corriges, 0, 1)
    return (pixels_corriges.reshape(h, w, 3) * 255).astype(np.uint8)

def corriger_image(image, cache_file, detect_squares):
    """
    Corrige les couleurs d'une image via la charte Macbeth.
    
    Cette fonction principale :
    1. Récupère les couleurs moyennes des 24 patchs
    2. Compare avec les couleurs cibles standard
    3. Optimise la transformation non linéaire
    4. Applique la correction à l'image entière
    
    Args:
        image (np.array): Image d'entrée en BGR
        cache_file (str): Chemin vers le fichier de cache des positions des carrés
        detect_squares (bool): Si True, détecte les carrés, sinon utilise le cache
    
    Returns:
        np.array: Image corrigée en BGR
    
    Raises:
        ValueError: Si le nombre de patchs détectés ne correspond pas à 24
    """
    # Récupération des couleurs moyennes des 24 patchs via get_average_colors (en BGR)
    measured_colors = get_average_colors(image, cache_file, detect_squares)
    measured_colors = np.array(measured_colors, dtype=np.float32)

    # Définition des couleurs cibles standard de la charte Macbeth en BGR
    target_colors = np.array([
        [68, 82, 115], [130, 150, 194], [157, 122, 98], [67, 108, 87], [177, 128, 133],
        [170, 189, 103], [44, 126, 214], [166, 91, 80], [99, 90, 193], [108, 60, 94],
        [64, 188, 157], [46, 163, 224], [150, 61, 56], [73, 148, 70], [60, 54, 175],
        [31, 199, 231], [149, 86, 187], [161, 133, 8], [242, 243, 243], [200, 200, 200],
        [160, 160, 160], [121, 122, 122], [85, 85, 85], [52, 52, 52]
    ], dtype=np.float32)

    if measured_colors.shape[0] != target_colors.shape[0]:
        raise ValueError("Erreur : Le nombre de patchs mesurés ne correspond pas au nombre de couleurs cibles (24).")

    # Normalisation des couleurs dans l'intervalle [0,1]
    measured_norm = measured_colors / 255.0
    target_norm = target_colors / 255.0

    # Calibration non linéaire pour obtenir les paramètres optimaux
    params = calibrer_transformation_non_lineaire(measured_norm, target_norm)

    # Application de la correction à l'image complète (en BGR) et retour
    return appliquer_correction_non_lineaire(image, params)


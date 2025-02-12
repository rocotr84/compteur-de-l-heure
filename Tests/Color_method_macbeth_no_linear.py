#!/usr/bin/env python3
"""
Ce programme utilise une charte de couleur Macbeth pour corriger les couleurs d'une image
avec une transformation non linéaire. Pour chaque canal (R, G, B), la correction est définie par :
    couleur_corrigée_j = (a_j*R + b_j*G + c_j*B + d_j) ** gamma_j
Les paramètres de cette transformation (15 au total) sont déterminés via une optimisation non linéaire 
(fonction least_squares de scipy.optimize) minimisant l'erreur entre les couleurs mesurées et cibles.
Assurez-vous d'adapter les chemins d'accès à l'image et au fichier JSON selon votre environnement.
"""

import cv2
import numpy as np
import json
import sys
from scipy.optimize import least_squares

def extraire_couleur_patch(image, rect):
    """
    Extrait la couleur moyenne d'un patch défini par un rectangle sur l'image.
    
    Args:
        image (np.array): Image en format RGB.
        rect (tuple): Coordonnées (x1, y1, x2, y2) du rectangle.
    
    Returns:
        np.array: Couleur moyenne (R, G, B) sous forme de tableau float.
    """
    x1, y1, x2, y2 = rect
    patch = image[y1:y2, x1:x2]
    moyenne = cv2.mean(patch)[:3]
    return np.array(moyenne, dtype=np.float32)

def charger_rectangles(json_path):
    """
    Charge la liste des rectangles à partir d'un fichier JSON.
    
    Args:
        json_path (str): Chemin vers le fichier JSON contenant les rectangles.
    
    Returns:
        list: Liste de tuples (x1, y1, x2, y2) des 24 patchs.
    """
    try:
        with open(json_path, 'r') as f:
            rects = json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON : {e}")
        sys.exit(1)
        
    rectangles = [tuple(rect) for rect in rects]
    if len(rectangles) != 24:
        print(f"Erreur : Le fichier doit contenir 24 rectangles, trouvé {len(rectangles)}.")
        sys.exit(1)
    return rectangles

def modele_non_lineaire(params, x):
    """
    Applique le modèle non linéaire aux données.
    
    Le modèle est défini pour chaque canal (R, G, B) par :
        f_j(x) = (a_j*R + b_j*G + c_j*B + d_j) ** gamma_j
    où x est une série de valeurs [R, G, B] normalisées, et params est un vecteur de 15 paramètres :
        Pour canal R : a, b, c, d, gamma
        Pour canal G : a, b, c, d, gamma
        Pour canal B : a, b, c, d, gamma
    
    Args:
        params (np.array): Vecteur de paramètres (15,).
        x (np.array): Données d'entrée de taille (n,3), avec des valeurs normalisées dans [0,1].
    
    Returns:
        np.array: Prédictions de taille (n,3).
    """
    # Extraire les paramètres pour chaque canal
    a_r, b_r, c_r, d_r, gamma_r = params[0:5]
    a_g, b_g, c_g, d_g, gamma_g = params[5:10]
    a_b, b_b, c_b, d_b, gamma_b = params[10:15]
    
    # Calculer la combinaison linéaire pour chaque canal
    R_lin = a_r*x[:, 0] + b_r*x[:, 1] + c_r*x[:, 2] + d_r
    G_lin = a_g*x[:, 0] + b_g*x[:, 1] + c_g*x[:, 2] + d_g
    B_lin = a_b*x[:, 0] + b_b*x[:, 1] + c_b*x[:, 2] + d_b
    
    # Éviter les valeurs négatives
    R_lin = np.maximum(R_lin, 1e-6)
    G_lin = np.maximum(G_lin, 1e-6)
    B_lin = np.maximum(B_lin, 1e-6)
    
    # Appliquer l'exposant (gamma)
    R_pred = np.power(R_lin, gamma_r)
    G_pred = np.power(G_lin, gamma_g)
    B_pred = np.power(B_lin, gamma_b)
    
    predictions = np.stack([R_pred, G_pred, B_pred], axis=1)
    return predictions

def calibrer_transformation_non_lineaire(measured, target):
    """
    Calcule les paramètres de la transformation non linéaire pour corriger les couleurs.
    
    Les couleurs mesurées et cibles doivent être normalisées dans l'intervalle [0,1].
    
    Args:
        measured (np.array): Matrice (n, 3) des couleurs mesurées.
        target (np.array): Matrice (n, 3) des couleurs cibles.
    
    Returns:
        np.array: Vecteur de paramètres optimaux (15,).
    """
    def residuals(params):
        pred = modele_non_lineaire(params, measured)
        return (pred - target).ravel()
    
    # Initialisation : pour chaque canal, on part de la transformation identitaire (sans mélange)
    # Canal R : [1, 0, 0, 0, 1], Canal G : [0, 1, 0, 0, 1], Canal B : [0, 0, 1, 0, 1]
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
    Applique la correction non linéaire à toute l'image.
    
    Args:
        image (np.array): Image en format RGB (uint8).
        params (np.array): Vecteur de paramètres de transformation non linéaire (15,).
    
    Returns:
        np.array: Image corrigée en format RGB (uint8).
    """
    h, w, _ = image.shape
    image_norm = image.astype(np.float32) / 255.0
    pixels = image_norm.reshape(-1, 3)
    pixels_corriges = modele_non_lineaire(params, pixels)
    pixels_corriges = np.clip(pixels_corriges, 0, 1)
    image_corrigee = (pixels_corriges.reshape(h, w, 3) * 255).astype(np.uint8)
    return image_corrigee

def main():
    # Chemins d'accès (modifier selon votre environnement)
    image_path = r"C:\Users\victo\Desktop\correction_im_v2.png"  # Image contenant la charte Macbeth
    json_path = "rectangles_parameters.json"                      # Fichier JSON avec les coordonnées des 24 patchs
    
    # Charger l'image (OpenCV lit en BGR)
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur : impossible de charger l'image.")
        sys.exit(1)
    
    # Conversion de BGR à RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Charger les rectangles définissant les patchs
    rectangles = charger_rectangles(json_path)
    
    # Extraire les couleurs moyennes mesurées pour chaque patch
    measured_colors = []
    for rect in rectangles:
        couleur = extraire_couleur_patch(image_rgb, rect)
        measured_colors.append(couleur)
    measured_colors = np.array(measured_colors, dtype=np.float32)
    
    # Définition des couleurs cibles standard de la charte Macbeth (en RGB)
    target_colors = np.array([
        [115, 82, 68],
        [194, 150, 130],
        [98, 122, 157],
        [87, 108, 67],
        [133, 128, 177],
        [103, 189, 170],
        [214, 126, 44],
        [80, 91, 166],
        [193, 90, 99],
        [94, 60, 108],
        [157, 188, 64],
        [224, 163, 46],
        [56, 61, 150],
        [70, 148, 73],
        [175, 54, 60],
        [231, 199, 31],
        [187, 86, 149],
        [8, 133, 161],
        [243, 243, 242],
        [200, 200, 200],
        [160, 160, 160],
        [122, 122, 121],
        [85, 85, 85],
        [52, 52, 52]
    ], dtype=np.float32)
    
    if measured_colors.shape[0] != target_colors.shape[0]:
        print("Erreur : Le nombre de patchs mesurés ne correspond pas au nombre de couleurs cibles (24).")
        sys.exit(1)
    
    # Normaliser les couleurs dans l'intervalle [0,1]
    measured_norm = measured_colors / 255.0
    target_norm = target_colors / 255.0
    
    # Calibrer la transformation non linéaire
    params = calibrer_transformation_non_lineaire(measured_norm, target_norm)
    print("Paramètres de correction non linéaire :")
    print(params)
    
    # Appliquer la correction non linéaire à l'image complète
    image_corrigee = appliquer_correction_non_lineaire(image_rgb, params)
    
    # Conversion de RGB à BGR pour l'affichage avec OpenCV
    image_corrigee_bgr = cv2.cvtColor(image_corrigee, cv2.COLOR_RGB2BGR)
    
    # Affichage des images
    #cv2.imshow("Image Originale", image)
    #cv2.imshow("Image Corrigée (Non Linéaire)", image_corrigee_bgr)
    
    # Redimensionner l'image corrigée pour une largeur de 1800 pixels
    hauteur, largeur = image_corrigee_bgr.shape[:2]
    nouvelle_largeur = 1800
    ratio_redim = nouvelle_largeur / largeur
    nouvelle_hauteur = int(hauteur * ratio_redim)
    image_corrigee_resized = cv2.resize(image_corrigee_bgr, (nouvelle_largeur, nouvelle_hauteur))
    cv2.imshow("Image Corrigée Non Linéaire (Width=1800)", image_corrigee_resized)
    
    print("Appuyez sur une touche pour fermer les fenêtres...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
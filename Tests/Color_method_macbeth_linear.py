#!/usr/bin/env python3
"""
Ce programme analyse une image contenant la charte de couleur Macbeth.
Il utilise un fichier JSON (rectangles_parameters.json) contenant les 24 rectangles
définissant les positions des patchs dans l'image. Pour chaque patch, la couleur moyenne
est extraite, puis une transformation linéaire (avec biais) est calculée par moindres carrés
afin d'ajuster les couleurs mesurées aux couleurs cibles standard.

Les calculs ne sont pas optimisés car le temps de calcul n'est pas une contrainte.
Assurez-vous de bien adapter les chemins d'accès à l'image et au fichier JSON selon votre environnement.
"""

import cv2
import numpy as np
import json
import sys

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
    # Utilisation de cv2.mean qui retourne une moyenne pour chaque canal (et un alpha éventuellement)
    moyenne = cv2.mean(patch)[:3]
    return np.array(moyenne, dtype=np.float32)

def calculer_transformation(measured, target):
    """
    Calcule la matrice de transformation (avec biais) pour corriger les couleurs avec une transformation linéaire.
    
    On cherche une transformation sous la forme :
        couleur_corrigée = [R, G, B, 1] @ A   avec A de dimension (4, 3)
        
    Args:
        measured (np.array): Matrice (n, 3) des couleurs mesurées.
        target (np.array): Matrice (n, 3) des couleurs cibles.
    
    Returns:
        np.array: Matrice de transformation A (4, 3).
    """
    n = measured.shape[0]
    # On ajoute une colonne de 1 pour modéliser le biais
    X = np.hstack([measured, np.ones((n, 1))])
    # On résout X @ A = target par moindres carrés
    A, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    return A

def appliquer_correction(image, A):
    """
    Applique la transformation de correction à toute l'image.
    
    Args:
        image (np.array): Image en format RGB (uint8).
        A (np.array): Matrice de transformation (4, 3).
    
    Returns:
        np.array: Image corrigée en format RGB (uint8).
    """
    h, w, c = image.shape
    image_float = image.astype(np.float32)
    pixels = image_float.reshape(-1, 3)
    # Ajout d'une colonne de 1 pour chaque pixel
    pixels_aug = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
    pixels_corriges = pixels_aug @ A
    pixels_corriges = np.clip(pixels_corriges, 0, 255)
    image_corrigee = pixels_corriges.reshape(h, w, 3).astype(np.uint8)
    return image_corrigee

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

def main():
    # Modifiez ces chemins selon votre environnement
    image_path = r"C:\Users\victo\Desktop\correction_im_v2.png"     # Chemin vers l'image contenant la charte Macbeth
    json_path = "rectangles_parameters.json"          # Fichier JSON avec les coordonnées des 24 patchs
    
    # Charger l'image (OpenCV lit en BGR)
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur : impossible de charger l'image.")
        sys.exit(1)
    
    # Conversion de BGR à RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Charger les rectangles définissant les patchs
    rectangles = charger_rectangles(json_path)
    
    # Extraction des couleurs moyennes mesurées pour chaque patch
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
    
    # Calcul de la matrice de correction la plus précise possible
    A = calculer_transformation(measured_colors, target_colors)
    print("Matrice de correction (A) :")
    print(A)
    
    # Appliquer la correction à l'image complète
    image_corrigee = appliquer_correction(image_rgb, A)
    
    # Conversion de RGB à BGR pour l'affichage avec OpenCV
    image_corrigee_bgr = cv2.cvtColor(image_corrigee, cv2.COLOR_RGB2BGR)
    
    # Affichage de l'image originale et de l'image corrigée
    #cv2.imshow("Image Originale", image)
    #cv2.imshow("Image Corrigée", image_corrigee_bgr)
    
    # Redimensionner l'image corrigée pour avoir une largeur de 1800 pixels
    hauteur, largeur = image_corrigee_bgr.shape[:2]
    nouvelle_largeur = 1800
    ratio_redim = nouvelle_largeur / largeur
    nouvelle_hauteur = int(hauteur * ratio_redim)
    image_corrigee_resized = cv2.resize(image_corrigee_bgr, (nouvelle_largeur, nouvelle_hauteur))
    cv2.imshow("Image Corrigée (Width=1800)", image_corrigee_resized)
    
    print("Appuyez sur une touche pour fermer les fenêtres...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
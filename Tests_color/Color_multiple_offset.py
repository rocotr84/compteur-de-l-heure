import cv2
import numpy as np
import os

def compute_affine_color_matrix(reference_colors, detected_colors):
    """
    Calcule la matrice de transformation affine (3x3) pour ajuster les couleurs détectées aux couleurs de référence.
    
    Args:
        reference_colors (list of tuple): Liste des 10 couleurs de référence en RGB.
        detected_colors (list of tuple): Liste des 10 couleurs mesurées en RGB.
    
    Returns:
        np.array: Matrice de transformation 3x3 pour correction des couleurs.
    """
    # Convertir en numpy array
    ref = np.array(reference_colors, dtype=np.float32)  # (10, 3)
    det = np.array(detected_colors, dtype=np.float32)  # (10, 3)

    # Résolution de l'équation : M * detected.T = reference.T  =>  M = reference.T * detected.T^-1
    retval, transformation_matrix = cv2.solve(det, ref, flags=cv2.DECOMP_SVD)

    # Vérifie si la solution est valide
    if retval:
        return transformation_matrix
    else:
        print("Erreur : la résolution de la matrice de transformation a échoué.")
        return None

def apply_affine_color_correction(image, transformation_matrix):
    """
    Applique la correction de couleur affine à une image.

    Args:
        image (np.array): Image OpenCV en BGR.
        transformation_matrix (np.array): Matrice 3x3 de transformation des couleurs.

    Returns:
        np.array: Image corrigée.
    """
    # Normaliser l'image en float32 pour éviter la saturation
    image_float = image.astype(np.float32) / 255.0

    # Appliquer la transformation aux couleurs (image.reshape pour appliquer sur tous les pixels)
    reshaped = image_float.reshape(-1, 3)  # Convertir en liste de pixels (N, 3)
    corrected = np.dot(reshaped, transformation_matrix.T)  # Matrice 3x3 appliquée

    # Revenir à l'intervalle [0, 255]
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

    # Remettre en forme
    return corrected.reshape(image.shape)

if __name__ == '__main__':
    # -------------------------------
    # 1. Définir les couleurs de calibration
    # -------------------------------
    # Couleurs de référence (attendues) en RGB
    reference_colors = [
        (170, 136, 235),
        (34, 16, 190),
        (20, 22, 48),
        (27, 175, 215),
        (83, 144, 16),
        (103, 124, 9),
        (146, 75, 48),
        (190, 161, 35),
        (10, 28, 9),
        (202, 204, 201)
    ]



    # Couleurs détectées sur l'image en RGB
    detected_colors = [
        (120, 78, 157),
        (23, 9, 104),
        (11, 11, 20),
        (13, 141, 169),
        (39, 99, 4),
        (53, 81, 1),
        (92, 29, 15),
        (117, 91, 6),
        (3, 21, 1),
       (181, 176, 173)
    ]

    # Calcul de la matrice de transformation
    transformation_matrix = compute_affine_color_matrix(reference_colors, detected_colors)
    if transformation_matrix is not None:
        print("Matrice de correction des couleurs :\n", transformation_matrix)
    else:
        print("❌ Impossible de calculer la matrice de transformation.")
        exit()

    # -------------------------------
    # 2. Charger l'image à corriger
    # -------------------------------
    image_path = r"C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\camera4K2\pink\Face\frame_0222.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erreur : Impossible de charger l'image. Vérifie le chemin.")
    else:
        # -------------------------------
        # 3. Appliquer la correction des couleurs
        # -------------------------------
        corrected_image = apply_affine_color_correction(image, transformation_matrix)

        # -------------------------------
        # 4. Sauvegarder l'image corrigée
        # -------------------------------
        folder = os.path.dirname(image_path)
        extension = os.path.splitext(image_path)[1]  # Ex. ".jpg"
        new_filename = "correction_im" + extension
        new_path = os.path.join(folder, new_filename)

        if cv2.imwrite(new_path, corrected_image):
            print("✅ Image corrigée sauvegardée sous :", new_path)
        else:
            print("❌ Erreur lors de la sauvegarde.")

        # Optionnel : affichage des images
        cv2.imshow("Image Originale", image)
        cv2.imshow("Image Corrigee", corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



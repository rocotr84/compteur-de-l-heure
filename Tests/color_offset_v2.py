import cv2
import numpy as np
import os

def compute_color_offset(reference_color, detected_color):
    """
    Calcule l'offset de couleur pour ajuster la couleur détectée à la couleur de référence.
    
    Args:
        reference_color (tuple): Couleur de référence en RGB.
        detected_color (tuple): Couleur mesurée en RGB.
    
    Returns:
        tuple: Offset de couleur en RGB.
    """
    ref = np.array(reference_color, dtype=np.float32)
    det = np.array(detected_color, dtype=np.float32)
    offset = ref - det
    return offset

def apply_color_offset(image, color_offset):
    """
    Applique l'offset de couleur à une image.

    Args:
        image (np.array): Image OpenCV en BGR.
        color_offset (tuple): Offset de couleur en RGB.

    Returns:
        np.array: Image corrigée.
    """
    # Normaliser l'image en float32 pour éviter la saturation
    image_float = image.astype(np.float32) / 255.0

    # Appliquer l'offset aux couleurs
    offset_array = np.array(color_offset, dtype=np.float32) / 255.0
    corrected = image_float + offset_array

    # Revenir à l'intervalle [0, 255]
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

    return corrected

if __name__ == '__main__':
    # -------------------------------
    # 1. Définir les couleurs de calibration
    # -------------------------------
    # Couleur de référence (blanc attendu) en RGB
    reference_color = (202, 205, 203)

    # Couleur détectée sur l'image en RGB
    detected_color = (175, 178, 183)  # Exemple de couleur détectée

    # Calcul de l'offset de couleur
    color_offset = compute_color_offset(reference_color, detected_color)
    print("Offset de couleur :\n", color_offset)

    # -------------------------------
    # 2. Charger l'image à corriger
    # -------------------------------
    image_path = r"C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\camera4K\Toute les couleurs\frame_0354.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erreur : Impossible de charger l'image. Vérifie le chemin.")
    else:
        # -------------------------------
        # 3. Appliquer la correction des couleurs
        # -------------------------------
        corrected_image = apply_color_offset(image, color_offset)

        # -------------------------------
        # 4. Sauvegarder l'image corrigée
        # -------------------------------
        folder = os.path.dirname(image_path)
        extension = os.path.splitext(image_path)[1]  # Ex. ".jpg"
        new_filename = "correction_im_v2" + extension
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
import cv2
import numpy as np
import os

def compute_global_offset(reference_colors, detected_colors):
    """
    Calcule l'offset global (en RGB) à appliquer sur l'image à partir de 6 paires de couleurs.
    
    Args:
        reference_colors (list of tuple): Liste des 6 couleurs de référence en RGB.
        detected_colors (list of tuple): Liste des 6 couleurs mesurées en RGB.
    
    Returns:
        np.array: Offset global moyen par canal (R, G, B) sous forme d'un tableau numpy.
    """
    # Convertir les listes en tableaux (forme (6, 3))
    ref = np.array(reference_colors, dtype=np.int16)
    det = np.array(detected_colors, dtype=np.int16)
    
    # Calculer l'offset pour chaque patch
    offsets = ref - det  # Chaque ligne correspond à (offset_R, offset_G, offset_B)
    
    # Calculer l'offset moyen par canal
    global_offset = np.mean(offsets, axis=0)
    
    # Arrondir pour obtenir des valeurs entières
    return np.round(global_offset).astype(np.int16)

def apply_offset(image, offset_rgb):
    """
    Applique un offset couleur à toute l'image.
    
    Args:
        image (np.array): Image en BGR (typique de OpenCV).
        offset_rgb (np.array): Offset en RGB à appliquer.
    
    Returns:
        np.array: Image corrigée.
    """
    # Conversion de l'offset depuis RGB vers BGR (inverser l'ordre)
    offset_bgr = offset_rgb[::-1]
    
    # Conversion de l'image en int16 pour éviter les dépassements
    image_int = image.astype(np.int16)
    
    # Appliquer l'offset (broadcasting sur tous les pixels)
    corrected = image_int + offset_bgr
    
    # Reconvertir dans l'intervalle [0, 255] en uint8
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected

if __name__ == '__main__':
    # -------------------------------
    # 1. Définir les couleurs de calibration
    # -------------------------------
    # Exemple : remplacez ces valeurs par vos mesures réelles.
    # Couleurs de référence (en RGB) attendues pour chacun des 6 patches.
    reference_colors = [
        (244, 133, 176),
        (199, 10, 38),
        (43, 10, 17),
        (224, 172, 27),
        (18, 137, 97),
        (44, 58, 147),
        (38, 152, 212),
        (9, 11, 10)
    ]
    
    # Couleurs détectées (en RGB) mesurées sur l'image de calibration.
    detected_colors = [
        (167, 73, 126),
        (106, 0, 22),
        (20, 1, 7),
        (173, 133, 9),
        (173, 133, 9),
        (14, 16, 93),
        (6, 76, 128),
        (1, 1, 3)
    ]
    
    # Calcul de l'offset global en RGB
    offset_rgb = compute_global_offset(reference_colors, detected_colors)
    print("Offset RGB global calculé :", offset_rgb)
    
    # -------------------------------
    # 2. Charger l'image à corriger
    # -------------------------------
    # Spécifiez le chemin de l'image que vous souhaitez corriger.
    image_path = r"C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\camera4K\Toute les couleurs\frame_0352.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur lors du chargement de l'image. Vérifiez le chemin et l'extension.")
    else:
        # -------------------------------
        # 3. Appliquer l'offset global à l'image
        # -------------------------------
        corrected_image = apply_offset(image, offset_rgb)
        
        # -------------------------------
        # 4. Sauvegarder l'image corrigée
        # -------------------------------
        # Sauvegarde dans le même dossier sous le nom "correction_im" en conservant l'extension d'origine.
        folder = os.path.dirname(image_path)
        extension = os.path.splitext(image_path)[1]  # Par exemple ".jpg" ou ".png"
        new_filename = "correction_im" + extension
        new_path = os.path.join(folder, new_filename)
        
        if cv2.imwrite(new_path, corrected_image):
            print("Image corrigée sauvegardée sous :", new_path)
        else:
            print("Erreur lors de la sauvegarde de l'image.")
        
        # Optionnel : affichage des images
        cv2.imshow("Image Originale", image)
        cv2.imshow("Image Corrigee", corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2
import numpy as np
import os

def calculer_offset(reference_rgb, detected_rgb):
    """
    Calcule l'offset à appliquer (en RGB) pour corriger la couleur détectée vers la couleur de référence.
    
    Paramètres :
      reference_rgb (tuple) : La couleur de référence en RGB (ex. (37, 146, 202)).
      detected_rgb (tuple)  : La couleur détectée en RGB (ex. (3, 58, 97)).
    
    Retourne :
      offset_rgb (np.array) : L'offset par canal en RGB.
    """
    ref = np.array(reference_rgb, dtype=np.int16)
    det = np.array(detected_rgb, dtype=np.int16)
    offset_rgb = ref - det
    return offset_rgb

def appliquer_offset(image, offset_rgb):
    """
    Applique l'offset à l'image pour corriger la couleur.
    
    Paramètres :
      image (np.array)      : Image chargée (en BGR, typiquement en uint8).
      offset_rgb (np.array) : Offset calculé en RGB.
    
    Retourne :
      image_corrigee (np.array) : L'image corrigée.
    """
    # Conversion de l'offset en RGB vers l'offset en BGR (OpenCV utilise le format BGR)
    offset_bgr = offset_rgb[::-1]  # Inverse l'ordre (R, G, B) -> (B, G, R)
    
    # Conversion de l'image en int16 pour éviter les débordements lors de l'addition
    image_int = image.astype(np.int16)
    # Application de l'offset à chaque pixel
    image_corrigee = image_int + offset_bgr
    # On restreint les valeurs à l'intervalle [0, 255] et on repasse en uint8
    image_corrigee = np.clip(image_corrigee, 0, 255).astype(np.uint8)
    return image_corrigee

if __name__ == '__main__':
    # Couleurs données en RGB
    reference_rgb = (37, 146, 202)
    detected_rgb  = (3, 58, 97)
    
    # Calcul de l'offset en RGB
    offset_rgb = calculer_offset(reference_rgb, detected_rgb)
    print("Offset RGB calculé :", offset_rgb)  # Affichera : [34 88 105]
    
    # Chemin de l'image à modifier.
    # Veillez à bien indiquer l'extension (ici, on prend l'exemple d'une image .jpg)
    image_path = r"C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\camera4K\Toute les couleurs\frame_0350.jpg"
    
    # Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur lors du chargement de l'image. Vérifiez le chemin et l'extension.")
    else:
        # Application de l'offset pour corriger l'image
        image_corrigee = appliquer_offset(image, offset_rgb)
        
        # Sauvegarder l'image corrigée dans le même dossier avec le nom "correction_im"
        dossier = os.path.dirname(image_path)
        extension = os.path.splitext(image_path)[1]  # Récupère l'extension (ex. ".jpg")
        nouveau_chemin = os.path.join(dossier, "correction_im" + extension)
        cv2.imwrite(nouveau_chemin, image_corrigee)
        print("Image corrigée sauvegardée sous :", nouveau_chemin)
        
        # Affichage de l'image originale et de l'image corrigée
        cv2.imshow("Image Originale", image)
        cv2.imshow("Image Corrigee", image_corrigee)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

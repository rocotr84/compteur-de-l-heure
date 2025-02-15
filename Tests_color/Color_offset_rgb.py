import cv2
import numpy as np
import os

def main():
    # 1. Chemin de l'image
    image_path = r"C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\136_2401\couleurs\IMG_3697.jpg"
    #image_path = r"C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\camera4K\Toute les couleurs\correction_im_v2.jpg"
   
    # Vérifier que le fichier existe
    if not os.path.exists(image_path):
        print("Erreur : Fichier introuvable.")
        return

    # 3. Définition des plages RGB (nom : (R, G, B))
    color_ranges = {
        'pink': (234, 137, 169),
        'red': (189, 17, 34),
        'burgundy': (128, 0, 32),
        'yellow': (255, 255, 0),
        'light_green': (144, 238, 144),
        'dark_green': (0, 100, 0),
        'dark_blue': (47, 77, 145),
        'light_blue': (35, 162, 187),
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    
    # 4. Traitement de l'image
    print(f"Fichier trouvé : {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur : Impossible de charger l'image {image_path}.")
        return

    # Redimensionner l'image pour qu'elle s'adapte mieux à l'écran
    scale_percent = 50  # Pourcentage de redimensionnement
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Afficher l'image redimensionnée
    cv2.imshow("Image", resized_img)

    while True:
        # 5. Sélection de la ROI par l'utilisateur
        roi = cv2.selectROI("Sélectionnez la ROI", resized_img, fromCenter=False, showCrosshair=True)
        x1, y1, w, h = roi
        x2, y2 = x1 + w, y1 + h

        if w == 0 or h == 0:
            print("Erreur : Aucune ROI sélectionnée.")
            continue

        # Ajuster les coordonnées de la ROI à l'image originale
        x1_orig = int(x1 / scale_percent * 100)
        y1_orig = int(y1 / scale_percent * 100)
        x2_orig = int(x2 / scale_percent * 100)
        y2_orig = int(y2 / scale_percent * 100)

        roi = img[y1_orig:y2_orig, x1_orig:x2_orig]  # Extraction de la région d'intérêt (ROI)

        # 6. Calcul de la moyenne des pixels dans la ROI (en RGB)
        moyenne_pixel = np.mean(roi.reshape(-1, 3), axis=0)
        print(f"Moyenne des pixels en RGB : R: {moyenne_pixel[2]:.2f}, G: {moyenne_pixel[1]:.2f}, B: {moyenne_pixel[0]:.2f}\n")

        # 7. Détection de la couleur dominante dans la ROI
        best_match = None
        min_distance = float('inf')

        for color_name, rgb_value in color_ranges.items():
            bgr_value = (rgb_value[2], rgb_value[1], rgb_value[0])  # Convertir RGB en BGR
            distance = np.linalg.norm(moyenne_pixel - np.array(bgr_value))
            if distance < min_distance:
                min_distance = distance
                best_match = color_name

        print(f"Couleur la plus proche en RGB pour {image_path} : {best_match} (distance : {min_distance})\n")

        # Dessiner le rectangle rouge autour de la ROI sur l'image redimensionnée
        cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rouge (BGR: 0,0,255)
        cv2.imshow("Image", resized_img)

        # Appuyer sur 'q' pour quitter la boucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

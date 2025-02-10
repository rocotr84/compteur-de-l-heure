import cv2
import numpy as np
import os

def main():
    # 1. Dossier contenant les images
    image_folder = r"C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\appareil_photo_cropt\pink\face"
    
    # Vérifier que le dossier existe
    if not os.path.exists(image_folder):
        print("Erreur : Dossier introuvable.")
        return

    # 2. Définition du rectangle central (50x50)
    rect_size = 50
    
    # 3. Définition des plages HSV (nom : (borne basse, borne haute))
    # yellow, dark blue, dark, light green ok
    color_ranges = {
        "noir": ((0, 0, 0), (180, 255, 50)),
        "blanc": ((0, 0, 200), (180, 30, 255)),
        "rouge_fonce": ((0, 50, 50), (10, 255, 255)),
        "bleu_fonce": ((100, 50, 50), (130, 255, 120)),  # Dark blue
        "bleu_clair": ((100, 50, 121), (130, 255, 255)),  # Light blue
        "vert_fonce": ((35, 50, 50), (85, 255, 255)),  # Dark green
        "rose": ((140, 50, 50), (170, 255, 255)),
        "jaune": ((20, 100, 100), (40, 255, 255)),
        "vert_clair": ((40, 50, 50), (80, 255, 255))  # Light green
    }
    
    # 4. Traitement de chaque image du dossier
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        
        # Vérifier si c'est bien un fichier image
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".jpeg")):
            continue
        
        print(f"Fichier trouvé : {filename}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur : Impossible de charger l'image {filename}.")
            continue

        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        x1, y1 = max(center_x - rect_size // 2, 0), max(center_y - rect_size // 2, 0)
        x2, y2 = min(center_x + rect_size // 2, width), min(center_y + rect_size // 2, height)
        
        roi = img[y1:y2, x1:x2]  # Extraction de la région d'intérêt (ROI)

        # 5. Conversion en HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Afficher les valeurs des pixels en HSV dans la console
        print(f"Valeurs des pixels en HSV pour {filename} :")
        for row in hsv_roi:
            for pixel in row:
                print(f"H: {pixel[0]}, S: {pixel[1]}, V: {pixel[2]}")
            print()
        print("Fin des valeurs HSV\n")
        
        # Calcul de la moyenne des pixels dans la ROI (en HSV)
        # On aplatit la matrice 2D en une liste de pixels (chaque pixel est un vecteur [H,S,V])
        moyenne_pixel = np.mean(hsv_roi.reshape(-1, 3), axis=0)
        print(f"Moyenne des pixels en HSV pour {filename} : H: {moyenne_pixel[0]:.2f}, S: {moyenne_pixel[1]:.2f}, V: {moyenne_pixel[2]:.2f}\n")

        # 6. Détection de la couleur dominante dans la ROI
        best_match = None
        max_count = 0
        roi_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]

        for color_name, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
            count = cv2.countNonZero(mask)
            print(f"{filename} - {color_name} : {count} pixels dans la ROI")
            if count > max_count:
                max_count = count
                best_match = color_name

        # Seuil : au moins 30% des pixels doivent correspondre à une couleur
        threshold = int(0.3 * roi_pixels)
        if max_count >= threshold:
            print(f"\n🎯 {filename} - Couleur détectée : {best_match} ({max_count} pixels sur {roi_pixels})")
        else:
            print(f"\n⚠️ {filename} - Aucune couleur significative détectée dans la ROI.")
            best_match = None

        # 7. Dessiner le rectangle rouge autour de la ROI sur l'image complète
        img_with_roi = img.copy()
        cv2.rectangle(img_with_roi, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rouge (BGR: 0,0,255)

        # 8. Affichage des résultats
        cv2.imshow(f"Image avec ROI - {filename}", img_with_roi)
        cv2.imshow("ROI (50x50 au centre)", roi)

        # Affichage du masque de la couleur détectée
        if best_match:
            mask = cv2.inRange(hsv_roi, np.array(color_ranges[best_match][0], dtype=np.uint8),
                               np.array(color_ranges[best_match][1], dtype=np.uint8))
            cv2.imshow(f"Masque {best_match} - {filename}", mask)

        cv2.waitKey(5000)  # Affiche l'image pendant 5 secondes
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import json

def downscale_image(image, max_width=1800):
    """
    Réduit la taille de l'image pour l'affichage,
    et retourne également le facteur de réduction appliqué.
    """
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_dim = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_dim), scale
    return image.copy(), 1.0

def selection_roi(image, window_name="Sélectionnez ROI"):
    """
    Permet à l'utilisateur de sélectionner une ROI (région d'intérêt) sur l'image.
    
    Args:
        image (np.array): L'image sur laquelle sélectionner la ROI.
        window_name (str): Titre de la fenêtre de sélection.
    
    Returns:
        tuple: (x, y, w, h) avec (x, y) le coin supérieur gauche,
               w et h la largeur et la hauteur de la ROI.
    """
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    return roi  # (x, y, w, h)

def dessiner_rectangles(image, rectangles):
    """
    Dessine sur une copie de l'image tous les rectangles définis.
    
    Args:
        image (np.array): Image originale.
        rectangles (list): Liste de rectangles sous forme (x1, y1, x2, y2).
    
    Returns:
        np.array: Image avec les rectangles dessinés.
    """
    img_copy = image.copy()
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_copy, str(i+1), (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img_copy

def main():
    # Charger l'image contenant la charte (à adapter selon votre environnement)
    image_path = r"C:\Users\victo\Desktop\IMG_3661.png"
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur : impossible de charger l'image.")
        return

    # Réduire la taille de l'image pour l'affichage interactif
    display_img, scale = downscale_image(image, max_width=1800)

    # Sélection interactive du premier patch (ligne 1, colonne 1)
    print("Sélectionnez le patch : ligne 1, colonne 1")
    roi1_small = selection_roi(display_img, "Sélectionnez le patch (ligne 1, colonne 1)")
    # Conversion des coordonnées vers l'échelle originale
    roi1 = (int(roi1_small[0] / scale), int(roi1_small[1] / scale),
            int(roi1_small[2] / scale), int(roi1_small[3] / scale))
    x0, y0, w, h = roi1

    # Sélection interactive du patch (ligne 2, colonne 1) pour déterminer l'offset vertical
    print("Sélectionnez le patch : ligne 2, colonne 1")
    roi2_small = selection_roi(display_img, "Sélectionnez le patch (ligne 2, colonne 1)")
    roi2 = (int(roi2_small[0] / scale), int(roi2_small[1] / scale),
            int(roi2_small[2] / scale), int(roi2_small[3] / scale))
    # L'offset vertical correspond à la différence d'ordonnée (y) entre la première et la deuxième sélection
    vertical_offset = roi2[1] - y0

    # Sélection interactive du patch (ligne 1, colonne 2) pour déterminer l'offset horizontal
    print("Sélectionnez le patch : ligne 1, colonne 2")
    roi3_small = selection_roi(display_img, "Sélectionnez le patch (ligne 1, colonne 2)")
    roi3 = (int(roi3_small[0] / scale), int(roi3_small[1] / scale),
            int(roi3_small[2] / scale), int(roi3_small[3] / scale))
    # L'offset horizontal correspond à la différence d'abscisse (x) entre le premier patch et ce patch
    horizontal_offset = roi3[0] - x0

    # Déduction automatique de la grille. Supposons ici une charte en 4 lignes et 6 colonnes.
    rows, cols = 4, 6
    rectangles = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * horizontal_offset
            y = y0 + r * vertical_offset
            rect = (int(x), int(y), int(x + w), int(y + h))
            rectangles.append(rect)

    # Exportation des paramètres de positionnement sous forme de liste
    print("Liste des rectangles (format: [x1, y1, x2, y2]) :")
    print(rectangles)

    # Sauvegarde en format JSON (conversion des tuples en listes)
    rectangles_json = [list(rect) for rect in rectangles]
    with open("rectangles_parameters.json", "w") as f:
        json.dump(rectangles_json, f, indent=4)
    print("Les paramètres de positionnement des rectangles ont été sauvegardés dans 'rectangles_parameters.json'.")

    # Afficher l'image avec la grille déduite (en réduisant la taille pour l'affichage)
    image_with_rectangles = dessiner_rectangles(image, rectangles)
    image_with_rectangles_display, _ = downscale_image(image_with_rectangles, max_width=1800)
    cv2.imshow("Grille de patches (méthode interactive)", image_with_rectangles_display)
    print("Appuyez sur une touche pour fermer la fenêtre...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
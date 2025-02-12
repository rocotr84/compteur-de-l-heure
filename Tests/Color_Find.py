import cv2
import numpy as np
from pathlib import Path

# Variables globales pour la gestion des rectangles
rect_start = None
rect_end = None
drawing = False
rectangles = []  # Liste pour stocker les coordonnées des rectangles

# Nom du dossier du projet à rechercher
PROJECT_NAME = "compteur-de-l-heure"

def find_project_root(current_path: Path, project_name: str) -> Path:
    """
    Remonte dans l'arborescence à partir de current_path jusqu'à trouver un dossier
    dont le nom correspond à project_name.
    """
    for parent in current_path.parents:
        if parent.name == project_name:
            return parent
    raise Exception(f"Répertoire racine du projet '{project_name}' non trouvé.")

# Récupérer le chemin absolu du fichier courant, et trouver le répertoire racine du projet
current_file = Path(__file__).resolve()
project_root = find_project_root(current_file, PROJECT_NAME)

# Construire le chemin de l'image en se basant sur la racine trouvée
#image_path = str(project_root / "assets" / "photos" / "136_2401" / "couleurs" / "IMG_3697.jpg")
#image_path = str(project_root / "assets" / "photos" / "camera4K" / "Toute les couleurs" / "frame_0352.jpg")
#image_path = str(project_root / "assets" / "photos" / "136_2401" / "Red" / "Face" / "IMG_3673.jpg")
image_path = str(project_root / "assets" / "photos" / "camera4K" / "Red" / "Face" / "frame_0181.jpg")

print("Chemin de l'image :", image_path)

# Fonction de callback pour la sélection de la zone avec la souris
def select_region(event, x, y, flags, param):
    global rect_start, rect_end, drawing, rectangles
    image = param  # L'image passée en paramètre à la fonction

    if event == cv2.EVENT_LBUTTONDOWN:
        # Lorsque l'utilisateur clique avec le bouton gauche de la souris
        rect_start = (x, y)  # Point de départ du rectangle
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # Lorsque la souris se déplace
        if drawing:
            rect_end = (x, y)  # Point final du rectangle
            temp_image = image.copy()
            # Dessiner tous les rectangles sélectionnés
            for rect in rectangles:
                cv2.rectangle(temp_image, rect[0], rect[1], (0, 255, 0), 2)
            cv2.rectangle(temp_image, rect_start, rect_end, (0, 255, 0), 2)  # Dessiner le rectangle en cours
            cv2.imshow("Sélectionner une zone", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        # Lorsque l'utilisateur relâche le bouton gauche
        rect_end = (x, y)
        drawing = False
        rectangles.append((rect_start, rect_end))  # Ajouter le rectangle à la liste
        cv2.rectangle(image, rect_start, rect_end, (0, 255, 0), 2)  # Dessiner le rectangle final
        cv2.imshow("Sélectionner une zone", image)

        # Calculer et afficher la couleur moyenne du rectangle sélectionné
        avg_rgb = get_average_rgb(image, [(rect_start, rect_end)])
        print(f"Rectangle sélectionné : {avg_rgb}")

def get_average_rgb(image, rectangles):
    """
    Calcule la valeur moyenne des pixels dans chaque zone sélectionnée.
    
    Args:
        image (np.array): L'image sur laquelle les zones ont été sélectionnées.
        rectangles (list): Liste des rectangles (coordonnées de départ et d'arrivée) à analyser.
    
    Returns:
        list: Liste des valeurs moyennes RGB pour chaque rectangle.
    """
    avg_colors = []
    for rect_start, rect_end in rectangles:
        # Définir les coordonnées de la zone sélectionnée
        x1, y1 = rect_start
        x2, y2 = rect_end
        
        # Extraire la région sélectionnée de l'image
        selected_region = image[y1:y2, x1:x2]

        # Calculer la moyenne des pixels dans la région
        avg_color_per_row = np.mean(selected_region, axis=0)  # Moyenne par ligne
        avg_color = np.mean(avg_color_per_row, axis=0)  # Moyenne globale
        avg_color_rgb = avg_color[::-1]  # Inverser l'ordre des canaux pour obtenir RGB
        avg_colors.append(tuple(np.round(avg_color_rgb).astype(int)))
    return avg_colors

if __name__ == '__main__':
    # Charger l'image
    image = cv2.imread(image_path)

    if image is None:
        print("Erreur : Impossible de charger l'image.")
    else:
        # Afficher l'image et permettre la sélection de la zone
        cv2.imshow("Sélectionner une zone", image)
        cv2.setMouseCallback("Sélectionner une zone", select_region, image)
        
        # Attendre que l'utilisateur termine la sélection
        print("Sélectionne plusieurs zones et appuie sur 'Esc' pour finir.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2
import numpy as np
import time
from pathlib import Path
import csv
from ultralytics import YOLO
import torch

# Configuration du projet
PROJECT_NAME = "compteur-de-l-heure"
CURRENT_FILE = Path(__file__).resolve()
FORCE_CPU = False  # Mettre à True pour forcer l'utilisation du CPU, False pour utiliser CUDA si disponible

# Paramètres de détection
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASS = 0  # Classe pour les personnes
MAX_PERSONS = 5  # Nombre maximum de personnes à détecter

# Ajouter la liste des modèles au début du fichier, après les autres configurations
YOLO_MODELS = [
    "yolov8n.pt",
    "yolov8x.pt",
    "yolov9t.pt",
    "yolov9e.pt",
    "yolov10n.pt",
    "yolov10x.pt",
    "yolo11n.pt", 
    "yolo11x.pt",
    "yolo11s.pt",
]

# Configuration des chemins
def get_paths():
    project_root = find_project_root(CURRENT_FILE, PROJECT_NAME)
    return {
        'models_dir': project_root / "assets" / "models",
        'images': project_root / "assets" / "photos" / "camera4K" / "2 couleurs" / "dur",
        'results': project_root / "Tests_detection" / "results.csv"
    }

# Configuration CSV
CSV_HEADER = [
    'Method', 
    'Model', 
    'Number of Images Tested',
    'Detection %', 
    'Total Detection Time (s)', 
    'Average Detection Time (s)',
    'Number of Persons Detected'
]

def find_project_root(current_path: Path, project_name: str) -> Path:
    """
    Remonte dans l'arborescence à partir de current_path jusqu'à trouver un dossier
    dont le nom correspond à project_name.
    """
    for parent in current_path.parents:
        if parent.name == project_name:
            return parent
    raise Exception(f"Répertoire racine du projet '{project_name}' non trouvé.")

def detect_person_yolo(image_path, model, confidence_threshold=0.5):
    """
    Ouvre l'image, réalise la détection des personnes via YOLO et retourne :
      - le pourcentage de confiance moyen des détections (en %),
      - le temps de détection pure YOLO,
      - le nombre de personnes détectées.
    """
    # Lire l'image (hors mesure de temps)
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Impossible d'ouvrir l'image: {image_path}")

    # Mesure uniquement le temps de détection YOLO
    start_time = time.perf_counter()  # Plus précis que time.time()
    results = model(img)
    detection_time = time.perf_counter() - start_time

    # Traitement des résultats (hors mesure de temps)
    result = results[0]
    if result.boxes is not None:
        device = 'cpu' if FORCE_CPU else ('cuda' if torch.cuda.is_available() else 'cpu')
        boxes = result.boxes.xyxy.to(device).numpy() if device == 'cpu' else result.boxes.xyxy.cuda().cpu().numpy()
        confidences = result.boxes.conf.to(device).numpy() if device == 'cpu' else result.boxes.conf.cuda().cpu().numpy()
        classes = result.boxes.cls.to(device).numpy() if device == 'cpu' else result.boxes.cls.cuda().cpu().numpy()

        # Filtrer uniquement les personnes (classe 0)
        person_detections = [(conf, box) for conf, box, cls in zip(confidences, boxes, classes) if int(cls) == 0 and conf > confidence_threshold]
        
        if person_detections:
            # Calculer la confiance pour toutes les détections sans filtrage
            person_detections.sort(key=lambda x: x[0], reverse=True)

            # Dessiner toutes les boîtes détectées
            for confidence, box in person_detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Personne {confidence*100:.1f}%"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Afficher l'image
            #cv2.imshow("Detection", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Calculer la confiance moyenne
            avg_confidence = sum(conf for conf, _ in person_detections) / len(person_detections) * 100
            return avg_confidence, detection_time, len(person_detections)

    return 0, detection_time, 0

def detecter_personnes(model, image):
    """
    Utilise le modèle YOLO (Ultralytics) pour détecter les personnes dans une image.
    Retournant les boîtes englobantes, les scores et les classes.
    """
    results = model(image)  # Traitement de l'image
    # On supposera que le modèle retourne un seul résultat (pour l'image)
    result = results[0]
    if result.boxes is None:
        return [], [], []
    boxes = result.boxes.xyxy.cpu().numpy()  # Coordonnées [x1, y1, x2, y2]
    scores = result.boxes.conf.cpu().numpy()   # Scores de confiance
    classes = result.boxes.cls.cpu().numpy()   # Indices de classe
    return boxes, scores, classes

def afficher_resultat(image, boxes, scores, classes, target_class=0):
    """
    Dessine une boîte (rectangle) autour de chaque personne détectée et affiche l'image.
    La classe cible 'personnes' correspond ici à l'indice 0 (selon l'ordre COCO).
    """
    for i, cls in enumerate(classes):
        if int(cls) == target_class:
            x1, y1, x2, y2 = boxes[i]
            # Conversion en entier
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Personne {scores[i]*100:.1f}%"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Détection avec YOLO11x", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Vérifier si CUDA est disponible
    device = 'cpu' if FORCE_CPU else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")

    # Récupérer les chemins
    paths = get_paths()
    
    # Charger la liste des images
    liste_images = list(paths['images'].glob("*.jpg")) + list(paths['images'].glob("*.png"))
    if not liste_images:
        print("Aucune image trouvée dans", paths['images'])
        exit(1)
    chemin_image = str(liste_images[1])
    print("Image utilisée:", chemin_image)

    # Boucle sur chaque modèle
    for model_name in YOLO_MODELS:
        model_path = paths['models_dir'] / model_name
        
        # Vérifier si le modèle existe
        if not model_path.exists():
            print(f"Le modèle {model_name} n'existe pas dans {paths['models_dir']}")
            continue
            
        print(f"\nTest avec le modèle: {model_name}")
        print(f"Chargement du modèle depuis {model_path}...")
        
        try:
            # Charger le modèle YOLO
            model = YOLO(str(model_path))
            model.to(device)

            # Faire une détection "à vide" pour le warmup
            print("Warmup du modèle...")
            _ = model(cv2.imread(chemin_image))
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            # Effectuer la vraie détection
            confidence, detection_time, nb_persons = detect_person_yolo(
                chemin_image, 
                model, 
                confidence_threshold=CONFIDENCE_THRESHOLD
            )
            
            print(f"Temps de détection: {detection_time:.4f} secondes")
            print(f"Confiance: {confidence:.2f}%")
            print(f"Nombre de personnes détectées: {nb_persons}")

            # Stocker les résultats dans le fichier CSV
            output_csv_path = Path(paths['results'])
            file_exists = output_csv_path.exists()
            
            with open(paths['results'], mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(CSV_HEADER)
                writer.writerow([
                    "dur",
                    model_name.replace('.pt', ''),
                    f"{confidence:.2f}",
                    f"{detection_time:.4f}",
                    nb_persons
                ])
                
        except Exception as e:
            print(f"Erreur lors du traitement du modèle {model_name}: {str(e)}")
            continue

    print("\nTests terminés. Résultats enregistrés dans:", paths['results'])

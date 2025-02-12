import cv2
import os
import glob

def save_frames(video_path, output_dir, frame_interval=20):
    """
    Sauvegarde une frame sur 'frame_interval' de la vidéo.
    Continue la numérotation à partir des images existantes.
    
    Args:
        video_path (str): Chemin vers la vidéo source
        output_dir (str): Dossier de destination des images
        frame_interval (int): Nombre de frames à sauter entre chaque sauvegarde
    """
    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Trouver le dernier numéro utilisé
    existing_frames = glob.glob(os.path.join(output_dir, "frame_*.jpg"))
    if existing_frames:
        last_num = max([int(f.split("_")[-1].split(".")[0]) for f in existing_frames])
        saved_count = last_num + 1
        print(f"Continuation à partir de l'image {saved_count}")
    else:
        saved_count = 0
        print("Démarrage d'une nouvelle séquence")
    
    # Ouverture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return
    
    frame_count = 0
    
    print("Démarrage de l'extraction des frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sauvegarde une frame sur frame_interval
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            if saved_count % 10 == 0:  # Affiche un statut tous les 10 sauvegardes
                print(f"Images sauvegardées: {saved_count}")
                
        frame_count += 1
    
    cap.release()
    print(f"\nTerminé! {saved_count - (last_num + 1 if existing_frames else 0)} nouvelles images ont été sauvegardées dans {output_dir}")
    print(f"Frames totales traitées: {frame_count}")
    print(f"Nombre total d'images dans le dossier: {saved_count}")

if __name__ == "__main__":
    # Chemins
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(PROJECT_ROOT, "compteur-de-l-heure", "assets", "marathon_2.mp4")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "compteur-de-l-heure", "assets", "frames")
    
    # Vérification de l'existence de la vidéo
    if not os.path.exists(VIDEO_PATH):
        print(f"Erreur: Le fichier vidéo {VIDEO_PATH} n'existe pas!")
        print(f"Chemin attendu: {VIDEO_PATH}")
    else:
        save_frames(VIDEO_PATH, OUTPUT_DIR, frame_interval=20)  # Utilise l'intervalle de 20 frames 
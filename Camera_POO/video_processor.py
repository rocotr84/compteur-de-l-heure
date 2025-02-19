import cv2
import numpy as np
import time
from config import config
from tracker import ObjectTracker
from macbeth_detector import MacbethDetector
from color_correction import ColorCorrection

class VideoProcessor:
    """Classe gérant le traitement vidéo et l'affichage"""
    def __init__(self):
        self.tracker = ObjectTracker()
        self.video_capture = None
        self.video_writer = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.crossing_counts = {"up": 0, "down": 0}
        
        # Nouveaux composants pour Macbeth
        self.macbeth_detector = MacbethDetector()
        self.color_correction = ColorCorrection()
        self.color_correction_count = 0  # Compteur spécifique pour la correction des couleurs
        
        self.initialize_video()

    def initialize_video(self):
        """Initialise la capture vidéo et le writer si nécessaire"""
        self.video_capture = cv2.VideoCapture(config.VIDEO_INPUT_PATH)
        
        if config.SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
            self.video_writer = cv2.VideoWriter(
                config.VIDEO_OUTPUT_PATH,
                fourcc,
                config.VIDEO_FPS,
                (config.output_width, config.output_height)
            )

    def process_frame(self, frame):
        """
        Traite un frame vidéo
        
        Args:
            frame: Image en format BGR
            
        Returns:
            tuple: (frame traité, objets suivis)
        """
        # Redimensionner le frame
        frame = cv2.resize(frame, (config.output_width, config.output_height))

        # Correction des couleurs Macbeth
        self.color_correction_count += 1
        if self.color_correction_count % config.COLOR_CORRECTION_INTERVAL == 0:
            try:
                colors_measured = self.macbeth_detector.get_average_colors(frame)
                self.color_correction.calibrate_transformation(
                    np.array(colors_measured),
                    np.array(config.MACBETH_REFERENCE_COLORS)
                )
                print("Calibration de la correction des couleurs effectuée")
            except Exception as e:
                print(f"Erreur lors de la calibration des couleurs: {e}")

        # Appliquer la correction des couleurs si disponible
        if self.color_correction.last_correction_params is not None:
            frame = self.color_correction.apply_correction(frame)

        # Détecter les objets
        detections = self.tracker.object_detector.detect_objects(frame)

        # Analyser la couleur des objets détectés
        for det in detections:
            color = self.tracker.analyze_object(frame, det)
            det['color'] = color

        # Mettre à jour le tracking
        tracked_objects = self.tracker.update(detections)

        # Vérifier les traversées de ligne
        self.tracker.check_line_crossing(config.line_start, config.line_end)

        # Mettre à jour les compteurs
        self.update_crossing_counts(tracked_objects)

        return frame, tracked_objects

    def update_crossing_counts(self, tracked_objects):
        """Met à jour les compteurs de traversée"""
        for obj in tracked_objects.values():
            if obj.crossed_line and obj.direction:
                if obj.direction not in self.crossing_counts:
                    self.crossing_counts[obj.direction] = 0
                self.crossing_counts[obj.direction] += 1
                obj.crossed_line = False  # Réinitialiser pour éviter le double comptage

    def draw_interface(self, frame, tracked_objects):
        """Dessine l'interface utilisateur sur le frame"""
        # Dessiner la ligne de comptage
        cv2.line(frame, config.line_start, config.line_end, (0, 255, 255), 2)

        # Afficher les compteurs
        cv2.putText(frame, f"Up: {self.crossing_counts['up']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Down: {self.crossing_counts['down']}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Afficher le FPS
        cv2.putText(frame, f"FPS: {self.fps:.2f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Ajout d'un indicateur de correction des couleurs
        if self.color_correction.last_correction_params is not None:
            cv2.putText(frame, "Color Correction: Active", 
                      (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Color Correction: Inactive", 
                      (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Dessiner les objets suivis
        for obj in tracked_objects.values():
            # Dessiner la trajectoire
            if config.SHOW_TRAJECTORIES and len(obj.centroids) > 1:
                points = np.array(obj.centroids, dtype=np.int32)
                cv2.polylines(frame, [points], False, (0, 255, 0), 2)

            # Dessiner le dernier centroïde
            if config.SHOW_CENTER and obj.centroids:
                center = obj.centroids[-1]
                cv2.circle(frame, center, 4, (0, 255, 0), -1)

            # Afficher la couleur détectée
            if config.SHOW_LABELS and obj.color:
                cv2.putText(frame, obj.color, 
                           (center[0] - 20, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def update_fps(self):
        """Met à jour le calcul des FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        self.frame_count += 1

    def run(self):
        """Lance le traitement vidéo"""
        try:
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                # Traiter le frame
                frame, tracked_objects = self.process_frame(frame)

                # Dessiner l'interface
                self.draw_interface(frame, tracked_objects)

                # Mettre à jour les FPS
                self.update_fps()

                # Afficher le résultat
                cv2.imshow('Frame', frame)

                # Sauvegarder la vidéo si nécessaire
                if config.SAVE_VIDEO and self.video_writer:
                    self.video_writer.write(frame)

                # Gestion de la sortie
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Nettoie les ressources"""
        if self.video_capture:
            self.video_capture.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run() 
from Track import ObjectTracking
from video_processor import VideoProcessor
from display_manager import DisplayManager
from config import *

def main():
    # Initialisation des composants
    tracker = ObjectTracking()
    video_proc = VideoProcessor(output_width, output_height, desired_fps)
    display = DisplayManager()
    
    # Configuration du processeur vidéo
    video_proc.load_mask(mask_path)
    
    # Configuration du tracker
    tracker.set_components(video_proc, display)
    
    # Lancement du tracking
    tracker.track_objects()

if __name__ == "__main__":
    main() 
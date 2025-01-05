import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, output_width=1280, output_height=720, desired_fps=30):
        self.output_width = output_width
        self.output_height = output_height
        self.desired_fps = desired_fps
        self.mask = None

    def load_mask(self, mask_path):
        try:
            self.mask = cv2.imread(mask_path, 0)
            if self.mask is None:
                self.mask = np.ones((self.output_height, self.output_width), 
                                  dtype=np.uint8) * 255
            else:
                self.mask = cv2.resize(self.mask, (self.output_width, self.output_height))
        except Exception as e:
            print(f"Erreur lors du chargement du masque: {e}")
            self.mask = np.ones((self.output_height, self.output_width), 
                              dtype=np.uint8) * 255

    def setup_video_capture(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Erreur : Impossible d'accéder à la vidéo.")
        cap.set(cv2.CAP_PROP_FPS, self.desired_fps)
        return cap

    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.output_width, self.output_height))
        if self.mask is not None:
            if self.mask.shape[:2] != frame.shape[:2]:
                self.mask = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
            frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        return frame 
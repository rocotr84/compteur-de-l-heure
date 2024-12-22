import numpy as np
import cv2
import os

# Récupérer le chemin du dossier contenant le script
current_dir = os.path.dirname(__file__)
# Construire le chemin vers la vidéo dans 'assets'
video_path = os.path.join(current_dir, "..", "assets", "video.mp4")

# Construire le chemin vers le modèle dans 'assets'
modele_path = os.path.join(current_dir, "..", "assets", "yolo11n.pt")

def get_limits(color):
    # HSV Values not RGB
    colors = {
        "noir": ((0, 0, 0), (180, 255, 50)),
        "blanc": ((0, 0, 200), (180, 30, 255)),
        "rouge_fonce": ((0, 50, 50), (10, 255, 255)),
        "bleu_fonce": ((100, 50, 50), (130, 255, 120)),  # Dark blue
        "bleu_clair": ((100, 50, 121), (130, 255, 255)),  # Light blue
        "vert_fonce": ((35, 50, 50), (85, 255, 255)),  # Dark green
        "rose": ((140, 50, 50), (170, 255, 255)),
        "jaune": ((20, 100, 100), (40, 255, 255)),
        "vert_clair": ((40, 50, 50), (80, 255, 255)),  # Light green
    }

    color = color.lower()
    if color in colors:
        lower_limit, upper_limit = colors[color]
        lower_limit = np.array(lower_limit, dtype=np.uint8)
        upper_limit = np.array(upper_limit, dtype=np.uint8)
        return lower_limit, upper_limit
    else:
        return None, None


def adjust_brightness(img, factor=1.2):
    # Convert to HSV to adjust brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def detect_color_in_roi(frame, roi, color_list):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]  # Crop the region of interest

    color_counts = {color: 0 for color in color_list}

    for color_name in color_list:
        lower_limit, upper_limit = get_limits(color_name)
        if lower_limit is not None and upper_limit is not None:
            mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
            # Crop the mask to the ROI area
            roi_mask = mask[roi[1]:roi[3], roi[0]:roi[2]]
            color_counts[color_name] = cv2.countNonZero(roi_mask)

    # Find the color with the most pixels in the ROI
    detected_color = max(color_counts, key=color_counts.get)
    return detected_color


def main():
    cap = cv2.VideoCapture(video_path)
    
    # Define the fixed region of interest (ROI) as (x1, y1, x2, y2)
    roi = (350, 600, 450, 650)  # Example: a square ROI from (100, 100) to (400, 400)

    while True:
        ret, frame = cap.read()

        if ret:
            # Mirror the frame horizontally
            mirrored_frame = cv2.flip(frame, 1)  # 1 for horizontal flip
            # Brighten the frame (adjust the factor as needed)
            brightened_frame = adjust_brightness(mirrored_frame, 1.5)

            # Draw the ROI on the frame
            cv2.rectangle(brightened_frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

            # Define the list of colors to detect
            colors_to_detect = ["noir", "blanc", "rouge_fonce", "bleu_fonce", "bleu_clair", 
                                "vert_fonce", "rose", "jaune", "vert_clair"]
            detected_color = detect_color_in_roi(brightened_frame, roi, colors_to_detect)

            # Display the detected color in the center of the ROI
            cv2.putText(brightened_frame, f"Detected Color: {detected_color.capitalize()}", 
                        (roi[0] + 10, roi[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 0, 255), 2, cv2.LINE_AA)

            # Display the instruction to press ESC key to close
            cv2.putText(brightened_frame, "Press ESC key to close", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('frame', brightened_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Check if the key pressed is the Escape key (ASCII 27)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import csv
from pathlib import Path
import time
import psutil
import GPUtil

# DÃ©finir la couleur yellow en RGB et HSV

defined_colors_rgb = { 
    'yellow': (255, 255, 0), 
    'red': (255, 0, 0), 
    'light_green': (144, 238, 144), 
    'dark_green': (0, 100, 0), 
    'pink': (255, 192, 203), 
    'white': (255, 255, 255), 
    'black': (0, 0, 0), 
    'light_blue': (38, 149, 202), 
    'burgundy': (128, 0, 32), 
    'dark_blue': (0, 0, 139)
    }

defined_colors_hsv = { 
    'yellow': (30, 100, 100),
    'red': (0, 100, 100),
    'light_green': (120, 25, 94),
    'dark_green': (120, 100, 39),
    'pink': (350, 25, 100),
    'white': (0, 0, 100), 
    'black': (0, 0, 0), 
    'light_blue': (195, 25, 90), 
    'burgundy': (345, 100, 50), 
    'dark_blue': (240, 100, 54)
    }


# Fonction pour trouver la couleur la plus proche en utilisant la distance euclidienne
def closest_color(color, color_dict):
    min_dist = float('inf')
    closest_color_name = None
    for color_name, defined_color in color_dict.items():
        dist = np.linalg.norm(np.array(color) - np.array(defined_color))
        if dist < min_dist:
            min_dist = dist
            closest_color_name = color_name
    return closest_color_name

# Fonction pour mesurer la consommation de CPU et de GPU
def measure_resources():
    cpu_usage = psutil.cpu_percent(interval=None)
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100 if gpus else 0
    return cpu_usage, gpu_usage

# Fonction pour afficher l'image avec un rectangle de 50x50 pixels dessinÃ© au centre
def display_image_with_rectangle(image_path):
    # Lire l'image
    img = cv2.imread(image_path)
    
    # DÃ©finir les coordonnÃ©es du rectangle central
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    start_x, start_y = center_x - 50, center_y - 50
    end_x, end_y = center_x + 50, center_y + 50
    
    # Dessiner le rectangle sur l'image
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Afficher l'image
    cv2.imshow('Image with Rectangle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Fonction pour trouver la couleur dominante dans un rectangle central de 50x50 pixels en utilisant KMeans en RGB
def find_dominant_color_rgb(image_path):
    start_time = time.time()
    cpu_start, gpu_start = measure_resources()
    
    # Lire l'image
    img = cv2.imread(image_path)
    
    # Extraire les couleurs dominantes dans un rectangle de 50x50 pixels au centre de l'image
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    start_x, start_y = center_x - 25, center_y - 25
    end_x, end_y = center_x + 25, center_y + 25
    
    # Assurez-vous que les coordonnÃ©es restent dans les limites de l'image
    start_x, start_y = max(0, start_x), max(0, start_y)
    end_x, end_y = min(w, end_x), min(h, end_y)
    
    center_rect = img[start_y:end_y, start_x:end_x]
    
    # Convertir en RGB pour l'analyse des couleurs
    img_rgb = cv2.cvtColor(center_rect, cv2.COLOR_BGR2RGB)
    pixels = np.float32(img_rgb.reshape(-1, 3))
    n_colors = 5
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Trouver la couleur dominante
    _, counts = np.unique(labels, return_counts=True)
    dominant_color_rgb = palette[np.argmax(counts)]
    
    # Trouver la couleur la plus proche parmi les couleurs dÃ©finies
    closest_color_name = closest_color(dominant_color_rgb, defined_colors_rgb)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    cpu_end, gpu_end = measure_resources()
    
    cpu_usage = (cpu_end - cpu_start)
    gpu_usage = (gpu_end - gpu_start)
    
    return closest_color_name, elapsed_time, cpu_usage, gpu_usage

# Fonction pour trouver la couleur dominante dans un rectangle central de 50x50 pixels en utilisant les moyennes HSV
def find_dominant_color_hsv(image_path):
    start_time = time.time()
    cpu_start, gpu_start = measure_resources()
    
    # Lire l'image
    img = cv2.imread(image_path)
    
    # Extraire les couleurs dominantes dans un rectangle de 50x50 pixels au centre de l'image
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    start_x, start_y = center_x - 25, center_y - 25
    end_x, end_y = center_x + 25, center_y + 25
    
    # Assurez-vous que les coordonnÃ©es restent dans les limites de l'image
    start_x, start_y = max(0, start_x), max(0, start_y)
    end_x, end_y = min(w, end_x), min(h, end_y)
    
    center_rect = img[start_y:end_y, start_x:end_x]
    
    # Convertir en HSV pour l'analyse des couleurs
    img_hsv = cv2.cvtColor(center_rect, cv2.COLOR_BGR2HSV)
    pixels = img_hsv.reshape(-1, 3)
    
    # Trouver les valeurs moyennes de Hue, Saturation et Value
    mean_hue = np.mean(pixels[:, 0])
    mean_saturation = np.mean(pixels[:, 1])
    mean_value = np.mean(pixels[:, 2])
    
    dominant_color_hsv = (mean_hue, mean_saturation, mean_value)
    
    # Trouver la couleur la plus proche parmi les couleurs dÃ©finies
    closest_color_name = closest_color(dominant_color_hsv, defined_colors_hsv)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    cpu_end, gpu_end = measure_resources()
    
    cpu_usage = (cpu_end - cpu_start)
    gpu_usage = (gpu_end - gpu_start)
    
    return closest_color_name, elapsed_time, cpu_usage, gpu_usage

# Fonction pour trouver la couleur dominante dans un rectangle central de 50x50 pixels en utilisant l'espace de couleur LAB
def find_dominant_color_lab(image_path):
    start_time = time.time()
    cpu_start, gpu_start = measure_resources()
    
    # Lire l'image
    img = cv2.imread(image_path)
    
    # Extraire les couleurs dominantes dans un rectangle de 50x50 pixels au centre de l'image
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    start_x, start_y = center_x - 25, center_y - 25
    end_x, end_y = center_x + 25, center_y + 25
    
    # Assurez-vous que les coordonnÃ©es restent dans les limites de l'image
    start_x, start_y = max(0, start_x), max(0, start_y)
    end_x, end_y = min(w, end_x), min(h, end_y)
    
    center_rect = img[start_y:end_y, start_x:end_x]
    
    # Convertir en LAB pour l'analyse des couleurs
    img_lab = cv2.cvtColor(center_rect, cv2.COLOR_BGR2LAB)
    pixels = img_lab.reshape(-1, 3)
    
    # Trouver les valeurs moyennes de L, A et B
    mean_l = np.mean(pixels[:, 0])
    mean_a = np.mean(pixels[:, 1])
    mean_b = np.mean(pixels[:, 2])
    
    dominant_color_lab = (mean_l, mean_a, mean_b)
    
    # Trouver la couleur la plus proche parmi les couleurs dÃ©finies
    closest_color_name = closest_color(dominant_color_lab, defined_colors_hsv)  # Utiliser HSV pour la comparaison
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    cpu_end, gpu_end = measure_resources()
    
    cpu_usage = (cpu_end - cpu_start)
    gpu_usage = (gpu_end - gpu_start)
    
    return closest_color_name, elapsed_time, cpu_usage, gpu_usage

# Fonction pour trouver la couleur dominante dans un rectangle central de 50x50 pixels en utilisant l'histogramme de couleurs
def find_dominant_color_histogram(image_path):
    start_time = time.time()
    cpu_start, gpu_start = measure_resources()
    
    # Lire l'image
    img = cv2.imread(image_path)
    
    # Extraire les couleurs dominantes dans un rectangle de 50x50 pixels au centre de l'image
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    start_x, start_y = center_x - 25, center_y - 25
    end_x, end_y = center_x + 25, center_y + 25
    
    # Assurez-vous que les coordonnÃ©es restent dans les limites de l'image
    start_x, start_y = max(0, start_x), max(0, start_y)
    end_x, end_y = min(w, end_x), min(h, end_y)
    
    center_rect = img[start_y:end_y, start_x:end_x]
    
    # Convertir en HSV pour l'analyse des couleurs
    img_hsv = cv2.cvtColor(center_rect, cv2.COLOR_BGR2HSV)
    
    # Calculer l'histogramme de la teinte
    hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
    dominant_hue = np.argmax(hist)
    
    # Trouver la saturation et la valeur moyennes pour la teinte dominante
    dominant_hue = int(dominant_hue)  # Assure-toi que c'est bien un entier
    mask = cv2.inRange(img_hsv, (dominant_hue, 50, 50), (dominant_hue, 255, 255))

    mean_saturation = cv2.mean(img_hsv[:, :, 1], mask=mask)[0]
    mean_value = cv2.mean(img_hsv[:, :, 2], mask=mask)[0]
    
    dominant_color_histogram = (dominant_hue, mean_saturation, mean_value)
    
    # Trouver la couleur la plus proche parmi les couleurs dÃ©finies
    closest_color_name = closest_color(dominant_color_histogram, defined_colors_hsv)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    cpu_end, gpu_end = measure_resources()
    
    cpu_usage = (cpu_end - cpu_start)
    gpu_usage = (gpu_end - gpu_start)
    
    return closest_color_name, elapsed_time, cpu_usage, gpu_usage

# Fonction pour analyser toutes les images d'un dossier spÃ©cifique Ã  la couleur yellow et stocker les rÃ©sultats dans un fichier CSV
def analyze_images_in_folder(color, base_folder, output_csv):
    folder_path = Path(base_folder) / color / 'face'
    image_paths = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
    
    results = {
        'rgb': {'count': 0, 'colors': [], 'total_time': 0, 'correct': 0, 'total_cpu': 0, 'total_gpu': 0},
        'hsv': {'count': 0, 'colors': [], 'total_time': 0, 'correct': 0, 'total_cpu': 0, 'total_gpu': 0},
        'lab': {'count': 0, 'colors': [], 'total_time': 0, 'correct': 0, 'total_cpu': 0, 'total_gpu': 0},
        'histogram': {'count': 0, 'colors': [], 'total_time': 0, 'correct': 0, 'total_cpu': 0, 'total_gpu': 0}
    }
    
    # Mesurer les ressources avant le traitement
    initial_cpu, initial_gpu = measure_resources()
    
    for image_path in image_paths:
        # Afficher l'image avec le rectangle
        display_image_with_rectangle(str(image_path))
        
        # Obtenir la couleur correcte Ã  partir du nom du dossier
        correct_color = color
        
        # MÃ©thode RGB
        dominant_color_rgb, time_rgb, cpu_rgb, gpu_rgb = find_dominant_color_rgb(str(image_path))
        results['rgb']['count'] += 1
        results['rgb']['colors'].append(dominant_color_rgb)
        results['rgb']['total_time'] += time_rgb
        results['rgb']['total_cpu'] += cpu_rgb
        results['rgb']['total_gpu'] += gpu_rgb
        if dominant_color_rgb == correct_color:
            results['rgb']['correct'] += 1
        
        # MÃ©thode HSV
        dominant_color_hsv_mean, time_hsv_mean, cpu_hsv_mean, gpu_hsv_mean = find_dominant_color_hsv(str(image_path))
        results['hsv']['count'] += 1
        results['hsv']['colors'].append(dominant_color_hsv_mean)
        results['hsv']['total_time'] += time_hsv_mean
        results['hsv']['total_cpu'] += cpu_hsv_mean
        results['hsv']['total_gpu'] += gpu_hsv_mean
        if dominant_color_hsv_mean == correct_color:
            results['hsv']['correct'] += 1
        
        # MÃ©thode LAB
        dominant_color_lab, time_lab, cpu_lab, gpu_lab = find_dominant_color_lab(str(image_path))
        results['lab']['count'] += 1
        results['lab']['colors'].append(dominant_color_lab)
        results['lab']['total_time'] += time_lab
        results['lab']['total_cpu'] += cpu_lab
        results['lab']['total_gpu'] += gpu_lab
        if dominant_color_lab == correct_color:
            results['lab']['correct'] += 1
        
        # MÃ©thode Histogramme
        dominant_color_histogram, time_histogram, cpu_histogram, gpu_histogram = find_dominant_color_histogram(str(image_path))
        results['histogram']['count'] += 1
        results['histogram']['colors'].append(dominant_color_histogram)
        results['histogram']['total_time'] += time_histogram
        results['histogram']['total_cpu'] += cpu_histogram
        results['histogram']['total_gpu'] += gpu_histogram
        if dominant_color_histogram == correct_color:
            results['histogram']['correct'] += 1
    
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 'Number of Images Tested', 'Colors Found', 'Total Calculation Time (s)', 'Average Calculation Time (s)', 'Percentage of Correct Colors', 'Average CPU Usage (%)', 'Average GPU Usage (%)'])
        
        for method in results:
            num_images = results[method]['count']
            total_time = results[method]['total_time']
            avg_time = total_time / num_images if num_images > 0 else 0
            correct_percentage = (results[method]['correct'] / num_images) * 100 if num_images > 0 else 0
            avg_cpu = results[method]['total_cpu'] / num_images if num_images > 0 else 0
            avg_gpu = results[method]['total_gpu'] / num_images if num_images > 0 else 0
            writer.writerow([
                f"{color.upper()}_{method.upper()}",
                num_images,
                ', '.join(results[method]['colors']),
                total_time,
                avg_time,
                correct_percentage,
                avg_cpu - initial_cpu,
                avg_gpu - initial_gpu
            ])

# Fonction principale pour analyser les images de la couleur yellow
def analyze_yellow(base_folder, output_csv):
    color = 'light_blue'
    analyze_images_in_folder(color, base_folder, output_csv)

# Exemple d'utilisation
base_folder = r'C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\assets\photos\camera4K'
output_csv = r'C:\Users\victo\Desktop\camera detection_2\compteur-de-l-heure\Tests\results.csv'
analyze_yellow(base_folder, output_csv)


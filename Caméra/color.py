import json
import math

# Charger les couleurs depuis le fichier JSON
def load_colors(filename='colors.json'):
    with open(filename, 'r') as json_file:
        return json.load(json_file)

# Fonction pour calculer la distance euclidienne
def euclidean_distance(rgb1, rgb2):
    return math.sqrt((rgb1[0] - rgb2[0]) ** 2 + (rgb1[1] - rgb2[1]) ** 2 + (rgb1[2] - rgb2[2]) ** 2)

# Fonction pour trouver la couleur la plus proche
def find_closest_color(input_rgb, color_list):
    closest_color = None
    min_distance = float('inf')
    
    for color in color_list:
        distance = euclidean_distance(input_rgb, color["rgb"])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    
    return closest_color

# Exemple d'utilisation
input_rgb = (100, 150, 200)  # Exemple de couleur à rechercher
color_list = load_colors('colors.json')  # Charger la liste de couleurs depuis le fichier JSON
closest_color = find_closest_color(input_rgb, color_list)

print(f"La couleur la plus proche est {closest_color['name']} avec le code hexadécimal {closest_color['hex']}")

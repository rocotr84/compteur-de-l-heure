import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv('compteur-de-l-heure/Tests_detection/results.csv', header=None)

# Renommer les colonnes pour plus de clarté
df.columns = ['Frame', 'Model', 'Score', 'Time', 'Count', 'Type']

# Analyser les données
# Calculer le temps moyen de détection par modèle
mean_detection_time = df.groupby('Model')['Time'].mean().reset_index()

# Créer un graphique
plt.figure(figsize=(10, 6))
plt.bar(mean_detection_time['Model'], mean_detection_time['Time'], color='lightgreen')
plt.xlabel('Modèle')
plt.ylabel('Temps moyen de détection (s)')
plt.title('Temps moyen de détection par modèle')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
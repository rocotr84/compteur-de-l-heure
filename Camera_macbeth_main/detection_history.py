import csv
from collections import defaultdict
from datetime import datetime
import os
import sqlite3
from config import SAVE_SQL, CSV_OUTPUT_PATH, SQL_DB_PATH
# Variables globales pour la gestion de l'historique des détections
person_detection_history = defaultdict(list)  # {person_id: [liste des valeurs détectées]}
csv_output_file = None
csv_output_writer = None
db_connection = None
db_cursor = None

def init_detection_history():
    """
    Initialise l'historique des détections selon le mode choisi (CSV ou SQLite).
    """
    global csv_output_file, csv_output_writer, db_connection, db_cursor
    
    if SAVE_SQL:
        try:
            db_connection = sqlite3.connect(SQL_DB_PATH)
            db_cursor = db_connection.cursor()
            
            # Création de la table si elle n'existe pas
            db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    person_id INTEGER NOT NULL,
                    detected_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            db_connection.commit()
            print(f"Base de données SQLite initialisée : {SQL_DB_PATH}")
        except sqlite3.Error as e:
            print(f"Erreur lors de l'initialisation de SQLite : {e}")
    else:
        # Initialisation CSV
        csv_output_file = open(CSV_OUTPUT_PATH, 'a', newline='', buffering=1)
        csv_output_writer = csv.writer(csv_output_file)
        print(f"Fichier CSV initialisé : {CSV_OUTPUT_PATH}")

def update_detection_value(person_id, detected_value):
    """
    Met à jour l'historique des détections pour une personne.
    
    Args:
        person_id (int): Identifiant unique de la personne
        detected_value (str): Valeur détectée (couleur/numéro)
    
    Notes:
        - Les valeurs None sont ignorées
        - L'historique est stocké dans person_detection_history
    """
    if detected_value is not None:
        person_detection_history[person_id].append(detected_value)

def get_dominant_detection(person_id):
    """
    Détermine la valeur la plus fréquente pour une personne donnée.
    
    Analyse l'historique des détections pour une personne et retourne
    la valeur qui apparaît le plus souvent, permettant de filtrer
    les détections erronées.
    
    Args:
        person_id (int): Identifiant unique de la personne
    
    Returns:
        str or None: La valeur la plus fréquente, ou None si aucune détection valide
    """
    person_detections = person_detection_history[person_id]
    if not person_detections:
        return None
    
    detection_frequencies = defaultdict(int)
    for detected_value in person_detections:
        if detected_value is not None:
            detection_frequencies[detected_value] += 1
        
    if not detection_frequencies:
        return None
    
    return max(detection_frequencies.items(), key=lambda x: x[1])[0]

def record_crossing(person_id, formatted_time):
    """
    Enregistre le passage d'une personne selon le mode choisi.
    
    Args:
        person_id (int): Identifiant unique de la personne
        formatted_time (str): Heure système formatée
    """
    dominant_detection = get_dominant_detection(person_id)
    if dominant_detection:
        if SAVE_SQL and db_connection and db_cursor:
            try:
                db_cursor.execute('''
                    INSERT INTO detections (timestamp, person_id, detected_value)
                    VALUES (?, ?, ?)
                ''', (formatted_time, person_id, dominant_detection))
                db_connection.commit()
                print(f"Enregistrement SQLite réussi : {formatted_time}, "
                      f"{person_id}, {dominant_detection}")
            except sqlite3.Error as e:
                print(f"Erreur d'enregistrement SQLite : {e}")
        elif not SAVE_SQL and csv_output_writer is not None and csv_output_file is not None:
            try:
                csv_output_writer.writerow([formatted_time, person_id, dominant_detection])
                csv_output_file.flush()
                os.fsync(csv_output_file.fileno())
                print(f"Enregistrement CSV réussi : {formatted_time}, "
                      f"{person_id}, {dominant_detection}")
            except Exception as e:
                print(f"Erreur d'enregistrement CSV : {e}")
        else:
            print("Erreur : Aucun système de stockage n'est correctement initialisé")
    
    if person_id in person_detection_history:
        del person_detection_history[person_id]

def cleanup():
    """
    Ferme proprement les connexions selon le mode utilisé.
    """
    if SAVE_SQL and db_connection:
        try:
            db_connection.close()
            print("Connexion à la base de données fermée")
        except sqlite3.Error as e:
            print(f"Erreur lors de la fermeture de la base de données : {e}")
    else:
        if csv_output_file:
            csv_output_file.close()
            print("Fichier CSV fermé") 
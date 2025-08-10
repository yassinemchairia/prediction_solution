from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import requests
import threading
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import os
import sys

# Créer le dossier logs s'il n'existe pas
os.makedirs('logs', exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/solution_prediction_api.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app, resources={r"/predict_solution": {"origins": ["http://angular_app:80", "http://192.168.107.129:4200"]}})# Paramètres
MODEL_PATH = "./solution_model.pkl"
TFIDF_PATH = "./tfidf_vectorizer.pkl"
ALERTES_CSV_PATH = "./alertes.csv"
SPRING_API_URL = "http://springboot:8087/alertes/export"
UPDATE_INTERVAL = 300

# Vérifier l'existence des fichiers
for path in [MODEL_PATH, TFIDF_PATH, ALERTES_CSV_PATH]:
    if not os.path.exists(path):
        logging.error(f"Fichier manquant : {path}. Assurez-vous que les scripts de prétraitement et d'entraînement ont été exécutés.")
        sys.exit(1)

# Charger le modèle et le vectoriseur
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_names = model_data['feature_names']
    tfidf = joblib.load(TFIDF_PATH)
    logging.info("Modèle et vectoriseur chargés avec succès")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle ou du vectoriseur : {str(e)}")
    sys.exit(1)

# Charger les données d'entraînement pour la recherche de similarité
try:
    alertes_df = pd.read_csv(ALERTES_CSV_PATH)
    logging.info(f"Données d'entraînement chargées : {len(alertes_df)} lignes")
except Exception as e:
    logging.error(f"Erreur lors du chargement de alertes.csv : {str(e)}")
    alertes_df = pd.DataFrame()

def fetch_and_retrain():
    global model, feature_names, tfidf, alertes_df
    while True:
        try:
            logging.info("Récupération des données depuis l'API Spring Boot")
            response = requests.get(SPRING_API_URL)
            if response.status_code != 200:
                logging.warning(f"Erreur HTTP {response.status_code} lors de la récupération des données")
                time.sleep(UPDATE_INTERVAL)
                continue

            data = response.json()
            if not data or len(data) == 0:
                logging.warning("Aucune donnée reçue de l'API Spring")
                time.sleep(UPDATE_INTERVAL)
                continue

            # Convertir en DataFrame
            df = pd.DataFrame(data)
            logging.info(f"Données reçues : {len(df)} lignes")

            # Nettoyage
            df['valeurDeclenchement'] = df['valeurDeclenchement'].fillna(0)
            df['description'] = df['description'].fillna('')
            df['solution'] = df['solution'].fillna('Vérification manuelle')
            df = df[(df['solution'] != '') & (df['satisfaction'] >= 3)]
            for col in ['typePanne', 'niveauGravite', 'typeCapteur', 'emplacement']:
                df[col] = df[col].str.upper()
            df['niveauGravite'] = df['niveauGravite'].replace('HIGHT_CRITICAL', 'HIGH_CRITICAL')
            df['typePanne'] = df['typePanne'].replace({'TEMPERATURE': 'CLIMATISATION', 'HUMIDITE': 'ENVIRONNEMENT'})
            logging.info(f"Après filtrage et remplacement de typePanne : {len(df)} lignes")

            if len(df) < 10:
                logging.warning(f"Trop peu de données ({len(df)} lignes) pour réentraîner le modèle")
                time.sleep(UPDATE_INTERVAL)
                continue

            # Mettre à jour alertes_df pour la recherche de similarité
            df.to_csv(ALERTES_CSV_PATH, index=False)
            alertes_df = df.copy()
            logging.info(f"Données d'entraînement mises à jour : {ALERTES_CSV_PATH}")

            # Normalisation
            scaler = MinMaxScaler()
            df['valeurDeclenchement_norm'] = scaler.fit_transform(df[['valeurDeclenchement']])

            # Encodage des variables catégoriques
            categorical_columns = ['typePanne', 'niveauGravite', 'typeCapteur', 'emplacement']
            df_encoded = pd.get_dummies(df, columns=categorical_columns)

            # Vectorisation de la description
            description_vectors = tfidf.fit_transform(df['description']).toarray()
            description_columns = [f"desc_{i}" for i in range(description_vectors.shape[1])]
            df_description = pd.DataFrame(description_vectors, columns=description_columns)

            # Combiner les features
            df_final = pd.concat([df_encoded, df_description], axis=1)
            features = [col for col in df_final.columns if col not in ['idAlerte', 'description', 'solution', 'satisfaction']]
            X = df_final[features]
            y = df_final['solution']

            # Équilibrage avec SMOTE
            smote = SMOTE(random_state=42, k_neighbors=min(3, len(y) - 1))
            X_balanced, y_balanced = smote.fit_resample(X, y)
            logging.info(f"Données après équilibrage : {len(X_balanced)} lignes")

            # Vérifier la distribution des classes
            class_counts = pd.Series(y_balanced).value_counts()
            logging.info(f"Distribution des classes après SMOTE : \n{class_counts.to_string()}")

            # Réentraîner le modèle
            model.fit(X_balanced, y_balanced)
            feature_names = X_balanced.columns.tolist()

            # Sauvegarder le modèle et le vectoriseur
            joblib.dump({'model': model, 'feature_names': feature_names}, MODEL_PATH)
            joblib.dump(tfidf, TFIDF_PATH)
            logging.info("Modèle et vectoriseur mis à jour avec succès")

        except Exception as e:
            logging.error(f"Erreur lors du réentraînement : {str(e)}")
        
        time.sleep(UPDATE_INTERVAL)

@app.route('/predict_solution', methods=['POST'])
def predict_solution():
    try:
        data = request.get_json()
        if not data:
            logging.error("Requête JSON vide")
            return jsonify({'error': 'Requête JSON vide'}), 400

        # Validation des entrées
        required_fields = ['typePanne', 'niveauGravite', 'valeurDeclenchement', 'typeCapteur', 'emplacement']
        for field in required_fields:
            if field not in data:
                logging.error(f"Champ manquant : {field}")
                return jsonify({'error': f"Champ '{field}' manquant"}), 400

        # Extraction et validation des données
        try:
            type_panne = str(data['typePanne']).upper()
            if type_panne not in ['ELECTRICITE', 'CLIMATISATION', 'ENVIRONNEMENT']:
                logging.error(f"Valeur non valide pour typePanne : {type_panne}")
                return jsonify({'error': f"Valeur non valide pour typePanne : {type_panne}. Valeurs attendues : ELECTRICITE, CLIMATISATION, ENVIRONNEMENT"}), 400
            niveau_gravite = str(data['niveauGravite']).upper().replace('HIGHT_CRITICAL', 'HIGH_CRITICAL')
            valeur_declenchement = float(data['valeurDeclenchement'])
            type_capteur = str(data['typeCapteur']).upper()
            emplacement = str(data['emplacement']).upper()
            description = str(data.get('description', '')).strip()
        except (ValueError, TypeError) as e:
            logging.error(f"Erreur de type dans les données d'entrée : {str(e)}")
            return jsonify({'error': f"Erreur de type dans les données : {str(e)}"}), 400

        # Créer DataFrame
        df = pd.DataFrame([{
            'typePanne': type_panne,
            'niveauGravite': niveau_gravite,
            'valeurDeclenchement': valeur_declenchement,
            'typeCapteur': type_capteur,
            'emplacement': emplacement,
            'description': description
        }])

        # Normalisation
        scaler = MinMaxScaler()
        df['valeurDeclenchement_norm'] = scaler.fit_transform(df[['valeurDeclenchement']])

        # Encodage des variables catégoriques
        categorical_columns = ['typePanne', 'niveauGravite', 'typeCapteur', 'emplacement']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        # Alignement des features catégoriques
        for col in feature_names:
            if col.startswith(('typePanne_', 'niveauGravite_', 'typeCapteur_', 'emplacement_')) and col not in df_encoded.columns:
                df_encoded[col] = 0

        # Vectorisation de la description
        description_vector = tfidf.transform([description]).toarray()
        description_columns = [f"desc_{i}" for i in range(description_vector.shape[1])]
        df_description = pd.DataFrame(description_vector, columns=description_columns)

        # Combiner les features
        df_final = pd.concat([df_encoded, df_description], axis=1)

        # Alignement avec les features du modèle
        for col in feature_names:
            if col not in df_final.columns:
                df_final[col] = 0
        df_final = df_final[feature_names]

        # Prédiction
        solution = model.predict(df_final)[0]
        logging.info(f"Prédiction réussie : {solution}")

        # Trouver les alertes similaires
        similar_alertes = []
        if not alertes_df.empty:
            # Préparer les données d'entraînement pour la similarité
            df_train = alertes_df.copy()
            df_train['valeurDeclenchement_norm'] = scaler.fit_transform(df_train[['valeurDeclenchement']])
            df_train_encoded = pd.get_dummies(df_train, columns=categorical_columns)
            for col in feature_names:
                if col.startswith(('typePanne_', 'niveauGravite_', 'typeCapteur_', 'emplacement_')) and col not in df_train_encoded.columns:
                    df_train_encoded[col] = 0
            train_description_vectors = tfidf.transform(df_train['description']).toarray()
            train_description_df = pd.DataFrame(train_description_vectors, columns=description_columns)
            df_train_final = pd.concat([df_train_encoded, train_description_df], axis=1)
            for col in feature_names:
                if col not in df_train_final.columns:
                    df_train_final[col] = 0
            df_train_final = df_train_final[feature_names]

            # Calculer la similarité cosinus
            similarities = cosine_similarity(df_final, df_train_final)[0]
            df_train['similarity'] = similarities
            # Filtrer les alertes avec la même solution, typePanne et niveauGravite
            similar_alertes_df = df_train[
                (df_train['solution'] == solution) &
                (df_train['typePanne'] == type_panne) &
                (df_train['niveauGravite'] == niveau_gravite)
            ]
            if not similar_alertes_df.empty:
                # Trier par similarité et prendre les 2 premières
                top_similar = similar_alertes_df.sort_values(by='similarity', ascending=False).head(2)
                similar_alertes = top_similar[['idAlerte', 'typePanne', 'niveauGravite', 'valeurDeclenchement', 'typeCapteur', 'emplacement', 'description', 'solution', 'satisfaction']].to_dict('records')
            else:
                # Si aucune alerte ne correspond, prendre les 2 plus similaires avec la même solution
                similar_alertes_df = df_train[df_train['solution'] == solution]
                if not similar_alertes_df.empty:
                    top_similar = similar_alertes_df.sort_values(by='similarity', ascending=False).head(2)
                    similar_alertes = top_similar[['idAlerte', 'typePanne', 'niveauGravite', 'valeurDeclenchement', 'typeCapteur', 'emplacement', 'description', 'solution', 'satisfaction']].to_dict('records')

        # Réponse JSON
        response = {
            'solution': str(solution),
            'similar_alertes': similar_alertes
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Erreur dans predict_solution : {str(e)}")
        return jsonify({'error': f"Erreur serveur : {str(e)}"}), 500

if __name__ == "__main__":
    try:
        threading.Thread(target=fetch_and_retrain, daemon=True).start()
        logging.info("Serveur Flask démarré sur le port 5000")
        app.run(port=5000, debug=False)
    except Exception as e:
        logging.error(f"Erreur lors du démarrage du serveur : {str(e)}")

        raise




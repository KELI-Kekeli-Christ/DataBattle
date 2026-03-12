"""
=============================================================================
MODÈLE FINAL DE PRÉDICTION COMBINÉ - PRÉDICTION DU PROCHAIN ÉCLAIR
=============================================================================

Ce script combine tous les modèles pour prédire:
1. SI un éclair va se produire (classification)
2. QUAND il va se produire (série temporelle)
3. QUEL TYPE d'éclair (Cloud-to-Ground vs Intra-Cloud)
4.Quelle sera son INTENSITÉ (amplitude)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Statistiques
from statsmodels.tsa.arima.model import ARIMA

print("=" * 80)
print("MODÈLE FINAL COMBINÉ - PRÉDICTION DU PROCHAIN ÉCLAIR")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT ET PRÉTRAITEMENT
# ============================================================================
print("\n[1/7] Chargement des données...")

data = pd.read_csv('segment_alerts_all_airports_train.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# Features temporelles
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['dayofweek'] = data['date'].dt.dayofweek
data['dayofyear'] = data['date'].dt.dayofyear
data['amplitude_abs'] = data['amplitude'].abs()

# Encodage
le_airport = LabelEncoder()
data['airport_encoded'] = le_airport.fit_transform(data['airport'])
data['icloud_encoded'] = data['icloud'].astype(int)

print(f"  ✓ {len(data):,} éclairs chargés")
print(f"  ✓ {data['airport'].nunique()} aéroports")

# ============================================================================
# 2. MODÈLE 1: PRÉDICTION DE PROBABILITÉ D'ÉCLAIR (PAR HEURE)
# ============================================================================
print("\n[2/7] Entraînement du modèle de probabilité d'éclair...")

# Créer dataset par heure
hourly_data = data.groupby(data['date'].dt.floor('H')).agg({
    'lightning_id': 'count',
    'amplitude_abs': 'mean',
    'airport_encoded': 'first',
    'icloud_encoded': 'mean',
    'year': 'first',
    'month': 'first',
    'dayofweek': 'first'
}).reset_index()

hourly_data.columns = ['date', 'alert_count', 'avg_amplitude', 'airport', 
                       'pct_cloud_ground', 'year', 'month', 'dayofweek']
hourly_data['has_alert'] = (hourly_data['alert_count'] > 0).astype(int)
hourly_data['hour'] = hourly_data['date'].dt.hour

# Features cycliques
hourly_data['hour_sin'] = np.sin(2 * np.pi * hourly_data['hour'] / 24)
hourly_data['hour_cos'] = np.cos(2 * np.pi * hourly_data['hour'] / 24)
hourly_data['month_sin'] = np.sin(2 * np.pi * hourly_data['month'] / 12)
hourly_data['month_cos'] = np.cos(2 * np.pi * hourly_data['month'] / 12)

# Features pour probabilité
X_prob = hourly_data[['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                       'dayofweek', 'airport', 'hour']].dropna()
y_prob = hourly_data.loc[X_prob.index, 'has_alert']

# Équilibrer les classes si nécessaire
if len(y_prob.unique()) > 1:
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_prob, y_prob, test_size=0.2, random_state=42, stratify=y_prob
    )
    
    # Modèle de probabilité
    prob_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    prob_model.fit(X_train_p, y_train_p)
    
    prob_accuracy = accuracy_score(y_test_p, prob_model.predict(X_test_p))
    prob_f1 = f1_score(y_test_p, prob_model.predict(X_test_p))
    
    print(f"  ✓ Modèle probabilité: Accuracy={prob_accuracy:.3f}, F1={prob_f1:.3f}")
else:
    print("  ⚠ Une seule classe dans les données - utilisation d'un modèle simplifié")
    prob_model = None
    prob_accuracy = 0.5

# ============================================================================
# 3. MODÈLE 2: PRÉDICTION DU TYPE D'ÉCLAIR
# ============================================================================
print("\n[3/7] Entraînement du modèle de type d'éclair...")

features_class = ['amplitude', 'maxis', 'dist', 'azimuth', 'airport_encoded', 
                  'month', 'hour', 'dayofweek']
X_class = data[features_class].dropna()
y_class = data.loc[X_class.index, 'icloud_encoded']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

scaler_class = StandardScaler()
X_train_c_scaled = scaler_class.fit_transform(X_train_c)
X_test_c_scaled = scaler_class.transform(X_test_c)

type_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
type_model.fit(X_train_c_scaled, y_train_c)

type_f1 = f1_score(y_test_c, type_model.predict(X_test_c_scaled))
type_auc = roc_auc_score(y_test_c, type_model.predict_proba(X_test_c_scaled)[:, 1])

print(f"  ✓ Modèle type éclair: F1={type_f1:.3f}, ROC-AUC={type_auc:.3f}")

# ============================================================================
# 4. MODÈLE 3: PRÉDICTION DE L'AMPLITUDE
# ============================================================================
print("\n[4/7] Entraînement du modèle d'amplitude...")

features_reg = ['maxis', 'dist', 'azimuth', 'airport_encoded', 
                'month', 'hour', 'dayofweek', 'icloud_encoded']
X_reg = data[features_reg].dropna()
y_reg = data.loc[X_reg.index, 'amplitude_abs']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled = scaler_reg.transform(X_test_r)

amp_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
amp_model.fit(X_train_r_scaled, y_train_r)

from sklearn.metrics import r2_score, mean_absolute_error
amp_r2 = r2_score(y_test_r, amp_model.predict(X_test_r_scaled))
amp_mae = mean_absolute_error(y_test_r, amp_model.predict(X_test_r_scaled))

print(f"  ✓ Modèle amplitude: R²={amp_r2:.3f}, MAE={amp_mae:.2f} kA")

# ============================================================================
# 5. MODÈLE 4: PRÉDICTION DU NOMBRE D'ÉCLAIRS (SÉRIE TEMPORELLE)
# ============================================================================
print("\n[5/7] Entraînement du modèle série temporelle...")

daily_alerts = data.groupby(data['date'].dt.date).size().reset_index()
daily_alerts.columns = ['date', 'count']
daily_alerts['date'] = pd.to_datetime(daily_alerts['date'])
daily_alerts = daily_alerts.set_index('date')

train_size = int(len(daily_alerts) * 0.8)
train_ts = daily_alerts['count'][:train_size]
test_ts = daily_alerts['count'][train_size:]

model_arima = ARIMA(train_ts, order=(5, 1, 2))
model_arima_fit = model_arima.fit()

predictions_arima = model_arima_fit.forecast(steps=len(test_ts))
arima_rmse = np.sqrt(((test_ts - predictions_arima) ** 2).mean())

print(f"  ✓ Modèle série temporelle: RMSE={arima_rmse:.2f} éclairs/jour")

# ============================================================================
# 6. FONCTION DE PRÉDICTION COMBINÉE
# ============================================================================
print("\n[6/7] Création de la fonction de prédiction...")

def predire_prochain_eclair(airport, date_heure=None):
    """
    Prédit les caractéristiques du prochain éclair.
    
    Paramètres:
    -----------
    airport : str - Nom de l'aéroport
    date_heure : datetime - Date et heure de prédiction (par défaut: maintenant)
    
    Retourne:
    --------
    dict - Prédictions: probabilité, type, amplitude estimée
    """
    if date_heure is None:
        date_heure = datetime.now()
    
    # Encoder l'aéroport
    if airport in le_airport.classes_:
        airport_code = le_airport.transform([airport])[0]
    else:
        return {"erreur": "Aéroport non reconnu"}
    
    # Préparer les features temporelles
    hour = date_heure.hour
    month = date_heure.month
    dayofweek = date_heure.weekday()
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # 1. Prédire la probabilité d'éclair
    if prob_model is not None:
        features_prob = np.array([[hour_sin, hour_cos, month_sin, month_cos, 
                                    dayofweek, airport_code, hour]])
        prob_eclair = prob_model.predict_proba(features_prob)[0][1]
    else:
        # Utiliser la fréquence historique
        prob_eclair = hourly_data['has_alert'].mean()
    
    # 2. Prédire le type d'éclair (si éclair probable)
    if prob_eclair > 0.3:
        # Features pour le type
        features_type = np.array([[30, 30, 10, 180, airport_code, 
                                   month, hour, dayofweek]])  # Valeurs moyennes
        features_type_scaled = scaler_class.transform(features_type)
        type_proba = type_model.predict_proba(features_type_scaled)[0]
        type_predit = type_model.predict(features_type_scaled)[0]
        type_nom = "Cloud-to-Ground" if type_predit == 1 else "Intra-Cloud"
        
        # 3. Prédire l'amplitude
        features_amp = np.array([[30, 10, 180, airport_code, month, hour, 
                                  dayofweek, type_predit]])
        features_amp_scaled = scaler_reg.transform(features_amp)
        amplitude_predite = amp_model.predict(features_amp_scaled)[0]
    else:
        type_nom = "N/A"
        amplitude_predite = 0
        type_proba = [0.5, 0.5]
    
    # 4. Prédire le nombre d'éclairs aujourd'hui
    forecast_next_7days = model_arima_fit.forecast(steps=7)
    moyenne_prevue = forecast_next_7days.mean()
    
    return {
        'aeroport': airport,
        'date_heure': date_heure.strftime("%Y-%m-%d %H:%M"),
        'probabilite_eclair': f"{prob_eclair*100:.1f}%",
        'type_eclair': type_nom,
        'type_probabilites': {
            'Intra-Cloud': f"{type_proba[0]*100:.1f}%",
            'Cloud-to-Ground': f"{type_proba[1]*100:.1f}%"
        },
        'amplitude_estimee_kA': f"{amplitude_predite:.1f}",
        'eclairs_prevus_7_jours': f"{moyenne_prevue:.1f}/jour",
        'niveau_risque': 'ÉLEVÉ' if prob_eclair > 0.6 else ('MOYEN' if prob_eclair > 0.3 else 'FAIBLE')
    }

# ============================================================================
# 7. TEST ET DÉMONSTRATION
# ============================================================================
print("\n[7/7] Test des prédictions...")

print("\n" + "=" * 80)
print("RÉSULTATS DES PRÉDICTIONS COMBINÉES")
print("=" * 80)

# Tester pour chaque aéroport
for airport in le_airport.classes_[:3]:  # Premiers 3 aéroports
    print(f"\n{'─'*60}")
    result = predire_prochain_eclair(airport)
    print(f"📍 Aéroport: {result['aeroport']}")
    print(f"   Date/Heure: {result['date_heure']}")
    print(f"   ⚡ Probabilité d'éclair: {result['probabilite_eclair']}")
    print(f"   🎯 Type prévu: {result['type_eclair']}")
    print(f"   📊 Amplitude estimée: {result['amplitude_estimee_kA']} kA")
    print(f"   📅 Prévision 7 jours: {result['eclairs_prevus_7_jours']}")
    print(f"   🚨 Niveau de risque: {result['niveau_risque']}")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RÉSUMÉ DU MODÈLE COMBINÉ")
print("=" * 80)

print(f"""
MODÈLE FINAL DE PRÉDICTION D'ÉCLAIR
====================================

Ce modèle combine 4 approches complémentaires:

1. 📊 MODÈLE DE PROBABILITÉ (Classification)
   - Prédit si un éclair va se produire
   - Features: heure, mois, jour, aéroport
   - Performance: Accuracy = {prob_accuracy:.1%}

2. 🎯 MODÈLE DE TYPE (Classification)
   - Prédit le type d'éclair (Cloud-to-Ground / Intra-Cloud)
   - Features: amplitude, distance, azimuth, aéroport
   - Performance: F1-Score = {type_f1:.1%}

3. 📈 MODÈLE D'AMPLITUDE (Régression)
   - Prédit l'intensité de l'éclair en kA
   - Features: maxis, distance, type, aéroport
   - Performance: R² = {amp_r2:.1%}

4. 📅 MODÈLE TEMPOREL (Série temporelle)
   - Prédit le nombre d'éclairs à venir
   - Modèle: ARIMA(5,1,2)
   - Performance: RMSE = {arima_rmse:.1f} éclairs/jour

COMMENT UTILISER:
-----------------
result = predire_prochain_eclair("LFPO")  # Aéroport
print(result)

Affiche:
- Probabilité d'éclair
- Type d'éclair prévu
- Amplitude estimée
- Niveau de risque
""")

print("\n✅ Modèle final prêt à l'emploi!")


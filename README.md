# 🏆 Data Battle IA PAU 2026 – Prédiction de fin d'alerte foudre

## 👥 Équipe
- Nom de l'équipe : FLYKDATA
- Membres :
  - Lilian HURST
  - Keli KEKELI
  - Farouk GNANKAMBARY
  - Yanis BOUZID

## 🎯 Problématique
Prédire avec précision le moment de fin d'une alerte foudre aux aéroports. Les alertes sont déclenchées lors de la détection d'éclairs et persistent pendant 30 minutes sans activité. L'objectif est de minimiser le temps d'évacuation inutile (gain) tout en limitant les faux négatifs (risque) en dessous de 2%.

**Défi**: Capturer la dynamique temporelle complexe des événements de foudre et fournir des prédictions fiables avec intervalles de confiance.

## 💡 Solution proposée
Approche **multi-modèle hybride avec stacking** :

1. **Ingestion & Preprocessing**
   - Chargement des données d'éclairs par aéroport et alert_id
   - Split 80/20 temporel (train/validation)
   - Génération de snapshots toutes les 2 minutes avec features enrichies

2. **Feature Engineering**
   - Features globales d'alerte : durée, nb d'éclairs, amplitude, distance, % cloud-to-ground
   - Clustering K-means (5 clusters) pour stratification
   - Features temporelles : taux d'éclairs (1/2/5/10/20 min), inter-arrivals, ratios de décroissance
   - Features spatio-temporelles : tendances amplitude/distance, vitesse d'approche

3. **Modèles d'ensemble**
   - **Modèle principal** : XGBoost MAE optimisé
   - **Modèles quantiles** : XGBoost Q10 et Q95 pour intervalles de confiance
   - **Modèles spécialisés** : XGBoost par cluster
   - **Meta-learner** : XGBoost stacking combinant tous les modèles

4. **Calibration & Scoring**
   - Score de confiance basé sur : gap ratio, temps depuis dernier éclair, taux d'activité, intervalle de prédiction, progression vers durée médiane du cluster
   - Optimisation du seuil θ sur la validation pour maximiser le gain sous contrainte R < 2%

5. **Évaluation Jury**
   - Métrique gain : heures économisées (end_baseline - end_predicted)
   - Métrique risque : fraction d'éclairs ≤ MIN_DIST_KM manqués (>end_predicted)

## ⚙️ Stack technique
- **Langages** : Python 3.8+
- **Frameworks ML** : XGBoost, scikit-learn
- **Data** : Pandas, NumPy
- **Clustering** : scikit-learn KMeans
- **Métriques** : scikit-learn MAE
- **Versioning modèles** : joblib
- **Tracking éco** : CodeCarbon (émissions CO₂)
- **Visualisation** : (optionnel) Matplotlib, Seaborn

## 🚀 Installation & exécution

### Prérequis
- Python 3.8 ou supérieur
- pip ou conda

### Installation

```bash
# 1. Cloner ou accéder au dossier du projet
cd DataBattle

# 2. Installer les dépendances
pip install numpy pandas scikit-learn xgboost joblib tqdm codecarbon

# Optionnel : pour GPU (XGBoost)
pip install xgboost-gpu
```

### Fichiers d'entrée requis
Placez ces fichiers dans le répertoire `DataBattle/` :
- `segment_alerts_all_airports_train.csv` - Données d'entraînement (éclairs + airport_alert_id)
- `segment_alerts_all_airports_eval.csv` - Données de test/validation (éclairs + airport_alert_id)

**Colonnes requises** :
```
lightning_id, lightning_airport_id, date, lon, lat, amplitude, 
maxis, icloud, dist, azimuth, airport, airport_alert_id, 
is_last_lightning_cloud_ground
```

### Exécution

```bash
# Entraîner le modèle complet et générer prédictions
python model.py
```

**Durée** : ~10-30 minutes (dépend du dataset et de la machine)

**Sortie** :
```
segment_alerts_valid.csv           → Validation split
predictions_val.csv                → Prédictions validation (prediction_date, predicted_date_end_alert, confidence)
predictions_test.csv               → Prédictions test (prediction_date, true_end_alert, predicted_date_end_alert, confidence)
co2_emissions.json                 → Émissions CO₂ (entraînement + inférence)
model_main.pkl                     → Modèle principal (sauvegardé)
model_low.pkl, model_high.pkl      → Modèles quantiles
cluster_models.pkl                 → Modèles par cluster
```

### Configuration

Ajustez les paramètres dans `model.py` (section CONFIGURATION) :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `MIN_ELAPSED` | 5.0 | Temps minimal (min) avant première prédiction |
| `MAX_GAP_MINUTES` | 30 | Durée du gap post-éclair définissant fin d'alerte |
| `MIN_DIST_KM` | 3 | Rayon (km) pour comptabiliser éclairs dangereux |
| `N_CLUSTERS` | 5 | Nombre de clusters K-means |
| `SNAPSHOT_STEP_MINUTES` | 2 | Pas temporel de snapshot |
| `QUANTILE_LOW` | 0.10 | Quantile bas pour intervalle de confiance |
| `QUANTILE_HIGH` | 0.95 | Quantile haut pour intervalle de confiance |
| `R_ACCEPT` | 0.02 | Seuil risque maximal toléré (2%) |

## 🌍 Tracking CO₂ et Impact Environnemental

Le modèle utilise **CodeCarbon** pour mesurer l'empreinte carbone :

- **Entraînement** : Suivi des émissions lors de l'apprentissage des 4 modèles XGBoost (main, quantiles bas/haut, cluster)
- **Inférence** : Suivi des émissions lors de la prédiction sur le test set

**Résultats sauvegardés dans `co2_emissions.json`** :
```json
{
  "training_kg_co2": 4.845805922623378e-07,
  "inference_kg_co2": 2.1623245618107487e-06,
  "training_energy_kwh": null,
  "inference_energy_kwh": null,
  "description": "\u00c9missions CO\u2082 pour l'entra\u00eenement et l'inf\u00e9rence sur le jeu de test"
}
```

**Interprétation** :
- `training_kg_co2` : CO₂ produit pendant l'entraînement des modèles
- `inference_kg_co2` : CO₂ produit pendant les prédictions sur le test
- Valeurs pour comparaison : 0.01 kg CO₂ ≈ 1 km en voiture

## 📊 Résultats & Métriques

Le script affiche automatiquement :

- **MAE validation** (min) : 3 modèles (global, cluster, stacking)
- **MAE test** : comparaison avec validation (surapprentissage ?)
- **Theta optimal** : seuil de confiance maximisant le gain
- **Gain/Risque** : heures économisées et fraction d'éclairs manqués
- **Performance test** : application du theta validation sur test

Exemple de sortie :
```
MAE validation STACKING : 12.34 min
Theta optimal (validation) : 0.64
Gain validation : 59.4 heures
Risque validation : 0.0053 (✓ < 0.02)
Gain test : 167.4 heures
Risque test : 0.0182 (✓ < 0.02)
```

## 🔍 Détails techniques

### Pipeline

1. **Chargement & Split** → 80% train, 20% validation
2. **Features alertes** → Duration, n_flashes, mean_amplitude, mean_dist, pct_cg
3. **Clustering** → K-means sur features alertes standardisées
4. **Snapshots** → Génération temporelle avec features enrichies
5. **Filtrage** → Suppression snapshots time_elapsed < 5 min
6. **Entraînement** → 4 modèles XGBoost + 1 meta-learner
7. **Prédictions** → Expm1 (inverse log1p) sur outputs log
8. **Calibration** → Score de confiance + optimisation theta
9. **Export** → CSV predictions + modèles .pkl

### Architecture modèles

```
𝐗_train (features) ──→ [Modèle Principal (MAE)] ──→ preds_main
                     ├→ [Modèle Q10]           ──→ interval_low
                     ├→ [Modèle Q95]           ──→ interval_high
                     └→ [5× Modèles Cluster]   ──→ preds_cluster
                     
{preds_main, preds_cluster, preds_low, preds_high, features} 
                     ──→ [Meta-learner XGBoost] ──→ t_combined (final)
```

### Optimisation de theta

Scan linéaire de θ ∈ [0.01, 0.99] par pas 0.01 :
- Pour chaque θ : filter predictions avec confidence ≥ θ
- Calculer gain et risque
- Sélectionner (θ, gain) maximisant gain sous R < 0.02

## ⚠️ Limitations & Améliorations futures

**Limitations actuelles** :
- Alertes ultra-courtes (< 5 min) non prédites car MIN_ELAPSED = 5.0
- Features statiques (pas de séquences) → RNN/LSTM pour évolution temporelle
- One-hot encoding airport → CatBoost pour traitement natif
- Seuil θ optimal peut drift sur nouveau test
- CodeCarbon ne capture que l'énergie CPU/GPU (pas mémoire complète)

**Améliorations possibles** :
1. Essayer LightGBM / CatBoost / Ensemble
2. Isotonic Regression pour meilleure calibration confiance
3. Modèle séparé pour alertes courtes
4. Features géographiques avancées (proximité côte, altitude)
5. Tracking drift en production

## � Fichiers de sortie

Après exécution de `model.py`, les fichiers suivants sont générés :

| Fichier | Description |
|---------|-------------|
| `segment_alerts_valid.csv` | Split validation (80/20 temporel) |
| `predictions_val.csv` | Prédictions sur validation (prediction_date, predicted_date_end_alert, confidence) |
| `predictions_test.csv` | Prédictions sur test (prediction_date, true_end_alert, predicted_date_end_alert, confidence) |
| `co2_emissions.json` | Émissions CO₂ entraînement + inférence (kg CO₂eq) |
| `model_main.pkl` | Modèle XGBoost principal (MAE) |
| `model_low.pkl` | Modèle XGBoost quantile Q10 |
| `model_high.pkl` | Modèle XGBoost quantile Q95 |
| `cluster_models.pkl` | Dictionnaire des 5 modèles spécialisés par cluster |

## �📝 Licence & Auteurs
Data Battle IA PAU 2026 - Tous droits réservés

---
**Dernière mise à jour** : 26 mars 2026

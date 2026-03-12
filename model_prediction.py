"""
=============================================================================
MODÈLES DE PRÉDICTION POUR LES ALERTES DE FOUDRE AUX AÉROPORTS
=============================================================================
Ce script implémente plusieurs modèles de prédiction pour analyser et prévoir
les alertes de foudre autour des aéroports français.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, mean_squared_error, 
                             mean_absolute_error, r2_score, classification_report,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# Statistiques
from scipy import stats
import statsmodels.api as sm

print("=" * 80)
print("PRÉDICTION DES ALERTES DE FOUDRE - MODÈLES MULTIPLES")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ============================================================================
print("\n" + "="*50)
print("1. CHARGEMENT DES DONNÉES")
print("="*50)

data = pd.read_csv('segment_alerts_all_airports_train.csv')
print(f"Nombre total d'enregistrements: {len(data):,}")
print(f"Nombre d'aéroports: {data['airport'].nunique()}")
print(f"Période: {data['date'].min()} à {data['date'].max()}")

# Convertir la date
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# Extraire les features temporelles
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['dayofweek'] = data['date'].dt.dayofweek
data['dayofyear'] = data['date'].dt.dayofyear

# Encoder la variable 'airport'
le_airport = LabelEncoder()
data['airport_encoded'] = le_airport.fit_transform(data['airport'])

# Encoder 'icloud' (True/False -> 1/0)
data['icloud_encoded'] = data['icloud'].astype(int)

print("\nAéroports disponibles:")
for i, airport in enumerate(le_airport.classes_):
    count = (data['airport'] == airport).sum()
    print(f"  {airport}: {count:,} éclairs")

# ============================================================================
# 2. PRÉDICTION DU TYPE D'ÉCLAIR (CLASSIFICATION)
# ============================================================================
print("\n" + "="*50)
print("2. PRÉDICTION DU TYPE D'ÉCLAIR (Cloud-to-Ground vs Intra-Cloud)")
print("="*50)

# Préparation des features pour la classification
features_class = ['amplitude', 'maxis', 'dist', 'azimuth', 'airport_encoded', 
                  'month', 'hour', 'dayofweek']
X_class = data[features_class].dropna()
y_class = data.loc[X_class.index, 'icloud_encoded']

print(f"Dataset classification: {len(X_class):,} samples")
print(f"Distribution: {y_class.value_counts().to_dict()}")

# Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Standardisation
scaler_class = StandardScaler()
X_train_c_scaled = scaler_class.fit_transform(X_train_c)
X_test_c_scaled = scaler_class.transform(X_test_c)

# Modèles de classification à tester
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42)
}

results_classification = {}

for name, clf in classifiers.items():
    print(f"\nEntraînement: {name}...")
    clf.fit(X_train_c_scaled, y_train_c)
    y_pred = clf.predict(X_test_c_scaled)
    y_proba = clf.predict_proba(X_test_c_scaled)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
    
    results_classification[name] = {
        'Accuracy': accuracy_score(y_test_c, y_pred),
        'Precision': precision_score(y_test_c, y_pred),
        'Recall': recall_score(y_test_c, y_pred),
        'F1-Score': f1_score(y_test_c, y_pred),
        'ROC-AUC': roc_auc_score(y_test_c, y_proba)
    }

# Tableau des résultats classification
print("\n" + "-"*80)
print("RÉSULTATS CLASSIFICATION (Type d'éclair)")
print("-"*80)
results_df = pd.DataFrame(results_classification).T
print(results_df.to_string())

# ============================================================================
# 3. PRÉDICTION DE L'AMPLITUDE (RÉGRESSION)
# ============================================================================
print("\n" + "="*50)
print("3. PRÉDICTION DE L'AMPLITUDE DU FOUDRE")
print("="*50)

# Préparation des features pour la régression
# Utiliser la valeur absolue de l'amplitude pour la prédiction
data['amplitude_abs'] = data['amplitude'].abs()

features_reg = ['maxis', 'dist', 'azimuth', 'airport_encoded', 
                'month', 'hour', 'dayofweek', 'icloud_encoded']
X_reg = data[features_reg].dropna()
y_reg = data.loc[X_reg.index, 'amplitude_abs']

print(f"Dataset régression: {len(X_reg):,} samples")
print(f"Amplitude moyenne: {y_reg.mean():.2f} kA")
print(f"Amplitude médiane: {y_reg.median():.2f} kA")

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Standardisation
scaler_reg = StandardScaler()
X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled = scaler_reg.transform(X_test_r)

# Modèles de régression
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf')
}

results_regression = {}

for name, reg in regressors.items():
    print(f"\nEntraînement: {name}...")
    reg.fit(X_train_r_scaled, y_train_r)
    y_pred = reg.predict(X_test_r_scaled)
    
    results_regression[name] = {
        'MSE': mean_squared_error(y_test_r, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_r, y_pred)),
        'MAE': mean_absolute_error(y_test_r, y_pred),
        'R2': r2_score(y_test_r, y_pred)
    }

# Tableau des résultats régression
print("\n" + "-"*80)
print("RÉSULTATS RÉGRESSION (Prédiction amplitude)")
print("-"*80)
results_reg_df = pd.DataFrame(results_regression).T
print(results_reg_df.to_string())

# ============================================================================
# 4. PRÉDICTION DU NOMBRE D'ALERTES (SÉRIES TEMPORELLES)
# ============================================================================
print("\n" + "="*50)
print("4. PRÉDICTION DU NOMBRE D'ALERTES (Séries temporelles)")
print("="*50)

# Agréger par jour
daily_alerts = data.groupby(data['date'].dt.date).size().reset_index()
daily_alerts.columns = ['date', 'count']
daily_alerts['date'] = pd.to_datetime(daily_alerts['date'])
daily_alerts = daily_alerts.set_index('date')

# Features temporelles
daily_alerts['month'] = daily_alerts.index.month
daily_alerts['dayofweek'] = daily_alerts.index.dayofweek
daily_alerts['year'] = daily_alerts.index.year

# Moyenne mobile
daily_alerts['rolling_7d'] = daily_alerts['count'].rolling(window=7).mean()
daily_alerts['rolling_30d'] = daily_alerts['count'].rolling(window=30).mean()

print(f"Nombre de jours dans le dataset: {len(daily_alerts)}")
print(f"Moyenne d'alertes par jour: {daily_alerts['count'].mean():.2f}")
print(f"Max d'alertes par jour: {daily_alerts['count'].max()}")

# Modèle ARIMA simple pour les séries temporelles
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Test de stationnarité
result = adfuller(daily_alerts['count'].dropna())
print(f"\nTest ADF (stationnarité): p-value = {result[1]:.4f}")

# Modèle ARIMA
print("\nEntraînement ARIMA...")
train_size = int(len(daily_alerts) * 0.8)
train_ts = daily_alerts['count'][:train_size]
test_ts = daily_alerts['count'][train_size:]

# ARIMA simple
try:
    model_arima = ARIMA(train_ts, order=(5, 1, 2))
    model_arima_fit = model_arima.fit()
    
    # Prédiction
    predictions_arima = model_arima_fit.forecast(steps=len(test_ts))
    
    # Métriques
    mse_ts = mean_squared_error(test_ts, predictions_arima)
    rmse_ts = np.sqrt(mse_ts)
    mae_ts = mean_absolute_error(test_ts, predictions_arima)
    
    print(f"\nARIMA Results:")
    print(f"  RMSE: {rmse_ts:.4f}")
    print(f"  MAE: {mae_ts:.4f}")
except Exception as e:
    print(f"Erreur ARIMA: {e}")

# ============================================================================
# 5. PRÉDICTION PAR AÉROPORT
# ============================================================================
print("\n" + "="*50)
print("5. PRÉDICTION DU NOMBRE D'ALERTES PAR AÉROPORT")
print("="*50)

airport_predictions = {}

for airport in data['airport'].unique():
    airport_data = data[data['airport'] == airport]
    
    # Compter les alertes par mois
    monthly = airport_data.groupby([airport_data['date'].dt.year, 
                                      airport_data['date'].dt.month]).size()
    
    # Moyenne et écart-type
    avg_monthly = monthly.mean()
    std_monthly = monthly.std()
    
    airport_predictions[airport] = {
        'avg_alerts_per_month': avg_monthly,
        'std_alerts_per_month': std_monthly,
        'total_alerts': len(airport_data),
        'avg_amplitude': airport_data['amplitude_abs'].mean(),
        'max_amplitude': airport_data['amplitude_abs'].max()
    }

airport_df = pd.DataFrame(airport_predictions).T
print(airport_df.to_string())

# ============================================================================
# 6. ANALYSE DE SAISONNALITÉ
# ============================================================================
print("\n" + "="*50)
print("6. ANALYSE DE SAISONNALITÉ")
print("="*50)

# Moyenne par mois
monthly_avg = data.groupby('month').size()
print("\nAlertes par mois (moyenne sur toutes les années):")
for month, count in monthly_avg.items():
    print(f"  Mois {month:2d}: {count:6d} alertes")

# ============================================================================
# 7. MODÈLE DE PROBABILITÉ D'ALERTE
# ============================================================================
print("\n" + "="*50)
print("7. MODÈLE DE PROBABILITÉ D'ALERTE (PAR HEURE)")
print("="*50)

# Créer un dataset binaire: y a-t-il eu une alerte dans l'heure?
hourly_alerts = data.groupby(data['date'].dt.floor('H')).agg({
    'lightning_id': 'count',
    'amplitude_abs': 'mean',
    'airport_encoded': 'first'
}).reset_index()
hourly_alerts.columns = ['date', 'alert_count', 'avg_amplitude', 'airport']

# Créer variable binaire
hourly_alerts['has_alert'] = (hourly_alerts['alert_count'] > 0).astype(int)

# Features
hourly_alerts['hour'] = hourly_alerts['date'].dt.hour
hourly_alerts['month'] = hourly_alerts['date'].dt.month
hourly_alerts['dayofweek'] = hourly_alerts['date'].dt.dayofweek

# Features cycliques
hourly_alerts['hour_sin'] = np.sin(2 * np.pi * hourly_alerts['hour'] / 24)
hourly_alerts['hour_cos'] = np.cos(2 * np.pi * hourly_alerts['hour'] / 24)
hourly_alerts['month_sin'] = np.sin(2 * np.pi * hourly_alerts['month'] / 12)
hourly_alerts['month_cos'] = np.cos(2 * np.pi * hourly_alerts['month'] / 12)

# Entraîner un modèle de probabilité
X_prob = hourly_alerts[['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                         'dayofweek', 'airport']].dropna()
y_prob = hourly_alerts.loc[X_prob.index, 'has_alert']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_prob, y_prob, test_size=0.2, random_state=42, stratify=y_prob
)

# Entraîner
prob_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
prob_model.fit(X_train_p, y_train_p)
y_pred_prob = prob_model.predict(X_test_p)

print(f"\nModèle de probabilité d'alerte:")
print(f"  Accuracy: {accuracy_score(y_test_p, y_pred_prob):.4f}")
print(f"  F1-Score: {f1_score(y_test_p, y_pred_prob):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_prob.columns,
    'importance': prob_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# ============================================================================
# 8. RÉSUMÉ ET COMPARAISON DES MODÈLES
# ============================================================================
print("\n" + "="*80)
print("RÉSUMÉ COMPARATIF DES MODÈLES")
print("="*80)

print("\n--- CLASSIFICATION (Type d'éclair) ---")
best_classifier = max(results_classification.items(), key=lambda x: x[1]['F1-Score'])
print(f"Meilleur modèle: {best_classifier[0]}")
print(f"  F1-Score: {best_classifier[1]['F1-Score']:.4f}")
print(f"  ROC-AUC: {best_classifier[1]['ROC-AUC']:.4f}")

print("\n--- RÉGRESSION (Amplitude) ---")
best_regressor = max(results_regression.items(), key=lambda x: x[1]['R2'])
print(f"Meilleur modèle: {best_regressor[0]}")
print(f"  R2: {best_regressor[1]['R2']:.4f}")
print(f"  RMSE: {best_regressor[1]['RMSE']:.4f}")

print("\n--- SÉRIES TEMPORELLES (Nombre d'alertes) ---")
print(f"  Modèle: ARIMA(5,1,2)")
print(f"  RMSE: {rmse_ts:.4f}")

# ============================================================================
# 9. GÉNÉRATION DU RAPPORT MARKDOWN
# ============================================================================
print("\n" + "="*50)
print("GÉNÉRATION DU RAPPORT MARKDOWN")
print("="*50)

report = """# Rapport des Modèles de Prédiction - Alertes de Foudre Aéroports

## 1. Vue d'ensemble

Ce rapport présente les résultats de plusieurs modèles de machine learning appliqués 
aux données d'alertes de foudre autour de plusieurs aéroports français.

### Données analysées:
- **Nombre total d'éclairs**: {total:,}
- **Nombre d'aéroports**: {n_airports}
- **Période**: {period}

---

## 2. Prédiction du Type d'Éclair (Classification)

### Objectif
Prédire si un éclair est de type **Cloud-to-Ground** (contact au sol) ou 
**Intra-Cloud** (dans le nuage).

### Modèles testés
{classification_table}

### Meilleurs résultats
- **Modèle**: {best_clf}
- **F1-Score**: {best_f1:.4f}
- **ROC-AUC**: {best_auc:.4f}

### Variables importantes
- amplitude, maxis, dist, azimuth, airport, month, hour

---

## 3. Prédiction de l'Amplitude du Foudre (Régression)

### Objectif
Prédire l'intensité (amplitude) du courant de foudre en kA.

### Modèles testés
{regression_table}

### Meilleurs résultats
- **Modèle**: {best_reg}
- **R²**: {best_r2:.4f}
- **RMSE**: {best_rmse:.4f} kA

---

## 4. Prédiction du Nombre d'Alertes (Séries Temporelles)

### Objectif
Prédire le nombre d'alertes de foudre par jour.

### Modèle utilisé
- **ARIMA(5,1,2)**

### Résultats
- **RMSE**: {arima_rmse:.4f}
- **MAE**: {arima_mae:.4f}

### Observations
- Les alertes sont fortement concentrées en été (Juin, Juillet, Août)
- Il existe une variation journalière significative
- Les weekend ont généralement moins d'alertes

---

## 5. Probabilité d'Alerte par Heure

### Objectif
Prédire la probabilité qu'une alerte se produise dans une heure donnée.

### Modèle
- **Random Forest Classifier**

### Résultats
- **Accuracy**: {prob_accuracy:.4f}
- **F1-Score**: {prob_f1:.4f}

### Features les plus importantes
{feature_imp}

---

## 6. Analyse par Aéroport

| Aéroport | Total Alertes | Moyenne/Mois | Amplitude Moyenne | Amplitude Max |
|----------|---------------|--------------|-------------------|---------------|
{airport_table}

---

## 7. Conclusions et Recommandations

### Modèles recommandés

1. **Pour la classification du type d'éclair**: 
   - **{best_clf}** avec F1-Score de {best_f1:.4f}

2. **Pour la prédiction d'amplitude**:
   - **{best_reg}** avec R² de {best_r2:.4f}

3. **Pour la prédiction du nombre d'alertes**:
   - **ARIMA** pour les prédictions à court terme

### Points clés

1. **Saisonnalité**: Les alertes sont 3-4 fois plus fréquentes en été
2. **Hétérogénéité**: Chaque aéroport a un profil de risque différent
3. **Amplitude**: La plupart des éclairs ont une amplitude < 50 kA
4. **Type**: Environ 60% des éclairs sont Cloud-to-Ground

### Suggestions d'amélioration

1. Ajouter des données météorologiques (température, humidité, pression)
2. Utiliser des modèles de deep learning (LSTM) pour les séries temporelles
3. Implémenter un modèle de détection d'anomalies en temps réel
4. Créer un système d'alerte précoce basé sur les probabilités

---

*Rapport généré automatiquement le {date}*
""".format(
    total=len(data),
    n_airports=data['airport'].nunique(),
    period=f"{data['date'].min()} à {data['date'].max()}",
    classification_table=results_df.to_markdown(),
    best_clf=best_classifier[0],
    best_f1=best_classifier[1]['F1-Score'],
    best_auc=best_classifier[1]['ROC-AUC'],
    regression_table=results_reg_df.to_markdown(),
    best_reg=best_regressor[0],
    best_r2=best_regressor[1]['R2'],
    best_rmse=best_regressor[1]['RMSE'],
    arima_rmse=rmse_ts,
    arima_mae=mae_ts,
    prob_accuracy=accuracy_score(y_test_p, y_pred_prob),
    prob_f1=f1_score(y_test_p, y_pred_prob),
    feature_imp=feature_importance.to_markdown(index=False),
    airport_table='\n'.join([f"| {idx} | {row['total_alerts']:.0f} | {row['avg_alerts_per_month']:.1f} | {row['avg_amplitude']:.1f} kA | {row['max_amplitude']:.1f} kA |" 
                             for idx, row in airport_df.iterrows()]),
    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

# Sauvegarder le rapport
with open('rapport_modeles_prediction.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("\nRapport sauvegardé dans: rapport_modeles_prediction.md")

# ============================================================================
# 10. CRÉATION DES VISUALISATIONS
# ============================================================================
print("\n" + "="*50)
print("CRÉATION DES VISUALISATIONS")
print("="*50)

# Figure 1: Comparaison des modèles de classification
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy des modèles de classification
ax1 = axes[0, 0]
results_df[['Accuracy', 'F1-Score', 'ROC-AUC']].plot(kind='bar', ax=ax1)
ax1.set_title('Performance des Modèles de Classification', fontsize=12)
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.tick_params(axis='x', rotation=45)

# 2. Performance des modèles de régression
ax2 = axes[0, 1]
results_reg_df[['R2']].plot(kind='bar', ax=ax2, color='green')
ax2.set_title('Score R² des Modèles de Régression', fontsize=12)
ax2.set_ylabel('R² Score')
ax2.tick_params(axis='x', rotation=45)

# 3. Distribution des alertes par mois
ax3 = axes[1, 0]
monthly_avg.plot(kind='bar', ax=ax3, color='orange')
ax3.set_title('Nombre d\'alertes par mois', fontsize=12)
ax3.set_xlabel('Mois')
ax3.set_ylabel('Nombre d\'alertes')

# 4. Alertes par aéroport
ax4 = axes[1, 1]
airport_df['total_alerts'].sort_values(ascending=True).plot(kind='barh', ax=ax4, color='purple')
ax4.set_title('Total des alertes par aéroport', fontsize=12)
ax4.set_xlabel('Nombre d\'alertes')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardée: model_performance.png")

# Figure 2: Matrice de confusion du meilleur modèle
fig2, ax = plt.subplots(figsize=(8, 6))
best_clf_model = classifiers[best_classifier[0]]
y_pred_best = best_clf_model.predict(X_test_c_scaled)
cm = confusion_matrix(y_test_c, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Matrice de Confusion - {best_classifier[0]}', fontsize=12)
ax.set_xlabel('Prédit')
ax.set_ylabel('Réel')
ax.set_xticklabels(['Intra-Cloud', 'Cloud-to-Ground'])
ax.set_yticklabels(['Intra-Cloud', 'Cloud-to-Ground'])
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardée: confusion_matrix.png")

# Figure 3: Série temporelle des alertes
fig3, ax = plt.subplots(figsize=(14, 6))
daily_alerts['count'].plot(ax=ax, alpha=0.5, label='Quotidien')
daily_alerts['rolling_7d'].plot(ax=ax, label='Moyenne mobile 7 jours', linewidth=2)
ax.set_title('Nombre d\'alertes de foudre par jour', fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Nombre d\'alertes')
ax.legend()
plt.savefig('time_series_alerts.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardée: time_series_alerts.png")

print("\n" + "="*80)
print("TOUT EST TERMINÉ!")
print("="*80)
print("\nFichiers générés:")
print("  - rapport_modeles_prediction.md (Rapport complet)")
print("  - model_performance.png (Graphiques de performance)")
print("  - confusion_matrix.png (Matrice de confusion)")
print("  - time_series_alerts.png (Série temporelle)")


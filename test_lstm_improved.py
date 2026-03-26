import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# ============================================================
# CONFIGURATION
# ============================================================
SEQUENCE_LENGTH = 15
MIN_ELAPSED = 5.0
MAX_GAP_MINUTES = 30
MIN_DIST_KM = 3
R_ACCEPT = 0.02
SNAPSHOT_STEP_MINUTES = 2
MAX_ELAPSED_MINUTES = 45

FEATURE_COLS = [
    'mean_amp_sofar', 'mean_dist_sofar', 'pct_cg_sofar',
    'rate_1min', 'rate_5min', 'rate_10min', 'rate_20min',
    'mean_inter_time', 'gap_ratio', 'last_inter_time'
]

# ============================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES DE TEST
# ============================================================
print("=" * 70)
print("1. Chargement des données de test")
print("=" * 70)

df_test = pd.read_csv('segment_alerts_all_airports_eval.csv', parse_dates=['date'])
df_test = df_test[df_test['airport_alert_id'].notna()].copy()
print(f"Test : {len(df_test):,} éclairs | {df_test['airport_alert_id'].nunique():,} alertes")

# ============================================================
# 2. GÉNÉRER LES SNAPSHOTS DE TEST
# ============================================================
def compute_alert_features(df_raw):
    return df_raw.groupby(['airport', 'airport_alert_id']).agg(
        duration_min   = ('date', lambda x: (x.max() - x.min()).total_seconds() / 60),
        n_flashes      = ('lightning_id', 'count'),
        mean_amplitude = ('amplitude', 'mean'),
        mean_dist      = ('dist', 'mean'),
        pct_cg         = ('icloud', lambda x: (~x).mean()),
    ).reset_index()

def generate_snapshots(df_alert, step_minutes=SNAPSHOT_STEP_MINUTES, max_elapsed=MAX_ELAPSED_MINUTES):
    alert_groups = df_alert.groupby(['airport', 'airport_alert_id'])
    snapshots = []

    for (airport, alert_id), group in tqdm(alert_groups, desc="Snapshots test", disable=True):
        group = group.sort_values('date').reset_index(drop=True).copy()
        start = group['date'].iloc[0]
        end   = group['date'].iloc[-1]
        current = start
        idx_end = 0

        while current <= end:
            while idx_end + 1 < len(group) and group['date'].iloc[idx_end + 1] <= current:
                idx_end += 1

            past = group.iloc[:idx_end + 1]
            if len(past) == 0:
                current += timedelta(minutes=step_minutes)
                continue

            time_elapsed = (current - start).total_seconds() / 60.0
            if time_elapsed > max_elapsed:
                break

            remaining = (end - current).total_seconds() / 60.0
            n_so_far = len(past)

            mean_amp_sofar = past['amplitude'].mean()
            mean_dist_sofar = past['dist'].mean()
            pct_cg_sofar = (~past['icloud']).mean()
            last_dist = past.iloc[-1]['dist']

            t1  = current - timedelta(minutes=1)
            t5  = current - timedelta(minutes=5)
            t10 = current - timedelta(minutes=10)
            t20 = current - timedelta(minutes=20)

            w1  = past[past['date'] >= t1]
            w5  = past[past['date'] >= t5]
            w10 = past[past['date'] >= t10]
            w20 = past[past['date'] >= t20]

            rate_1min  = len(w1) / 1.0
            rate_5min  = len(w5) / 5.0
            rate_10min = len(w10) / 10.0
            rate_20min = len(w20) / 20.0

            if len(past) >= 2:
                inter_times = past['date'].diff().dt.total_seconds().dropna() / 60.0
                mean_inter_time = inter_times.mean()
                last_inter_time = inter_times.iloc[-1]
                gap_ratio = last_inter_time / (inter_times.quantile(0.50) + 1e-6)
            else:
                mean_inter_time = 0.0
                last_inter_time = 0.0
                gap_ratio = 0.0

            snapshots.append({
                'airport': airport,
                'alert_id': alert_id,
                'snapshot_time': current,
                'remaining_min': remaining,
                'time_elapsed': time_elapsed,
                'mean_amp_sofar': mean_amp_sofar,
                'mean_dist_sofar': mean_dist_sofar,
                'pct_cg_sofar': pct_cg_sofar,
                'last_dist': last_dist,
                'rate_1min': rate_1min,
                'rate_5min': rate_5min,
                'rate_10min': rate_10min,
                'rate_20min': rate_20min,
                'mean_inter_time': mean_inter_time,
                'last_inter_time': last_inter_time,
                'gap_ratio': gap_ratio,
            })

            current += timedelta(minutes=step_minutes)

    return pd.DataFrame(snapshots)

print("\n2. Génération snapshots test...")
test_alerts = compute_alert_features(df_test)
test_snapshots = generate_snapshots(df_test)
test_snapshots = test_snapshots.reset_index(drop=True)
test_snapshots = test_snapshots[test_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()

print(f"Test snapshots : {len(test_snapshots):,}")

# ============================================================
# 3. CHARGER LES SCALERS ET CLUSTERING
# ============================================================
print("\n3. Chargement des scalers...")

with open('lstm_scaler_x.pkl', 'rb') as f:
    scaler_x = pickle.load(f)

with open('lstm_scaler_alert.pkl', 'rb') as f:
    scaler_alert = pickle.load(f)

with open('lstm_kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

print("✓ Scalers chargés")

# ============================================================
# 4. PRÉPARER LES SÉQUENCES LSTM
# ============================================================
print("\n4. Préparation séquences...")

sequences_X = []
sequences_meta = []

alert_groups = test_snapshots.groupby(['airport', 'alert_id'])

for (airport, alert_id), group in tqdm(alert_groups, desc="Séquences test"):
    group = group.sort_values('snapshot_time').reset_index(drop=True).copy()
    
    features = group[FEATURE_COLS].values
    
    if len(features) >= SEQUENCE_LENGTH:
        scaler_local = StandardScaler()
        features_scaled = scaler_local.fit_transform(features)
        
        for i in range(len(features_scaled) - SEQUENCE_LENGTH + 1):
            seq = features_scaled[i:i + SEQUENCE_LENGTH]
            sequences_X.append(seq)
            sequences_meta.append({
                'airport': airport,
                'alert_id': alert_id,
                'snapshot_time': group['snapshot_time'].iloc[i + SEQUENCE_LENGTH - 1],
                'true_remaining_min': group['remaining_min'].iloc[i + SEQUENCE_LENGTH - 1],
                'last_dist': group['last_dist'].iloc[i + SEQUENCE_LENGTH - 1],
            })

X_test_seq = np.array(sequences_X)
print(f"X_test shape: {X_test_seq.shape}")
print(f"Nombre de séquences : {len(sequences_meta)}")

# ============================================================
# 5. NORMALISATION
# ============================================================
print("\n5. Normalisation...")

n_features = X_test_seq.shape[2]
X_test_reshaped = X_test_seq.reshape(-1, n_features)
X_test_scaled = scaler_x.transform(X_test_reshaped)
X_test_norm = X_test_scaled.reshape(len(X_test_seq), SEQUENCE_LENGTH, n_features)

print(f"X_test normalized: {X_test_norm.shape}")

# ============================================================
# 6. CHARGER LE MODÈLE
# ============================================================
print("\n6. Chargement du modèle LSTM amélioré...")

try:
    model = load_model('lstm_model_improved.h5')
except (ValueError, TypeError) as e:
    print("Erreur de désérialisation, chargement alternatif...")
    model = load_model('lstm_model_improved.h5', compile=False)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

print("✓ Modèle chargé")

# ============================================================
# 7. PRÉDICTIONS
# ============================================================
print("\n7. Prédictions...")

y_pred_log = model.predict(X_test_norm, verbose=0)
y_pred_raw = np.expm1(y_pred_log.flatten())
y_pred_raw = np.maximum(y_pred_raw, 0.0)

y_true = np.array([m['true_remaining_min'] for m in sequences_meta])

mae_pred = mean_absolute_error(y_true, y_pred_raw)
rmse_pred = np.sqrt(mean_squared_error(y_true, y_pred_raw))

print(f"MAE LSTM amélioré  : {mae_pred:.4f} min")
print(f"RMSE LSTM amélioré : {rmse_pred:.4f} min")

# ============================================================
# 8. CALCUL DU SCORE DE CONFIANCE INTELLIGENT
# ============================================================
print("\n8. Calcul score de confiance...")

# Résidus (erreurs absolues)
residuals = np.abs(y_true - y_pred_raw)
residual_median = np.median(residuals)
residual_p75 = np.percentile(residuals, 75)
residual_p90 = np.percentile(residuals, 90)

# Score basé sur les résidus : plus l'erreur est petite, plus confiance est haute
confidence_from_residual = 1.0 - (residuals / (residual_p90 + 1e-6))
confidence_from_residual = np.clip(confidence_from_residual, 0, 1)

# Facteur de distance (éclairs proches = plus de risque = moins confiance)
last_dists = np.array([m['last_dist'] for m in sequences_meta])
distance_factor = np.clip(last_dists / MIN_DIST_KM, 0, 1)

# Score final
confidence_scores = confidence_from_residual * (0.7) + distance_factor * (0.3)
confidence_scores = np.clip(confidence_scores, 0.01, 0.99)

print(f"Confidence min/max : {confidence_scores.min():.3f} / {confidence_scores.max():.3f}")
print(f"Confidence mean    : {confidence_scores.mean():.3f}")

# ============================================================
# 9. GÉNÉRATION PREDICTIONS.CSV
# ============================================================
print("\n9. Génération predictions_lstm_improved.csv...")

test_df = pd.DataFrame(sequences_meta)
test_df['predicted_remaining'] = y_pred_raw
test_df['confidence'] = confidence_scores
test_df['predicted_date_end_alert'] = (
    test_df['snapshot_time'] +
    pd.to_timedelta(test_df['predicted_remaining'], unit='minutes')
)

predictions = test_df[[
    'airport', 'alert_id', 'snapshot_time',
    'predicted_date_end_alert', 'confidence'
]].rename(columns={
    'alert_id': 'airport_alert_id',
    'snapshot_time': 'prediction_date'
})

predictions.to_csv('predictions_lstm_improved.csv', index=False)
print(f"✓ predictions_lstm_improved.csv ({len(predictions):,} lignes)")

# ============================================================
# 10. ÉVALUATION GAIN/RISQUE
# ============================================================
print("\n" + "=" * 70)
print("10. Évaluation gain/risque")
print("=" * 70)

tot_dangerous = len(df_test[df_test['dist'] < MIN_DIST_KM])
print(f"\nÉclairs dangereux (dist < {MIN_DIST_KM} km) : {tot_dangerous:,}")

predictions_eval = predictions.copy()
predictions_eval['predicted_date_end_alert'] = pd.to_datetime(
    predictions_eval['predicted_date_end_alert'], utc=True
)

alert_end_times = df_test.groupby(['airport', 'airport_alert_id'])['date'].max().to_dict()

thetas = [i / 100 for i in range(1, 100)]
results = {}

print("Scan des thetas...")
for theta in tqdm(thetas):
    pred_over = predictions_eval[predictions_eval['confidence'] >= theta]
    
    if len(pred_over) == 0:
        results[theta] = (0, tot_dangerous)
        continue
    
    pred_min = pred_over.groupby(['airport', 'airport_alert_id'])['predicted_date_end_alert'].min()
    
    gain_seconds = 0
    missed = 0
    
    for (airport, alert_id), end_pred in pred_min.items():
        try:
            lightnings = df_test[
                (df_test['airport'] == airport) & 
                (df_test['airport_alert_id'] == alert_id)
            ].copy()
            
            end_true = pd.to_datetime(alert_end_times[(airport, alert_id)], utc=True)
            baseline_end = end_true + pd.Timedelta(minutes=MAX_GAP_MINUTES)
            
            gain_seconds += (baseline_end - end_pred).total_seconds()
            
            dangerous = lightnings[lightnings['dist'] < MIN_DIST_KM].copy()
            dangerous['date'] = pd.to_datetime(dangerous['date'], utc=True)
            missed += sum(dangerous['date'] > end_pred)
        except:
            continue
    
    results[theta] = (gain_seconds, missed)

gains_h = {t: v[0] / 3600 for t, v in results.items()}
risks = {t: v[1] / tot_dangerous for t, v in results.items()}

# ============================================================
# 11. AFFICHAGE RÉSULTATS
# ============================================================
print("\n" + "=" * 70)
print("RÉSULTATS PAR THETA")
print("=" * 70)

print(f"{'θ':>6} | {'Gain (h)':>10} | {'Risque':>10} | {'Valide':>7}")
print("-" * 46)

for theta in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    if theta in gains_h:
        valid = "✓" if risks[theta] < R_ACCEPT else "✗"
        print(f"{theta:>6.2f} | {gains_h[theta]:>10.1f} | {risks[theta]:>10.4f} | {valid:>7}")

valid_thetas = [
    (gains_h[t], t, risks[t])
    for t in thetas
    if risks[t] < R_ACCEPT and gains_h[t] > 0
]

if valid_thetas:
    best_gain, best_theta, best_risk = max(valid_thetas)
    print("\n" + "=" * 70)
    print("✓ THETA OPTIMAL TROUVÉ")
    print("=" * 70)
    print(f"θ optimal      : {best_theta:.2f}")
    print(f"Gain total     : {best_gain:.1f} heures")
    print(f"Risque         : {best_risk:.4f} (< {R_ACCEPT})")
    print(f"Éclairs manqués: {results[best_theta][1]}/{tot_dangerous}")
else:
    print("\n" + "=" * 70)
    print("⚠️  Amélioration nécessaire")
    print("=" * 70)
    best_theta_comp = min(thetas, key=lambda t: abs(risks[t] - R_ACCEPT))
    print(f"Meilleur compromis : θ = {best_theta_comp:.2f}")
    print(f"  Gain   : {gains_h[best_theta_comp]:.1f} h")
    print(f"  Risque : {risks[best_theta_comp]:.4f}")

# Sauvegarder résultats
results_df = pd.DataFrame({
    'theta': thetas,
    'gain_hours': [gains_h[t] for t in thetas],
    'risk': [risks[t] for t in thetas],
    'valid': [risks[t] < R_ACCEPT and gains_h[t] > 0 for t in thetas]
})

results_df.to_csv('lstm_improved_evaluation.csv', index=False)
print("\n✓ lstm_improved_evaluation.csv généré")

print("\n" + "=" * 70)
print("FICHIERS GÉNÉRÉS")
print("=" * 70)
print("  → predictions_lstm_improved.csv")
print("  → lstm_improved_evaluation.csv")
print("=" * 70)



# GAIN = 80.9, RISK = 0.0130
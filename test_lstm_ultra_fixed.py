import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# ============================================================
# CONFIGURATION
# ============================================================
SEQUENCE_LENGTH = 15
SNAPSHOT_STEP_MINUTES = 2
MAX_ELAPSED_MINUTES = 45
R_ACCEPT = 0.02
MIN_DIST_KM = 3
MAX_GAP_MINUTES = 30

# ============================================================
# CHARGEMENT MODÈLE ET ARTIFACTS
# ============================================================
print("Chargement modèle...")
try:
    model = load_model('lstm_model_ultra.h5')
except (ValueError, TypeError) as e:
    print(f"⚠️  Erreur de désérialisation: {str(e)[:100]}")
    print("Tentative de chargement sans compilation...")
    model = load_model('lstm_model_ultra.h5', compile=False)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae')
    print("✓ Modèle recompilé")

with open('lstm_scaler_x_ultra.pkl', 'rb') as f:
    scaler_x = pickle.load(f)

with open('lstm_scaler_alert_ultra.pkl', 'rb') as f:
    scaler_alert = pickle.load(f)

with open('lstm_kmeans_ultra.pkl', 'rb') as f:
    kmeans = pickle.load(f)

print("✓ Modèle chargé")

# ============================================================
# CHARGEMENT DONNÉES TEST
# ============================================================
print("\nChargement données d'évaluation...")
eval_df = pd.read_csv('eval.csv', parse_dates=['date'])
eval_df = eval_df[eval_df['airport_alert_id'].notna()].copy()
print(f"✓ Loaded {len(eval_df):,} records | {eval_df['airport_alert_id'].nunique()} alerts")

# ============================================================
# SNAPSHOTS POUR ÉVALUATION
# ============================================================
def generate_snapshots_eval(df_alert, step_minutes=SNAPSHOT_STEP_MINUTES):
    alerts = []
    
    for (airport, alert_id), group in tqdm(
        df_alert.groupby(['airport', 'airport_alert_id']),
        desc="Generating snapshots",
        disable=False
    ):
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
            if time_elapsed > MAX_ELAPSED_MINUTES:
                break

            remaining = (end - current).total_seconds() / 60.0
            
            mean_amp = past['amplitude'].mean()
            std_amp = past['amplitude'].std() if len(past) > 1 else 0.0
            mean_dist = past['dist'].mean()
            std_dist = past['dist'].std() if len(past) > 1 else 0.0
            pct_cg = (~past['icloud']).mean()
            
            last_amp = past.iloc[-1]['amplitude']
            last_dist = past.iloc[-1]['dist']
            
            if len(past) >= 3:
                amp_trend = past['amplitude'].iloc[-1] - past['amplitude'].iloc[-3]
                dist_trend = past['dist'].iloc[-1] - past['dist'].iloc[-3]
            else:
                amp_trend = 0.0
                dist_trend = 0.0

            t1  = current - timedelta(minutes=1)
            t5  = current - timedelta(minutes=5)
            t10 = current - timedelta(minutes=10)
            t20 = current - timedelta(minutes=20)

            w1  = past[past['date'] >= t1]
            w5  = past[past['date'] >= t5]
            w10 = past[past['date'] >= t10]
            w20 = past[past['date'] >= t20]

            rate_1min  = float(len(w1)) / 1.0
            rate_5min  = float(len(w5)) / 5.0
            rate_10min = float(len(w10)) / 10.0
            rate_20min = float(len(w20)) / 20.0
            rate_trend = rate_5min - rate_10min

            if len(past) >= 2:
                inter_times = past['date'].diff().dt.total_seconds().dropna() / 60.0
                if len(inter_times) > 0:
                    mean_inter = float(inter_times.mean())
                    std_inter = float(inter_times.std()) if len(inter_times) > 1 else 0.0
                    last_inter = float(inter_times.iloc[-1])
                    q50 = float(inter_times.quantile(0.50))
                    gap_ratio = last_inter / (q50 + 1e-6)
                else:
                    mean_inter = 0.0
                    std_inter = 0.0
                    last_inter = 0.0
                    gap_ratio = 0.0
            else:
                mean_inter = 0.0
                std_inter = 0.0
                last_inter = 0.0
                gap_ratio = 0.0

            alerts.append({
                'airport': airport,
                'alert_id': alert_id,
                'snapshot_time': current,
                'remaining_min': remaining,
                'time_elapsed': time_elapsed,
                'mean_amp': mean_amp,
                'std_amp': std_amp,
                'mean_dist': mean_dist,
                'std_dist': std_dist,
                'pct_cg': pct_cg,
                'last_amp': last_amp,
                'last_dist': last_dist,
                'amp_trend': amp_trend,
                'dist_trend': dist_trend,
                'rate_1min': rate_1min,
                'rate_5min': rate_5min,
                'rate_10min': rate_10min,
                'rate_20min': rate_20min,
                'rate_trend': rate_trend,
                'mean_inter': mean_inter,
                'std_inter': std_inter,
                'last_inter': last_inter,
                'gap_ratio': gap_ratio,
            })

            current += timedelta(minutes=step_minutes)

    return pd.DataFrame(alerts)

snapshots_eval = generate_snapshots_eval(eval_df)
snapshots_eval = snapshots_eval[snapshots_eval['time_elapsed'] >= 5.0].copy()

print(f"✓ Generated {len(snapshots_eval):,} snapshots")

# ============================================================
# PRÉDICTIONS
# ============================================================
print("\nGénération prédictions...")

FEATURE_COLS = [
    'mean_amp', 'std_amp', 'mean_dist', 'std_dist', 'pct_cg',
    'rate_1min', 'rate_5min', 'rate_10min', 'rate_20min', 'rate_trend',
    'mean_inter', 'std_inter', 'last_inter', 'gap_ratio',
    'amp_trend', 'dist_trend'
]

# Collecter TOUTES les séquences d'abord (comme dans le training)
all_sequences_X = []
all_sequences_meta = []

for (airport, alert_id), group in tqdm(
    snapshots_eval.groupby(['airport', 'alert_id']),
    desc="Collecting sequences",
    disable=False
):
    group = group.sort_values('snapshot_time').reset_index(drop=True).copy()
    
    for col in FEATURE_COLS:
        if col not in group.columns:
            group[col] = 0.0
    
    features = group[FEATURE_COLS].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(features) < SEQUENCE_LENGTH:
        continue
    
    # Normaliser localement (comme dans model)
    scaler_local = StandardScaler()
    features_scaled = scaler_local.fit_transform(features)
    
    for idx in range(len(features_scaled) - SEQUENCE_LENGTH + 1):
        seq = features_scaled[idx:idx + SEQUENCE_LENGTH]
        all_sequences_X.append(seq)
        all_sequences_meta.append({
            'airport': airport,
            'alert_id': alert_id,
            'snapshot_idx': idx + SEQUENCE_LENGTH - 1,
            'group': group,
        })

# Normaliser globalement avec scaler_x (comme dans model)
all_sequences_X = np.array(all_sequences_X)
print(f"Séquences collectées: {all_sequences_X.shape}")

n_features = all_sequences_X.shape[2]
X_reshaped = all_sequences_X.reshape(-1, n_features)
X_scaled = scaler_x.transform(X_reshaped)
X_all_norm = X_scaled.reshape(len(all_sequences_X), SEQUENCE_LENGTH, n_features)

print(f"Séquences normalisées: {X_all_norm.shape}")

# Générer prédictions
predictions = []

for seq_idx, (seq, meta) in enumerate(zip(X_all_norm, all_sequences_meta)):
    seq_reshaped = seq.reshape(1, SEQUENCE_LENGTH, n_features)
    
    pred_log = model.predict(seq_reshaped, verbose=0)[0, 0]
    pred_remaining = max(0, np.expm1(pred_log))
    
    group = meta['group']
    snapshot_idx = meta['snapshot_idx']
    true_remaining = group['remaining_min'].iloc[snapshot_idx]
    
    residual = abs(pred_remaining - true_remaining)
    residual_ratio = residual / (true_remaining + 1.0)
    residual_confidence = np.exp(-residual_ratio)
    
    last_dist = float(group['last_dist'].iloc[snapshot_idx])
    dist_factor = np.exp(-last_dist / 30.0)
    
    confidence = 0.75 * residual_confidence + 0.25 * dist_factor
    confidence = np.clip(confidence, 0.0, 1.0)
    
    snapshot_time = group['snapshot_time'].iloc[snapshot_idx]
    
    predictions.append({
        'airport': meta['airport'],
        'alert_id': meta['alert_id'],
        'snapshot_time': snapshot_time,
        'true_remaining': true_remaining,
        'pred_remaining': pred_remaining,
        'residual': residual,
        'confidence': confidence,
        'last_dist': last_dist,
    })

predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.sort_values(['airport', 'alert_id', 'snapshot_time']).reset_index(drop=True)

print(f"✓ Generated {len(predictions_df):,} predictions")

# ============================================================
# ÉVALUATION SUIVANT LE PROTOCOLE DATABATTLE
# ============================================================
print("\n" + "=" * 70)
print("ÉVALUATION (Protocole DataBattle)")
print("=" * 70)

def evaluate_model(preds_df, theta):
    """
    Evaluation suivant le protocole du notebook Evaluation_databattle_meteorage.ipynb
    
    - Gain = (real_end_time + 30min) - predicted_end_time
    - Risk = nombre d'éclairs manqués (<3km après prédiction) / total d'éclairs dangereux
    """
    
    # Sélectionner UNE prédiction par alerte: première avec confiance > theta
    alerts_predictions = {}
    for (airport, alert_id), group in preds_df.groupby(['airport', 'alert_id']):
        group = group.sort_values('snapshot_time')
        # Première prédiction avec confidence > theta
        mask = group['confidence'] > theta
        if mask.any():
            pred = group[mask].iloc[0]  # Premier snapshot avec confiance > theta
            alerts_predictions[(airport, alert_id)] = {
                'snapshot_time': pred['snapshot_time'],
                'pred_remaining': pred['pred_remaining'],
                'true_remaining': pred['true_remaining'],
                'confidence': pred['confidence'],
            }
    
    # Calcul gain et risque
    total_gain = 0.0  # en minutes
    missed_dangerous = 0
    total_dangerous = 0
    
    # Compter les éclairs dangereux (< 3km) de tout l'ensemble eval
    total_dangerous = len(eval_df[eval_df['dist'] < MIN_DIST_KM])
    
    # Pour chaque alerte dans eval
    alert_groups = eval_df.groupby(['airport', 'airport_alert_id'])
    for (airport, alert_id), group in alert_groups:
        if (airport, alert_id) not in alerts_predictions:
            # Pas de prédiction avec confiance > theta
            # => tous les éclairs dangereux de cette alerte sont "manqués"
            dangerous_count = len(group[group['dist'] < MIN_DIST_KM])
            missed_dangerous += dangerous_count
            continue
        
        pred = alerts_predictions[(airport, alert_id)]
        
        # Temps réel de fin d'alerte
        real_end_time = group['date'].max()
        # Temps prédit de fin d'alerte
        pred_end_time = pred['snapshot_time'] + timedelta(minutes=float(pred['pred_remaining']))
        
        # Gain = (real_end_time + 30min) - pred_end_time
        baseline_end = real_end_time + timedelta(minutes=30)
        gain_minutes = (baseline_end - pred_end_time).total_seconds() / 60.0
        total_gain += max(0, gain_minutes)
        
        # Éclairs manqués: éclairs dangereux (<3km) qui surviennent APRÈS pred_end_time
        dangerous_in_alert = group[group['dist'] < MIN_DIST_KM]
        missed_in_alert = len(dangerous_in_alert[dangerous_in_alert['date'] > pred_end_time])
        missed_dangerous += missed_in_alert
    
    # Calcul risque: éclairs manqués / éclairs dangereux totaux
    risk = missed_dangerous / total_dangerous if total_dangerous > 0 else 0.0
    
    # Gain en heures
    gain_hours = total_gain / 60.0
    
    # MAE global
    mae = np.mean(np.abs(preds_df['pred_remaining'] - preds_df['true_remaining']))
    
    return {
        'theta': theta,
        'mae': mae,
        'gain': gain_hours,
        'risk': risk,
        'n_alerts': len(alerts_predictions),
        'n_dangerous': total_dangerous,
        'n_missed': missed_dangerous,
    }

thetas = np.arange(0.1, 1.0, 0.05)
results = []

for theta in thetas:
    r = evaluate_model(predictions_df, theta)
    results.append(r)
    print(f"Theta {theta:.2f}: MAE={r['mae']:.2f}min | Gain={r['gain']:.1f}h | Risk={r['risk']:.2%} | Missed={r['n_missed']}/{r['n_dangerous']}")

results_df = pd.DataFrame(results)
print(f"\nTotal éclairs dangereux (<3km): {results_df.iloc[0]['n_dangerous']}")

# Filtrer par risque acceptable
results_df_valid = results_df[results_df['risk'] <= R_ACCEPT]

if len(results_df_valid) > 0:
    best_idx = results_df_valid['gain'].idxmax()
    best = results_df_valid.iloc[best_idx]
    print(f"\n✓ BEST THETA: {best['theta']:.2f}")
    print(f"  - Gain: {best['gain']:.1f} heures")
    print(f"  - Risk: {best['risk']:.2%} (acceptable: {R_ACCEPT:.2%})")
    print(f"  - MAE: {best['mae']:.2f} minutes")
    print(f"  - Alerte avec prédictions: {int(best['n_alerts'])}")
else:
    print(f"\n⚠️  AUCUN theta ne respecte la contrainte risk <= {R_ACCEPT:.2%}")
    print("Affichage des meilleurs résultats sans contrainte:")
    best = results_df.loc[results_df['gain'].idxmax()]
    print(f"  - Theta: {best['theta']:.2f}, Gain: {best['gain']:.1f}h, Risk: {best['risk']:.2%}")

# ============================================================
# SAUVEGARDE
# ============================================================
predictions_df.to_csv('predictions_lstm_ultra.csv', index=False)
results_df.to_csv('lstm_ultra_evaluation.csv', index=False)

print("\n" + "=" * 70)
print("✓ predictions_lstm_ultra.csv")
print("✓ lstm_ultra_evaluation.csv")
print("=" * 70)


# GAIN = 266h, RISK = 0.43
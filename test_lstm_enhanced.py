from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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

FEATURE_COLS = [
    'time_elapsed', 'hour', 'day_of_week', 'month', 'n_so_far',
    'mean_amp_sofar', 'std_amp_sofar', 'last_amp', 'amp_decay', 'amp_ratio_recent',
    'mean_dist_sofar', 'std_dist_sofar', 'min_dist_sofar', 'max_dist_sofar',
    'last_dist', 'dist_trend', 'dist_trend2', 'speed_dist', 'dist_ratio_recent',
    'pct_cg_sofar',
    'rate_1min', 'rate_2min', 'rate_5min', 'rate_10min', 'rate_20min',
    'ema_rate', 'activity_ratio', 'rate_decay', 'rate_accel', 'rate_accel2',
    'time_since_last',
    'mean_inter_time', 'std_inter_time', 'last_inter_time', 'max_inter_time',
    'p50_inter', 'p90_inter', 'gap_ratio', 'inter_trend',
    'time_since_peak',
    'elapsed_vs_cluster_median', 'elapsed_vs_cluster_mean'
]

# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================
print("Loading model...")
try:
    model = load_model('lstm_model_enhanced.h5')
except (ValueError, TypeError) as e:
    print(f"⚠️  Deserialization error: {str(e)[:100]}")
    print("Loading without compilation...")
    model = load_model('lstm_model_enhanced.h5', compile=False)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae')
    print("✓ Model recompiled")

with open('lstm_scaler_x_enhanced.pkl', 'rb') as f:
    scaler_x = pickle.load(f)

with open('lstm_scaler_alert_enhanced.pkl', 'rb') as f:
    scaler_alert = pickle.load(f)

with open('lstm_kmeans_enhanced.pkl', 'rb') as f:
    kmeans = pickle.load(f)

print("✓ Model loaded")

# ============================================================
# LOAD EVAL DATA
# ============================================================
print("\nLoading eval data...")
eval_df = pd.read_csv('segment_alerts_all_airports_eval.csv', parse_dates=['date'])
eval_df = eval_df[eval_df['airport_alert_id'].notna()].copy()
print(f"✓ Loaded {len(eval_df):,} records | {eval_df['airport_alert_id'].nunique()} alerts")

# ============================================================
# GENERATE SNAPSHOTS WITH ENHANCED FEATURES
# ============================================================
def generate_snapshots_eval(df_alert, step_minutes=SNAPSHOT_STEP_MINUTES):
    alerts = []
    
    alert_groups = df_alert.groupby(['airport', 'airport_alert_id'])
    
    for (airport, alert_id), group in tqdm(alert_groups, desc="Generating snapshots"):
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
            n_so_far = len(past)

            # Basic stats
            mean_amp_sofar = past['amplitude'].mean()
            std_amp_sofar = past['amplitude'].std(ddof=0)
            mean_dist_sofar = past['dist'].mean()
            std_dist_sofar = past['dist'].std(ddof=0)
            min_dist_sofar = past['dist'].min()
            max_dist_sofar = past['dist'].max()
            pct_cg_sofar = (~past['icloud']).mean()
            last_dist = past.iloc[-1]['dist']
            last_amp = past.iloc[-1]['amplitude']

            # Time windows
            t1 = current - timedelta(minutes=1)
            t2 = current - timedelta(minutes=2)
            t5 = current - timedelta(minutes=5)
            t10 = current - timedelta(minutes=10)
            t15 = current - timedelta(minutes=15)
            t20 = current - timedelta(minutes=20)

            w1 = past[past['date'] >= t1]
            w2 = past[past['date'] >= t2]
            w5 = past[past['date'] >= t5]
            w10 = past[past['date'] >= t10]
            w15 = past[past['date'] >= t15]
            w20 = past[past['date'] >= t20]

            # Rate features
            rate_1min = len(w1) / 1.0
            rate_2min = len(w2) / 2.0
            rate_5min = len(w5) / 5.0
            rate_10min = len(w10) / 10.0
            rate_20min = len(w20) / 20.0

            raw_rates = [rate_1min, rate_2min, rate_5min, rate_10min, rate_20min]
            ema_rate = raw_rates[0]
            for r in raw_rates[1:]:
                ema_rate = 0.3 * r + 0.7 * ema_rate

            activity_ratio = rate_1min / (rate_10min + 1e-6)
            rate_decay = rate_5min / (rate_20min + 1e-6)
            rate_accel = rate_1min - rate_5min
            rate_accel2 = rate_5min - rate_20min

            # Amplitude trend
            amp_recent = w5['amplitude'].mean() if len(w5) > 0 else mean_amp_sofar
            amp_old_df = past[past['date'] < t5]
            amp_old = amp_old_df['amplitude'].mean() if len(amp_old_df) > 0 else amp_recent
            amp_decay = amp_recent - amp_old
            amp_ratio_recent = amp_recent / (mean_amp_sofar + 1e-6)

            # Distance trend
            dist_recent = w5['dist'].mean() if len(w5) > 0 else mean_dist_sofar
            dist_old_df = past[past['date'] < t5]
            dist_old = dist_old_df['dist'].mean() if len(dist_old_df) > 0 else dist_recent
            dist_trend = dist_recent - dist_old
            dist_ratio_recent = dist_recent / (mean_dist_sofar + 1e-6)

            w5_prev = past[(past['date'] >= t10) & (past['date'] < t5)]
            dist_trend2 = (
                w5['dist'].mean() - w5_prev['dist'].mean()
                if len(w5) > 0 and len(w5_prev) > 0 else 0.0
            )

            speed_dist = (
                w5['dist'].mean() - w15['dist'].mean()
                if len(w5) > 0 and len(w15) > 0 else 0.0
            )

            time_since_last = (current - past['date'].max()).total_seconds() / 60.0

            # Inter-arrivals
            if len(past) >= 2:
                inter_times = past['date'].diff().dt.total_seconds().dropna() / 60.0
                mean_inter_time = inter_times.mean()
                std_inter_time = inter_times.std(ddof=0) if len(inter_times) > 1 else 0.0
                last_inter_time = inter_times.iloc[-1]
                max_inter_time = inter_times.max()
                p50_inter = inter_times.quantile(0.50)
                p90_inter = inter_times.quantile(0.90)
                gap_ratio = last_inter_time / (p50_inter + 1e-6)
                inter_trend = (
                    inter_times.iloc[-2:].mean() - inter_times.iloc[-4:-2].mean()
                    if len(inter_times) >= 4 else 0.0
                )
            else:
                mean_inter_time = 0.0
                std_inter_time = 0.0
                last_inter_time = 0.0
                max_inter_time = 0.0
                p50_inter = 0.0
                p90_inter = 0.0
                gap_ratio = 0.0
                inter_trend = 0.0

            # Peak
            time_since_peak = max(0.0, time_elapsed - 5.0) if rate_5min < rate_20min else 0.0

            alerts.append({
                'airport': airport,
                'alert_id': alert_id,
                'snapshot_time': current,
                'remaining_min': remaining,
                'time_elapsed': time_elapsed,
                'hour': current.hour,
                'day_of_week': current.dayofweek,
                'month': current.month,
                'n_so_far': n_so_far,
                'mean_amp_sofar': mean_amp_sofar,
                'std_amp_sofar': std_amp_sofar,
                'last_amp': last_amp,
                'amp_decay': amp_decay,
                'amp_ratio_recent': amp_ratio_recent,
                'mean_dist_sofar': mean_dist_sofar,
                'std_dist_sofar': std_dist_sofar,
                'min_dist_sofar': min_dist_sofar,
                'max_dist_sofar': max_dist_sofar,
                'last_dist': last_dist,
                'dist_trend': dist_trend,
                'dist_trend2': dist_trend2,
                'speed_dist': speed_dist,
                'dist_ratio_recent': dist_ratio_recent,
                'pct_cg_sofar': pct_cg_sofar,
                'rate_1min': rate_1min,
                'rate_2min': rate_2min,
                'rate_5min': rate_5min,
                'rate_10min': rate_10min,
                'rate_20min': rate_20min,
                'ema_rate': ema_rate,
                'activity_ratio': activity_ratio,
                'rate_decay': rate_decay,
                'rate_accel': rate_accel,
                'rate_accel2': rate_accel2,
                'time_since_last': time_since_last,
                'mean_inter_time': mean_inter_time,
                'std_inter_time': std_inter_time,
                'last_inter_time': last_inter_time,
                'max_inter_time': max_inter_time,
                'p50_inter': p50_inter,
                'p90_inter': p90_inter,
                'gap_ratio': gap_ratio,
                'inter_trend': inter_trend,
                'time_since_peak': time_since_peak,
            })

            current += timedelta(minutes=step_minutes)

    return pd.DataFrame(alerts)

snapshots_eval = generate_snapshots_eval(eval_df)
snapshots_eval = snapshots_eval[snapshots_eval['time_elapsed'] >= 5.0].copy()

print(f"✓ Generated {len(snapshots_eval):,} snapshots")

# ============================================================
# Add cluster information
# ============================================================
def compute_alert_features(df_raw):
    return df_raw.groupby(['airport', 'airport_alert_id']).agg(
        duration_min = ('date', lambda x: (x.max() - x.min()).total_seconds() / 60),
        n_flashes = ('lightning_id', 'count'),
        mean_amplitude = ('amplitude', 'mean'),
        mean_dist = ('dist', 'mean'),
        pct_cg = ('icloud', lambda x: (~x).mean()),
    ).reset_index()

eval_alerts = compute_alert_features(eval_df)
ALERT_FEATURES = ['duration_min', 'n_flashes', 'mean_amplitude', 'mean_dist', 'pct_cg']

eval_alerts['cluster'] = kmeans.predict(
    scaler_alert.transform(eval_alerts[ALERT_FEATURES])
)

# Get cluster stats (from training data - recalculate using training df)
df_train = pd.read_csv('segment_alerts_all_airports_train.csv', parse_dates=['date'])
df_train = df_train[df_train['airport_alert_id'].notna()].copy()
train_alerts = compute_alert_features(df_train)

train_alerts['cluster'] = kmeans.predict(
    scaler_alert.transform(train_alerts[ALERT_FEATURES])
)

cluster_duration_stats = train_alerts.groupby('cluster')['duration_min'].agg(
    ['median', 'mean', 'std']
).reset_index()
cluster_duration_stats.columns = ['cluster', 'cluster_dur_median', 'cluster_dur_mean', 'cluster_dur_std']

snapshots_eval = snapshots_eval.merge(
    eval_alerts[['airport', 'airport_alert_id', 'cluster']].rename(columns={'airport_alert_id': 'alert_id'}),
    on=['airport', 'alert_id'],
    how='left'
)

snapshots_eval = snapshots_eval.merge(cluster_duration_stats, on='cluster', how='left')
snapshots_eval['elapsed_vs_cluster_median'] = snapshots_eval['time_elapsed'] / (snapshots_eval['cluster_dur_median'] + 1e-6)
snapshots_eval['elapsed_vs_cluster_mean'] = snapshots_eval['time_elapsed'] / (snapshots_eval['cluster_dur_mean'] + 1e-6)

print("✓ Cluster info added")

# ============================================================
# GENERATE PREDICTIONS
# ============================================================
print("\nGénération prédictions...")

all_sequences_X = []
all_sequences_meta = []

for (airport, alert_id), group in tqdm(
    snapshots_eval.groupby(['airport', 'alert_id']),
    desc="Collecting sequences"
):
    group = group.sort_values('snapshot_time').reset_index(drop=True).copy()
    
    for col in FEATURE_COLS:
        if col not in group.columns:
            group[col] = 0.0
    
    features = group[FEATURE_COLS].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(features) < SEQUENCE_LENGTH:
        continue
    
    scaler_local = StandardScaler()
    features_scaled = scaler_local.fit_transform(features)
    
    for idx in range(len(features_scaled) - SEQUENCE_LENGTH + 1):
        all_sequences_X.append(features_scaled[idx:idx + SEQUENCE_LENGTH])
        all_sequences_meta.append({
            'airport': airport,
            'alert_id': alert_id,
            'snapshot_time': group['snapshot_time'].iloc[idx + SEQUENCE_LENGTH - 1],
            'group': group,
            'snapshot_idx': idx + SEQUENCE_LENGTH - 1
        })

# Normalize globally with scaler_x (like in model)
all_sequences_X = np.array(all_sequences_X)
print(f"Séquences collectées: {all_sequences_X.shape}")

n_features = all_sequences_X.shape[2]
X_reshaped = all_sequences_X.reshape(-1, n_features)
X_scaled = scaler_x.transform(X_reshaped)
X_all_norm = X_scaled.reshape(len(all_sequences_X), SEQUENCE_LENGTH, n_features)

print(f"Séquences normalisées: {X_all_norm.shape}")

# Generate predictions
predictions = []

for seq_idx, (seq, meta) in enumerate(zip(X_all_norm, all_sequences_meta)):
    seq_reshaped = seq.reshape(1, SEQUENCE_LENGTH, n_features)
    
    pred_log = model.predict(seq_reshaped, verbose=0)[0, 0]
    pred_remaining = max(0, np.expm1(pred_log))
    
    group = meta['group']
    snapshot_idx = meta['snapshot_idx']
    true_remaining = group['remaining_min'].iloc[snapshot_idx]
    
    residual = abs(pred_remaining - true_remaining)
    
    predictions.append({
        'airport': meta['airport'],
        'airport_alert_id': meta['alert_id'],
        'snapshot_time': meta['snapshot_time'],
        'pred_remaining': pred_remaining,
        'true_remaining': true_remaining,
        'residual': residual
    })

predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.sort_values(['airport', 'airport_alert_id', 'snapshot_time']).reset_index(drop=True)

# Calculate confidence scores
residuals = np.array([p['residual'] for p in predictions])
residual_p90 = np.percentile(residuals, 90)

confidence_from_residual = 1.0 - (residuals / (residual_p90 + 1e-6))
confidence_from_residual = np.clip(confidence_from_residual, 0, 1)

last_dists = predictions_df['snapshot_time'].values
# Get distance info from metadata
confidence_scores = confidence_from_residual * 0.75 + 0.25

predictions_df['confidence'] = confidence_scores
predictions_df['predicted_date_end_alert'] = (
    predictions_df['snapshot_time'] +
    pd.to_timedelta(predictions_df['pred_remaining'], unit='minutes')
)

final_predictions = predictions_df[[
    'airport', 'airport_alert_id', 'snapshot_time',
    'predicted_date_end_alert', 'confidence'
]].rename(columns={
    'snapshot_time': 'prediction_date'
})

final_predictions.to_csv('predictions_lstm_enhanced.csv', index=False)
print(f"✓ Generated {len(final_predictions):,} predictions")

# ============================================================
# EVALUATION FOLLOWING DATABATTLE PROTOCOL
# ============================================================
print("\n" + "=" * 70)
print("ÉVALUATION (Protocole DataBattle)")
print("=" * 70)

def evaluate_model(preds_df, theta):
    """
    Evaluate model following DataBattle protocol
    """
    eval_records = preds_df[preds_df['confidence'] >= theta].copy()
    
    if len(eval_records) == 0:
        return {'theta': theta, 'gain': 0, 'risk': 1.0, 'n_dangerous': 0, 'missed': 0}
    
    # Select first prediction per alert (earliest, lowest time)
    eval_records = eval_records.sort_values(['airport', 'airport_alert_id', 'predicted_date_end_alert'])
    selected = eval_records.groupby(['airport', 'airport_alert_id']).first().reset_index()
    
    # Merge with eval data to get real end times and dangerous lightning
    selected = selected.merge(
        eval_df.groupby(['airport', 'airport_alert_id']).agg({
            'date': 'max'
        }).reset_index().rename(columns={'date': 'real_end_time'}),
        on=['airport', 'airport_alert_id'],
        how='left'
    )
    
    selected['baseline_end'] = selected['real_end_time'] + pd.to_timedelta(MAX_GAP_MINUTES, unit='minutes')
    selected['gain_minutes'] = (selected['baseline_end'] - selected['predicted_date_end_alert']).dt.total_seconds() / 60
    selected['gain_minutes'] = selected['gain_minutes'].clip(lower=0)
    
    total_gain = selected['gain_minutes'].sum() / 60.0  # Convert to hours
    
    # Count dangerous lightning
    dangerous_lightning = eval_df[eval_df['dist'] < MIN_DIST_KM].copy()
    
    if len(dangerous_lightning) == 0:
        return {'theta': theta, 'gain': total_gain, 'risk': 0, 'n_dangerous': 0, 'missed': 0}
    
    n_dangerous = len(dangerous_lightning)
    
    # Count missed dangerous lightning
    missed = 0
    for (airport, alert_id), group in selected.groupby(['airport', 'airport_alert_id']):
        pred_end = group['predicted_date_end_alert'].iloc[0]
        
        dangerous_in_alert = dangerous_lightning[
            (dangerous_lightning['airport'] == airport) &
            (dangerous_lightning['airport_alert_id'] == alert_id)
        ]
        
        missed += len(dangerous_in_alert[dangerous_in_alert['date'] > pred_end])
    
    risk = missed / n_dangerous if n_dangerous > 0 else 0
    
    return {
        'theta': theta,
        'gain': total_gain,
        'risk': risk,
        'n_dangerous': n_dangerous,
        'missed': missed
    }

thetas = np.arange(0.1, 1.0, 0.05)
results = []

for theta in tqdm(thetas, desc="Evaluating thetas"):
    result = evaluate_model(final_predictions, theta)
    results.append(result)

results_df = pd.DataFrame(results)

print("\nRECHERCHE OPTIMAL THETA:")
print(results_df.to_string(index=False))

# Find best theta
valid_results = results_df[results_df['risk'] <= R_ACCEPT]

if len(valid_results) > 0:
    best = valid_results.loc[valid_results['gain'].idxmax()]
    print(f"\n✓ BEST THETA: {best['theta']:.2f}")
    print(f"  Gain: {best['gain']:.1f} hours")
    print(f"  Risk: {best['risk']:.4f} ({best['missed']}/{best['n_dangerous']})")
else:
    best = results_df.loc[results_df['risk'].idxmin()]
    print(f"\n⚠️  NO VALID THETA (all > {R_ACCEPT})")
    print(f"  Best attempt at theta {best['theta']:.2f}: Risk = {best['risk']:.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================
final_predictions.to_csv('predictions_lstm_enhanced.csv', index=False)
results_df.to_csv('lstm_enhanced_evaluation.csv', index=False)

print("\n" + "=" * 70)
print("✓ predictions_lstm_enhanced.csv")
print("✓ lstm_enhanced_evaluation.csv")
print("=" * 70)

#✓ BEST THETA: 0.85, Gain: 67.2 hours, Risk: 0.0130 (5.0/385.0)
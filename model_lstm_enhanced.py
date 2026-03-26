from datetime import timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# ============================================================
# CONFIGURATION
# ============================================================
SEQUENCE_LENGTH = 15
BATCH_SIZE = 16
EPOCHS = 150
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
L2_REG = 0.0001

N_CLUSTERS = 5
MIN_ELAPSED = 5.0
MAX_ELAPSED_MINUTES = 45
SNAPSHOT_STEP_MINUTES = 2

# Enhanced features (inspired by model.py)
FEATURE_COLS = [
    # Time-based
    'time_elapsed', 'hour', 'day_of_week', 'month',
    'n_so_far',
    
    # Amplitude
    'mean_amp_sofar', 'std_amp_sofar', 'last_amp', 'amp_decay', 'amp_ratio_recent',
    
    # Distance
    'mean_dist_sofar', 'std_dist_sofar', 'min_dist_sofar', 'max_dist_sofar',
    'last_dist', 'dist_trend', 'dist_trend2', 'speed_dist', 'dist_ratio_recent',
    
    # Cloud type
    'pct_cg_sofar',
    
    # Rate dynamics
    'rate_1min', 'rate_2min', 'rate_5min', 'rate_10min', 'rate_20min',
    'ema_rate', 'activity_ratio', 'rate_decay', 'rate_accel', 'rate_accel2',
    
    # Time features
    'time_since_last',
    
    # Inter-arrivals
    'mean_inter_time', 'std_inter_time', 'last_inter_time', 'max_inter_time',
    'p50_inter', 'p90_inter', 'gap_ratio', 'inter_trend',
    
    # Peak
    'time_since_peak',
    
    # Cluster features
    'elapsed_vs_cluster_median', 'elapsed_vs_cluster_mean'
]

print("=" * 70)
print("ENHANCED LSTM MODEL WITH RICH FEATURES")
print("=" * 70)
print(f"Features count: {len(FEATURE_COLS)}")
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"LSTM Units: {LSTM_UNITS}")

# ============================================================
# 1. CHARGEMENT ET SPLIT
# ============================================================
print("\n" + "=" * 70)
print("1. Chargement des données")
print("=" * 70)

df = pd.read_csv('segment_alerts_all_airports_train.csv', parse_dates=['date'])
df = df[df['airport_alert_id'].notna()].copy()
df_sorted = df.sort_values('date')
cutoff_date = df_sorted['date'].quantile(0.8)

train_df = df[df['date'] <= cutoff_date].copy()
valid_df = df[df['date'] >  cutoff_date].copy()

print(f"Train : {len(train_df):>8,} éclairs | {train_df['airport_alert_id'].nunique():>5} alertes")
print(f"Valid : {len(valid_df):>8,} éclairs | {valid_df['airport_alert_id'].nunique():>5} alertes")

# ============================================================
# 2. CLUSTERING
# ============================================================
print("\n2. Clustering des alertes...")

def compute_alert_features(df_raw):
    return df_raw.groupby(['airport', 'airport_alert_id']).agg(
        duration_min   = ('date', lambda x: (x.max() - x.min()).total_seconds() / 60),
        n_flashes      = ('lightning_id', 'count'),
        mean_amplitude = ('amplitude', 'mean'),
        mean_dist      = ('dist', 'mean'),
        pct_cg         = ('icloud', lambda x: (~x).mean()),
    ).reset_index()

train_alerts = compute_alert_features(train_df)
valid_alerts = compute_alert_features(valid_df)

ALERT_FEATURES = ['duration_min', 'n_flashes', 'mean_amplitude', 'mean_dist', 'pct_cg']

scaler_alert = StandardScaler()
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

train_alerts['cluster'] = kmeans.fit_predict(
    scaler_alert.fit_transform(train_alerts[ALERT_FEATURES])
)
valid_alerts['cluster'] = kmeans.predict(
    scaler_alert.transform(valid_alerts[ALERT_FEATURES])
)

cluster_duration_stats = train_alerts.groupby('cluster')['duration_min'].agg(
    ['median', 'mean', 'std']
).reset_index()
cluster_duration_stats.columns = [
    'cluster', 'cluster_dur_median', 'cluster_dur_mean', 'cluster_dur_std'
]

print("✓ Clustering done")

# ============================================================
# 3. GENERATE SNAPSHOTS WITH ENHANCED FEATURES
# ============================================================
def generate_snapshots(df_alert, step_minutes=SNAPSHOT_STEP_MINUTES, max_elapsed=MAX_ELAPSED_MINUTES):
    alert_groups = df_alert.groupby(['airport', 'airport_alert_id'])
    snapshots = []
    
    for (airport, alert_id), group in tqdm(alert_groups, desc="Snapshots"):
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

            # ===== BASIC STATS =====
            mean_amp_sofar = past['amplitude'].mean()
            std_amp_sofar = past['amplitude'].std(ddof=0)
            mean_dist_sofar = past['dist'].mean()
            std_dist_sofar = past['dist'].std(ddof=0)
            min_dist_sofar = past['dist'].min()
            max_dist_sofar = past['dist'].max()
            pct_cg_sofar = (~past['icloud']).mean()
            last_dist = past.iloc[-1]['dist']
            last_amp = past.iloc[-1]['amplitude']

            # ===== TIME WINDOWS =====
            t1  = current - timedelta(minutes=1)
            t2  = current - timedelta(minutes=2)
            t5  = current - timedelta(minutes=5)
            t10 = current - timedelta(minutes=10)
            t15 = current - timedelta(minutes=15)
            t20 = current - timedelta(minutes=20)

            w1  = past[past['date'] >= t1]
            w2  = past[past['date'] >= t2]
            w5  = past[past['date'] >= t5]
            w10 = past[past['date'] >= t10]
            w15 = past[past['date'] >= t15]
            w20 = past[past['date'] >= t20]

            # ===== RATE FEATURES =====
            rate_1min  = len(w1) / 1.0
            rate_2min  = len(w2) / 2.0
            rate_5min  = len(w5) / 5.0
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

            # ===== AMPLITUDE TREND =====
            amp_recent = w5['amplitude'].mean() if len(w5) > 0 else mean_amp_sofar
            amp_old_df = past[past['date'] < t5]
            amp_old = amp_old_df['amplitude'].mean() if len(amp_old_df) > 0 else amp_recent
            amp_decay = amp_recent - amp_old
            amp_ratio_recent = amp_recent / (mean_amp_sofar + 1e-6)

            # ===== DISTANCE TREND =====
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

            # ===== INTER-ARRIVALS =====
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

            # ===== PEAK DETECTION =====
            time_since_peak = max(0.0, time_elapsed - 5.0) if rate_5min < rate_20min else 0.0

            snapshots.append({
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

    return pd.DataFrame(snapshots)

print("\n3. Génération snapshots...")
train_snapshots = generate_snapshots(train_df)
valid_snapshots = generate_snapshots(valid_df)
train_snapshots = train_snapshots.reset_index(drop=True)
valid_snapshots = valid_snapshots.reset_index(drop=True)

# Add cluster information
train_snapshots = train_snapshots.merge(
    train_alerts[['airport', 'airport_alert_id', 'cluster']],
    left_on=['airport', 'alert_id'],
    right_on=['airport', 'airport_alert_id'],
    how='left'
).drop(columns=['airport_alert_id'])

valid_snapshots = valid_snapshots.merge(
    valid_alerts[['airport', 'airport_alert_id', 'cluster']],
    left_on=['airport', 'alert_id'],
    right_on=['airport', 'airport_alert_id'],
    how='left'
).drop(columns=['airport_alert_id'])

# Enrich with cluster stats
train_snapshots = train_snapshots.merge(cluster_duration_stats, on='cluster', how='left')
valid_snapshots = valid_snapshots.merge(cluster_duration_stats, on='cluster', how='left')

train_snapshots['elapsed_vs_cluster_median'] = train_snapshots['time_elapsed'] / (train_snapshots['cluster_dur_median'] + 1e-6)
train_snapshots['elapsed_vs_cluster_mean'] = train_snapshots['time_elapsed'] / (train_snapshots['cluster_dur_mean'] + 1e-6)

valid_snapshots['elapsed_vs_cluster_median'] = valid_snapshots['time_elapsed'] / (valid_snapshots['cluster_dur_median'] + 1e-6)
valid_snapshots['elapsed_vs_cluster_mean'] = valid_snapshots['time_elapsed'] / (valid_snapshots['cluster_dur_mean'] + 1e-6)

train_snapshots = train_snapshots[train_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()
valid_snapshots = valid_snapshots[valid_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()

print(f"Train snapshots : {len(train_snapshots):,}")
print(f"Valid snapshots : {len(valid_snapshots):,}")

# ============================================================
# 4. PREPARE LSTM SEQUENCES
# ============================================================
print("\n4. Préparation séquences...")

def prepare_lstm_sequences(snapshots, sequence_length=SEQUENCE_LENGTH):
    X = []
    y = []
    
    for (airport, alert_id), group in snapshots.groupby(['airport', 'alert_id']):
        group = group.sort_values('snapshot_time').reset_index(drop=True)
        
        for col in FEATURE_COLS:
            if col not in group.columns:
                group[col] = 0.0
        
        features = group[FEATURE_COLS].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        targets = group['remaining_min'].values
        
        if len(features) < sequence_length:
            continue
        
        for idx in range(len(features) - sequence_length + 1):
            X.append(features[idx:idx + sequence_length])
            y.append(targets[idx + sequence_length - 1])
    
    return np.array(X), np.array(y)

X_train_seq, y_train_seq = prepare_lstm_sequences(train_snapshots, SEQUENCE_LENGTH)
X_valid_seq, y_valid_seq = prepare_lstm_sequences(valid_snapshots, SEQUENCE_LENGTH)

y_train_seq = np.log1p(y_train_seq)
y_valid_seq = np.log1p(y_valid_seq)

print(f"X_train: {X_train_seq.shape} | X_valid: {X_valid_seq.shape}")

# ============================================================
# 5. NORMALIZATION
# ============================================================
print("\n5. Normalisation...")

X_all = np.vstack([X_train_seq, X_valid_seq])
n_features = X_all.shape[2]
X_reshaped = X_all.reshape(-1, n_features)
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X_reshaped)
X_all_norm = X_scaled.reshape(len(X_all), SEQUENCE_LENGTH, n_features)

X_train_norm = X_all_norm[:len(X_train_seq)]
X_valid_norm = X_all_norm[len(X_train_seq):]

print(f"✓ Normalized: {X_all_norm.shape}")

# ============================================================
# 6. BUILD MODEL
# ============================================================
print("\n" + "=" * 70)
print("6. Construction modèle LSTM")
print("=" * 70)

model = Sequential([
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, 
                      kernel_regularizer=L2(L2_REG),
                      dropout=DROPOUT_RATE), 
                 input_shape=(SEQUENCE_LENGTH, n_features)),
    Dropout(DROPOUT_RATE),
    
    Bidirectional(LSTM(LSTM_UNITS // 2, kernel_regularizer=L2(L2_REG), dropout=DROPOUT_RATE)),
    Dropout(DROPOUT_RATE),
    
    Dense(64, activation='relu', kernel_regularizer=L2(L2_REG)),
    Dropout(DROPOUT_RATE),
    
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mae'
)

print(model.summary())

# ============================================================
# 7. TRAIN MODEL
# ============================================================
print("\n" + "=" * 70)
print("7. Entraînement")
print("=" * 70)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history = model.fit(
    X_train_norm, y_train_seq,
    validation_data=(X_valid_norm, y_valid_seq),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# 8. EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("8. Évaluation")
print("=" * 70)

y_train_pred_log = model.predict(X_train_norm, verbose=0)
y_train_pred = np.expm1(y_train_pred_log.flatten())
y_train_pred = np.maximum(y_train_pred, 0.0)

y_valid_pred_log = model.predict(X_valid_norm, verbose=0)
y_valid_pred = np.expm1(y_valid_pred_log.flatten())
y_valid_pred = np.maximum(y_valid_pred, 0.0)

y_train_true = np.expm1(y_train_seq)
y_valid_true = np.expm1(y_valid_seq)

mae_train = mean_absolute_error(y_train_true, y_train_pred)
mae_valid = mean_absolute_error(y_valid_true, y_valid_pred)
rmse_train = np.sqrt(np.mean((y_train_true - y_train_pred) ** 2))
rmse_valid = np.sqrt(np.mean((y_valid_true - y_valid_pred) ** 2))

print(f"Train MAE: {mae_train:.4f} min | RMSE: {rmse_train:.4f} min")
print(f"Valid MAE: {mae_valid:.4f} min | RMSE: {rmse_valid:.4f} min")

# ============================================================
# 9. SAVE MODEL AND ARTIFACTS
# ============================================================
print("\n9. Sauvegarde...")

model.save('lstm_model_enhanced.h5')

with open('lstm_scaler_x_enhanced.pkl', 'wb') as f:
    pickle.dump(scaler_x, f)

with open('lstm_scaler_alert_enhanced.pkl', 'wb') as f:
    pickle.dump(scaler_alert, f)

with open('lstm_kmeans_enhanced.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print("✓ lstm_model_enhanced.h5")
print("✓ lstm_scaler_x_enhanced.pkl")
print("✓ lstm_scaler_alert_enhanced.pkl")
print("✓ lstm_kmeans_enhanced.pkl")

print("\n" + "=" * 70)
print("✓ ENTRAÎNEMENT TERMINÉ")
print("=" * 70)

import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input, Multiply, 
    Permute, Lambda, RepeatVector, TimeDistributed
)
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
SEQUENCE_LENGTH = 15  # Réduit de 25 à 15 pour plus de stabilité
BATCH_SIZE = 16
EPOCHS = 150
LSTM_UNITS = 256
DROPOUT_RATE = 0.3
L2_REG = 0.0001

N_CLUSTERS = 5
MIN_ELAPSED = 5.0
MAX_ELAPSED_MINUTES = 45
SNAPSHOT_STEP_MINUTES = 2

print("=" * 70)
print("CONFIGURATION LSTM ULTRA")
print("=" * 70)
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"LSTM Units: {LSTM_UNITS}")
print(f"Batch Size: {BATCH_SIZE}")

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

print("✓ Clustering done")

# ============================================================
# 3. SNAPSHOTS
# ============================================================
def generate_snapshots(df_alert, step_minutes=SNAPSHOT_STEP_MINUTES, max_elapsed=MAX_ELAPSED_MINUTES):
    alert_groups = df_alert.groupby(['airport', 'airport_alert_id'])
    snapshots = []

    for (airport, alert_id), group in tqdm(alert_groups, desc="Snapshots", disable=True):
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

            snapshots.append({
                'airport': airport,
                'alert_id': alert_id,
                'snapshot_time': current,
                'remaining_min': remaining,
                'time_elapsed': time_elapsed,
                'hour': current.hour,
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

    return pd.DataFrame(snapshots)

print("\n3. Génération snapshots...")
train_snapshots = generate_snapshots(train_df)
valid_snapshots = generate_snapshots(valid_df)
train_snapshots = train_snapshots.reset_index(drop=True)
valid_snapshots = valid_snapshots.reset_index(drop=True)

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

train_snapshots = train_snapshots[train_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()
valid_snapshots = valid_snapshots[valid_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()

print(f"Train: {len(train_snapshots):,} | Valid: {len(valid_snapshots):,}")

# ============================================================
# 4. SÉQUENCES LSTM
# ============================================================
print("\n4. Préparation séquences...")

FEATURE_COLS = [
    'mean_amp', 'std_amp', 'mean_dist', 'std_dist', 'pct_cg',
    'rate_1min', 'rate_5min', 'rate_10min', 'rate_20min', 'rate_trend',
    'mean_inter', 'std_inter', 'last_inter', 'gap_ratio',
    'amp_trend', 'dist_trend'
]

def prepare_lstm_sequences(snapshots, sequence_length=SEQUENCE_LENGTH):
    sequences_X = []
    sequences_y = []
    
    snapshots = snapshots.copy()
    for col in FEATURE_COLS:
        if col not in snapshots.columns:
            snapshots[col] = 0.0
    
    alert_groups = snapshots.groupby(['airport', 'alert_id'])
    
    for (airport, alert_id), group in tqdm(alert_groups, desc="Seqs", disable=True):
        group = group.sort_values('snapshot_time').reset_index(drop=True).copy()
        features = group[FEATURE_COLS].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(features) >= sequence_length:
            scaler_local = StandardScaler()
            features_scaled = scaler_local.fit_transform(features)
            
            for i in range(len(features_scaled) - sequence_length):
                seq = features_scaled[i:i + sequence_length]
                target = group['remaining_min'].iloc[i + sequence_length]
                sequences_X.append(seq)
                sequences_y.append(max(0, target))
    
    return np.array(sequences_X), np.array(sequences_y)

X_train_seq, y_train_seq = prepare_lstm_sequences(train_snapshots, SEQUENCE_LENGTH)
X_valid_seq, y_valid_seq = prepare_lstm_sequences(valid_snapshots, SEQUENCE_LENGTH)

if len(X_train_seq) == 0 or len(X_valid_seq) == 0:
    print("\nERROR: No sequences generated!")
    print(f"X_train: {X_train_seq.shape}, X_valid: {X_valid_seq.shape}")
    exit(1)

y_train_seq = np.log1p(y_train_seq)
y_valid_seq = np.log1p(y_valid_seq)

print(f"X_train: {X_train_seq.shape} | X_valid: {X_valid_seq.shape}")

# ============================================================
# 5. NORMALISATION
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
# 6. MODÈLE LSTM ULTRA
# ============================================================
print("\n" + "=" * 70)
print("6. Construction modèle LSTM ultra")
print("=" * 70)

inputs = Input(shape=(SEQUENCE_LENGTH, n_features))

x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, 
                       kernel_regularizer=L2(L2_REG),
                       dropout=DROPOUT_RATE))(inputs)
x = Dropout(DROPOUT_RATE)(x)

x = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=False,
                       kernel_regularizer=L2(L2_REG),
                       dropout=DROPOUT_RATE))(x)
x = Dropout(DROPOUT_RATE)(x)

x = Dense(128, activation='relu', kernel_regularizer=L2(L2_REG))(x)
x = Dropout(DROPOUT_RATE)(x)

x = Dense(64, activation='relu', kernel_regularizer=L2(L2_REG))(x)
x = Dropout(DROPOUT_RATE)(x)

x = Dense(32, activation='relu', kernel_regularizer=L2(L2_REG))(x)

outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae', metrics=['mae'])

print(model.summary())

# ============================================================
# 7. ENTRAÎNEMENT
# ============================================================
print("\n" + "=" * 70)
print("7. Entraînement")
print("=" * 70)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001, verbose=1)
]

history = model.fit(
    X_train_norm, y_train_seq,
    validation_data=(X_valid_norm, y_valid_seq),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 8. ÉVALUATION
# ============================================================
print("\n" + "=" * 70)
print("8. Évaluation")
print("=" * 70)

y_train_pred = model.predict(X_train_norm, verbose=0)
y_valid_pred = model.predict(X_valid_norm, verbose=0)

y_train_raw = np.expm1(y_train_seq)
y_valid_raw = np.expm1(y_valid_seq)
y_train_pred_raw = np.expm1(y_train_pred.flatten())
y_valid_pred_raw = np.expm1(y_valid_pred.flatten())

mae_train = mean_absolute_error(y_train_raw, y_train_pred_raw)
mae_valid = mean_absolute_error(y_valid_raw, y_valid_pred_raw)
rmse_valid = np.sqrt(mean_squared_error(y_valid_raw, y_valid_pred_raw))

print(f"\nTrain - MAE: {mae_train:.4f} min")
print(f"Valid - MAE: {mae_valid:.4f} min")
print(f"Valid - RMSE: {rmse_valid:.4f} min")

# ============================================================
# 9. SAUVEGARDE
# ============================================================
print("\n" + "=" * 70)
print("9. Sauvegarde")
print("=" * 70)

model.save('lstm_model_ultra.h5')
print("✓ lstm_model_ultra.h5")

with open('lstm_scaler_x_ultra.pkl', 'wb') as f:
    pickle.dump(scaler_x, f)
print("✓ lstm_scaler_x_ultra.pkl")

with open('lstm_scaler_alert_ultra.pkl', 'wb') as f:
    pickle.dump(scaler_alert, f)
print("✓ lstm_scaler_alert_ultra.pkl")

with open('lstm_kmeans_ultra.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print("✓ lstm_kmeans_ultra.pkl")

history_df = pd.DataFrame(history.history)
history_df.to_csv('lstm_training_history_ultra.csv', index=False)
print("✓ lstm_training_history_ultra.csv")

print("\n" + "=" * 70)
print("ENTRAÎNEMENT TERMINÉ!")
print("=" * 70)

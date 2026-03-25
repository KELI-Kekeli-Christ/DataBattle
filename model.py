import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Chargement et split 80/20
# ------------------------------------------------------------
print("Chargement des données d'entraînement...")
df = pd.read_csv('segment_alerts_all_airports_train.csv', parse_dates=['date'])
df = df[df['airport_alert_id'].notna()].copy()

df_sorted = df.sort_values('date')
cutoff_date = df_sorted['date'].quantile(0.8)
train_df = df[df['date'] <= cutoff_date].copy()
valid_df = df[df['date'] > cutoff_date].copy()

valid_df.to_csv('segment_alerts_valid.csv', index=False)
print(f"Train : {len(train_df)} éclairs, {train_df['airport_alert_id'].nunique()} alertes")
print(f"Valid : {len(valid_df)} éclairs, {valid_df['airport_alert_id'].nunique()} alertes")

# ------------------------------------------------------------
# 2. Génération des snapshots par minute
# ------------------------------------------------------------
def generate_snapshots(df_alert):
    alert_groups = df_alert.groupby(['airport', 'airport_alert_id'])
    snapshots = []

    for (airport, alert_id), group in tqdm(alert_groups, desc="Snapshots"):
        group = group.sort_values('date')
        start = group['date'].min()
        end = group['date'].max()
        current = start

        while current <= end:
            past = group[group['date'] <= current]
            if len(past) == 0:
                current += timedelta(minutes=1)
                continue

            time_elapsed = (current - start).total_seconds() / 60.0
            remaining = (end - current).total_seconds() / 60.0
            n_so_far = len(past)
            mean_amp_sofar = past['amplitude'].mean()
            std_amp_sofar = past['amplitude'].std()
            mean_dist_sofar = past['dist'].mean()
            std_dist_sofar = past['dist'].std()
            min_dist_sofar = past['dist'].min()
            max_dist_sofar = past['dist'].max()
            pct_cloud_ground_sofar = (~past['icloud']).mean()

            recent_1min  = past[past['date'] >= current - timedelta(minutes=1)]
            recent_5min  = past[past['date'] >= current - timedelta(minutes=5)]
            recent_10min = past[past['date'] >= current - timedelta(minutes=10)]
            rate_1min  = len(recent_1min) / 1.0
            rate_5min  = len(recent_5min) / 5.0
            rate_10min = len(recent_10min) / 10.0
            last_dist = past.iloc[-1]['dist']

            recent_5min_prev = past[
                (past['date'] >= current - timedelta(minutes=10)) &
                (past['date'] <  current - timedelta(minutes=5))
            ]
            dist_trend = (
                recent_5min['dist'].mean() - recent_5min_prev['dist'].mean()
                if len(recent_5min) > 0 and len(recent_5min_prev) > 0 else 0.0
            )

            time_since_last = (current - past['date'].max()).total_seconds() / 60.0
            activity_ratio = rate_1min / (rate_10min + 1e-6)

            # ✅ Features inter-éclairs — signal clé de fin d'orage
            if len(past) >= 2:
                inter_times = past['date'].diff().dt.total_seconds().dropna() / 60.0
                mean_inter_time = inter_times.mean()
                last_inter_time = inter_times.iloc[-1] if len(inter_times) > 0 else 0.0
                if len(inter_times) >= 4:
                    inter_trend = inter_times.iloc[-2:].mean() - inter_times.iloc[-4:-2].mean()
                else:
                    inter_trend = 0.0
            else:
                mean_inter_time = 0.0
                last_inter_time = 0.0
                inter_trend = 0.0

            snapshots.append({
                'airport': airport,
                'alert_id': alert_id,
                'snapshot_time': current,
                'time_elapsed': time_elapsed,
                'n_so_far': n_so_far,
                'mean_amp_sofar': mean_amp_sofar,
                'std_amp_sofar': std_amp_sofar,
                'mean_dist_sofar': mean_dist_sofar,
                'std_dist_sofar': std_dist_sofar,
                'min_dist_sofar': min_dist_sofar,
                'max_dist_sofar': max_dist_sofar,
                'pct_cloud_ground_sofar': pct_cloud_ground_sofar,
                'rate_1min': rate_1min,
                'rate_5min': rate_5min,
                'rate_10min': rate_10min,
                'last_dist': last_dist,
                'dist_trend': dist_trend,
                'time_since_last': time_since_last,
                'activity_ratio': activity_ratio,
                'mean_inter_time': mean_inter_time,
                'last_inter_time': last_inter_time,
                'inter_trend': inter_trend,
                'hour': current.hour,
                'day_of_week': current.dayofweek,
                'month': current.month,
                'remaining_min': remaining,
            })
            current += timedelta(minutes=1)

    return pd.DataFrame(snapshots)

print("\nGénération des snapshots train...")
train_snapshots = generate_snapshots(train_df)
print("\nGénération des snapshots validation...")
valid_snapshots = generate_snapshots(valid_df)

# ------------------------------------------------------------
# 3. Clustering (fitté sur train uniquement)
# ------------------------------------------------------------
alert_features = ['duration_min', 'n_flashes', 'mean_amplitude', 'mean_dist', 'pct_cloud_ground']

train_alerts = train_df.groupby(['airport', 'airport_alert_id']).agg(
    duration_min=('date', lambda x: (x.max() - x.min()).total_seconds() / 60.0),
    n_flashes=('lightning_id', 'count'),
    mean_amplitude=('amplitude', 'mean'),
    mean_dist=('dist', 'mean'),
    pct_cloud_ground=('icloud', lambda x: (~x).mean())
).reset_index()

scaler = StandardScaler()
X_alert_scaled = scaler.fit_transform(train_alerts[alert_features])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
train_alerts['cluster'] = kmeans.fit_predict(X_alert_scaled)

train_snapshots = train_snapshots.merge(
    train_alerts[['airport', 'airport_alert_id', 'cluster']],
    left_on=['airport', 'alert_id'], right_on=['airport', 'airport_alert_id'],
    how='left'
)

valid_alerts = valid_df.groupby(['airport', 'airport_alert_id']).agg(
    duration_min=('date', lambda x: (x.max() - x.min()).total_seconds() / 60.0),
    n_flashes=('lightning_id', 'count'),
    mean_amplitude=('amplitude', 'mean'),
    mean_dist=('dist', 'mean'),
    pct_cloud_ground=('icloud', lambda x: (~x).mean())
).reset_index()

X_valid_alert_scaled = scaler.transform(valid_alerts[alert_features])
valid_alerts['cluster'] = kmeans.predict(X_valid_alert_scaled)

valid_snapshots = valid_snapshots.merge(
    valid_alerts[['airport', 'airport_alert_id', 'cluster']],
    left_on=['airport', 'alert_id'], right_on=['airport', 'airport_alert_id'],
    how='left'
)

assert not train_snapshots['cluster'].isna().any()
assert not valid_snapshots['cluster'].isna().any()

# ------------------------------------------------------------
# 4. Features et matrices X/y
# ------------------------------------------------------------
features = ['time_elapsed', 'n_so_far', 'mean_amp_sofar', 'std_amp_sofar',
            'mean_dist_sofar', 'std_dist_sofar', 'min_dist_sofar', 'max_dist_sofar',
            'pct_cloud_ground_sofar', 'rate_1min', 'rate_5min', 'rate_10min',
            'last_dist', 'dist_trend', 'time_since_last', 'activity_ratio',
            'mean_inter_time', 'last_inter_time', 'inter_trend',
            'hour', 'day_of_week', 'month', 'cluster']

airport_dummies_train = pd.get_dummies(train_snapshots['airport'], prefix='airport')
X_train = pd.concat([train_snapshots[features], airport_dummies_train], axis=1)
y_train = train_snapshots['remaining_min']  # ✅ retour à remaining_min

airport_dummies_valid = pd.get_dummies(valid_snapshots['airport'], prefix='airport')
for col in set(X_train.columns) - set(airport_dummies_valid.columns):
    if col.startswith('airport_'):
        airport_dummies_valid[col] = 0
X_valid = pd.concat([valid_snapshots[features], airport_dummies_valid], axis=1)
X_valid = X_valid[X_train.columns]

# ------------------------------------------------------------
# 5. Entraînement de 3 modèles : médiane, q10, q90
# ------------------------------------------------------------
params = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
              random_state=42, verbosity=0)

print("\nEntraînement modèle médiane...")
model_mid = xgb.XGBRegressor(objective='reg:squarederror', **params)
model_mid.fit(X_train, y_train)

print("Entraînement modèle pessimiste (q90)...")
model_q90 = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9, **params)
model_q90.fit(X_train, y_train)

print("Entraînement modèle optimiste (q10)...")
model_q10 = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.1, **params)
model_q10.fit(X_train, y_train)

# ------------------------------------------------------------
# 6. Prédictions et confidence calibrée (sans data leak)
# ------------------------------------------------------------
y_pred_mid = model_mid.predict(X_valid)
y_pred_q90 = model_q90.predict(X_valid)
y_pred_q10 = model_q10.predict(X_valid)

valid_snapshots['predicted_remaining'] = np.maximum(y_pred_mid, 0)

# ✅ Confidence = uniquement incertitude du modèle, pas de data leak
interval = np.maximum(y_pred_q90 - y_pred_q10, 1e-6)
valid_snapshots['confidence'] = 1 / (1 + interval / np.median(interval))

mae = mean_absolute_error(valid_snapshots['remaining_min'], valid_snapshots['predicted_remaining'])
print(f"\nMAE validation (minutes): {mae:.2f}")

# ------------------------------------------------------------
# 7. Génération de predictions_val.csv
# ------------------------------------------------------------
valid_snapshots['predicted_date_end_alert'] = (
    valid_snapshots['snapshot_time'] +
    pd.to_timedelta(valid_snapshots['predicted_remaining'], unit='minutes')
)

predictions_val = valid_snapshots[['airport', 'alert_id', 'snapshot_time',
                                    'predicted_date_end_alert', 'confidence']].copy()
predictions_val.rename(columns={
    'alert_id': 'airport_alert_id',
    'snapshot_time': 'prediction_date'
}, inplace=True)

predictions_val.to_csv('predictions_val.csv', index=False)
print("Fichier predictions_val.csv sauvegardé.")
print("→ Utiliser 'segment_alerts_valid.csv' dans le notebook d'évaluation.")

# ------------------------------------------------------------
# 8. Génération de predictions_test.csv (fichier jury)
# ------------------------------------------------------------
print("\nChargement des données de test...")
df_test = pd.read_csv('dataset_set.csv', parse_dates=['date'])
df_test_labeled = df_test[df_test['airport_alert_id'].notna()].copy()
print(f"Test : {len(df_test_labeled)} éclairs, {df_test_labeled['airport_alert_id'].nunique()} alertes")

print("\nGénération des snapshots de test...")
test_snapshots = generate_snapshots(df_test_labeled)

test_alerts = df_test_labeled.groupby(['airport', 'airport_alert_id']).agg(
    duration_min=('date', lambda x: (x.max() - x.min()).total_seconds() / 60.0),
    n_flashes=('lightning_id', 'count'),
    mean_amplitude=('amplitude', 'mean'),
    mean_dist=('dist', 'mean'),
    pct_cloud_ground=('icloud', lambda x: (~x).mean())
).reset_index()

X_test_alert_scaled = scaler.transform(test_alerts[alert_features])
test_alerts['cluster'] = kmeans.predict(X_test_alert_scaled)

test_snapshots = test_snapshots.merge(
    test_alerts[['airport', 'airport_alert_id', 'cluster']],
    left_on=['airport', 'alert_id'], right_on=['airport', 'airport_alert_id'],
    how='left'
)

airport_dummies_test = pd.get_dummies(test_snapshots['airport'], prefix='airport')
X_test = pd.concat([test_snapshots[features], airport_dummies_test], axis=1)
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_train.columns]

t_pred_mid = model_mid.predict(X_test)
t_pred_q90 = model_q90.predict(X_test)
t_pred_q10 = model_q10.predict(X_test)

test_snapshots['predicted_remaining'] = np.maximum(t_pred_mid, 0)

interval_test = np.maximum(t_pred_q90 - t_pred_q10, 1e-6)
test_snapshots['confidence'] = 1 / (1 + interval_test / np.median(interval_test))

test_snapshots['predicted_date_end_alert'] = (
    test_snapshots['snapshot_time'] +
    pd.to_timedelta(test_snapshots['predicted_remaining'], unit='minutes')
)

predictions_test = test_snapshots[['airport', 'alert_id', 'snapshot_time',
                                    'predicted_date_end_alert', 'confidence']].copy()
predictions_test.rename(columns={
    'alert_id': 'airport_alert_id',
    'snapshot_time': 'prediction_date'
}, inplace=True)

predictions_test.to_csv('predictions_test.csv', index=False)
print("Fichier predictions_test.csv sauvegardé. → C'est ce fichier à envoyer au jury.")
import json

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
import joblib  # pour sauvegarder les modèles
from codecarbon import EmissionsTracker

# ============================================================
# CONFIGURATION
# ============================================================
tracker = EmissionsTracker(project_name="orages_prediction", measure_power_secs=10)
QUANTILE_LOW  = 0.10
QUANTILE_HIGH = 0.95

R_ACCEPT    = 0.02
N_CLUSTERS  = 5
MIN_ELAPSED = 5.0
MAX_GAP_MINUTES = 30
MIN_DIST_KM = 3

# Optimisation temps / surapprentissage
SNAPSHOT_STEP_MINUTES = 2        # au lieu de 1 minute
MAX_ELAPSED_MINUTES   = 45       # snapshots trop anciens ignorés

# ----------------------------
# Modèle principal (optimisé MAE)
# ----------------------------
POINT_PARAMS = dict(
    n_estimators=600,          # réduit
    max_depth=5,               # réduit
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,        # augmenté
    reg_alpha=1.0,             # augmenté
    reg_lambda=3.0,            # augmenté
    random_state=42,
    verbosity=0,
    tree_method='hist',
    objective='reg:absoluteerror'
)

# ----------------------------
# Modèles quantiles (incertitude)
# ----------------------------
QUANTILE_PARAMS = dict(
    n_estimators=300,          # réduit
    max_depth=4,               # réduit
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    verbosity=0,
    tree_method='hist',
)

# ----------------------------
# Stacking model
# ----------------------------
STACK_PARAMS = dict(
    n_estimators=200,          # réduit
    max_depth=2,               # réduit
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.5,             # augmenté
    reg_lambda=2.0,            # augmenté
    random_state=42,
    verbosity=0,
    tree_method='hist',
    objective='reg:absoluteerror'
)

# ============================================================
# 1. CHARGEMENT ET SPLIT 80/20
# ============================================================
print("=" * 70)
print("1. Chargement des données")
print("=" * 70)

df = pd.read_csv('segment_alerts_all_airports_train.csv', parse_dates=['date'])
df = df[df['airport_alert_id'].notna()].copy()
df_sorted = df.sort_values('date')
cutoff_date = df_sorted['date'].quantile(0.8)

train_df = df[df['date'] <= cutoff_date].copy()
valid_df = df[df['date'] >  cutoff_date].copy()

valid_df.to_csv('segment_alerts_valid.csv', index=False)

print(f"Train : {len(train_df):>8,} éclairs | {train_df['airport_alert_id'].nunique():>5} alertes")
print(f"Valid : {len(valid_df):>8,} éclairs | {valid_df['airport_alert_id'].nunique():>5} alertes")

# ============================================================
# 2. FEATURES ALERTES GLOBALES
# ============================================================
def compute_alert_features(df_raw):
    return df_raw.groupby(['airport', 'airport_alert_id']).agg(
        duration_min   = ('date', lambda x: (x.max() - x.min()).total_seconds() / 60),
        n_flashes      = ('lightning_id', 'count'),
        mean_amplitude = ('amplitude', 'mean'),
        mean_dist      = ('dist', 'mean'),
        pct_cg         = ('icloud', lambda x: (~x).mean()),
    ).reset_index()

print("\n2. Calcul des features alertes...")
train_alerts = compute_alert_features(train_df)
valid_alerts = compute_alert_features(valid_df)

ALERT_FEATURES = ['duration_min', 'n_flashes', 'mean_amplitude', 'mean_dist', 'pct_cg']

# ============================================================
# 3. CLUSTERING
# ============================================================
print("\n3. Clustering des alertes...")

scaler = StandardScaler()
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

train_alerts['cluster'] = kmeans.fit_predict(
    scaler.fit_transform(train_alerts[ALERT_FEATURES])
)
valid_alerts['cluster'] = kmeans.predict(
    scaler.transform(valid_alerts[ALERT_FEATURES])
)

# Stats de durée par cluster
cluster_duration_stats = train_alerts.groupby('cluster')['duration_min'].agg(
    ['median', 'mean', 'std']
).reset_index()
cluster_duration_stats.columns = [
    'cluster', 'cluster_dur_median', 'cluster_dur_mean', 'cluster_dur_std'
]

print("Stats durée par cluster :")
print(cluster_duration_stats)

# ============================================================
# 4. GÉNÉRATION DES SNAPSHOTS ENRICHIS (optimisée)
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
            # Avancer l'index des points observés
            while idx_end + 1 < len(group) and group['date'].iloc[idx_end + 1] <= current:
                idx_end += 1

            past = group.iloc[:idx_end + 1]
            if len(past) == 0:
                current += timedelta(minutes=step_minutes)
                continue

            time_elapsed = (current - start).total_seconds() / 60.0
            if time_elapsed > max_elapsed:
                break

            remaining    = (end - current).total_seconds() / 60.0
            n_so_far     = len(past)

            # Stats globales
            mean_amp_sofar  = past['amplitude'].mean()
            std_amp_sofar   = past['amplitude'].std(ddof=0)
            mean_dist_sofar = past['dist'].mean()
            std_dist_sofar  = past['dist'].std(ddof=0)
            min_dist_sofar  = past['dist'].min()
            max_dist_sofar  = past['dist'].max()
            pct_cg_sofar    = (~past['icloud']).mean()
            last_dist       = past.iloc[-1]['dist']
            last_amp        = past.iloc[-1]['amplitude']

            # Fenêtres temporelles
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
            rate_decay     = rate_5min / (rate_20min + 1e-6)
            rate_accel     = rate_1min - rate_5min
            rate_accel2    = rate_5min - rate_20min

            # Amplitude / distance
            amp_recent = w5['amplitude'].mean() if len(w5) > 0 else mean_amp_sofar
            amp_old_df = past[past['date'] < t5]
            amp_old    = amp_old_df['amplitude'].mean() if len(amp_old_df) > 0 else amp_recent
            amp_decay  = amp_recent - amp_old
            amp_ratio_recent = amp_recent / (mean_amp_sofar + 1e-6)

            dist_recent = w5['dist'].mean() if len(w5) > 0 else mean_dist_sofar
            dist_old_df = past[past['date'] < t5]
            dist_old    = dist_old_df['dist'].mean() if len(dist_old_df) > 0 else dist_recent
            dist_trend  = dist_recent - dist_old
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
                std_inter_time  = inter_times.std(ddof=0) if len(inter_times) > 1 else 0.0
                last_inter_time = inter_times.iloc[-1]
                max_inter_time  = inter_times.max()
                p50_inter       = inter_times.quantile(0.50)
                p90_inter       = inter_times.quantile(0.90)
                gap_ratio       = last_inter_time / (p50_inter + 1e-6)
                inter_trend     = (
                    inter_times.iloc[-2:].mean() - inter_times.iloc[-4:-2].mean()
                    if len(inter_times) >= 4 else 0.0
                )
            else:
                mean_inter_time = 0.0
                std_inter_time  = 0.0
                last_inter_time = 0.0
                max_inter_time  = 0.0
                p50_inter       = 0.0
                p90_inter       = 0.0
                gap_ratio       = 0.0
                inter_trend     = 0.0

            # Peak
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

print("\n4. Génération des snapshots...")
train_snapshots = generate_snapshots(train_df)
valid_snapshots = generate_snapshots(valid_df)

train_snapshots = train_snapshots.reset_index(drop=True)
valid_snapshots = valid_snapshots.reset_index(drop=True)

print(f"Train snapshots : {len(train_snapshots):,}")
print(f"Valid snapshots : {len(valid_snapshots):,}")

# ============================================================
# 5. MERGE CLUSTERS + STATS CLUSTERS
# ============================================================
print("\n5. Merge clusters + stats clusters...")

def enrich_snapshots_with_clusters(snaps, alerts, cluster_stats):
    snaps = snaps.merge(
        alerts[['airport', 'airport_alert_id', 'cluster']],
        left_on=['airport', 'alert_id'],
        right_on=['airport', 'airport_alert_id'],
        how='left'
    ).drop(columns=['airport_alert_id'])

    snaps = snaps.merge(cluster_stats, on='cluster', how='left')

    snaps['elapsed_vs_cluster_median'] = snaps['time_elapsed'] / (snaps['cluster_dur_median'] + 1e-6)
    snaps['elapsed_vs_cluster_mean']   = snaps['time_elapsed'] / (snaps['cluster_dur_mean'] + 1e-6)
    snaps['cluster_dur_std']           = snaps['cluster_dur_std'].fillna(0.0)

    return snaps

train_snapshots = enrich_snapshots_with_clusters(train_snapshots, train_alerts, cluster_duration_stats)
valid_snapshots = enrich_snapshots_with_clusters(valid_snapshots, valid_alerts, cluster_duration_stats)

# ============================================================
# 6. FILTRAGE DES SNAPSHOTS PEU INFORMATIFS
# ============================================================
print("\n6. Filtrage snapshots (time_elapsed >= MIN_ELAPSED)...")

train_snapshots = train_snapshots[train_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()
valid_snapshots = valid_snapshots[valid_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()

print(f"Train filtré : {len(train_snapshots):,}")
print(f"Valid filtré : {len(valid_snapshots):,}")

# ============================================================
# 7. FEATURES
# ============================================================
FEATURES = [
    'time_elapsed', 'hour', 'day_of_week', 'month',
    'n_so_far',

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

    'cluster',
    'cluster_dur_median', 'cluster_dur_mean', 'cluster_dur_std',
    'elapsed_vs_cluster_median', 'elapsed_vs_cluster_mean',
]

def build_X(snaps, ref_columns=None):
    airport_dummies = pd.get_dummies(snaps['airport'], prefix='airport')

    X = pd.concat([
        snaps[FEATURES].reset_index(drop=True),
        airport_dummies.reset_index(drop=True)
    ], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if ref_columns is not None:
        for col in ref_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[ref_columns]

    return X

X_train = build_X(train_snapshots)
X_valid = build_X(valid_snapshots, ref_columns=X_train.columns.tolist())
TRAIN_COLUMNS = X_train.columns.tolist()

# Cible log-transformée
y_train_raw = train_snapshots['remaining_min'].values
y_valid_raw = valid_snapshots['remaining_min'].values

y_train = np.log1p(y_train_raw)
y_valid = np.log1p(y_valid_raw)

print(f"\n7. Matrices")
print(f"X_train : {X_train.shape}")
print(f"X_valid : {X_valid.shape}")

# ============================================================
# 8. ENTRAÎNEMENT DES MODÈLES (avec early stopping via xgb.train)
# ============================================================
print("\n" + "=" * 70)
print("8. Entraînement des modèles")
print("=" * 70)

tracker.start()

def train_booster(params, dtrain, dvalid, num_boost_round=None, early_stopping=50, verbose=False):
    if num_boost_round is None:
        num_boost_round = params.get('n_estimators', 1000)
    params_clean = {k: v for k, v in params.items() if k not in ['n_estimators', 'verbosity']}
    if 'verbosity' in params:
        params_clean['verbosity'] = params['verbosity']
    if 'objective' not in params_clean:
        params_clean['objective'] = 'reg:absoluteerror'

    evals = [(dtrain, 'train'), (dvalid, 'eval')]
    booster = xgb.train(
        params_clean,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping,
        verbose_eval=verbose
    )
    return booster

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 8a. Modèle principal
print("\n[8a] Modèle principal (MAE, log target)...")
model_main = train_booster(POINT_PARAMS, dtrain, dvalid, early_stopping=50, verbose=False)

# 8b. Modèles quantiles
print(f"\n[8b] Modèle quantile bas q{QUANTILE_LOW}...")
params_low = QUANTILE_PARAMS.copy()
params_low.update({'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_LOW})
model_low = train_booster(params_low, dtrain, dvalid, early_stopping=50, verbose=False)

print(f"\n[8c] Modèle quantile haut q{QUANTILE_HIGH}...")
params_high = QUANTILE_PARAMS.copy()
params_high.update({'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_HIGH})
model_high = train_booster(params_high, dtrain, dvalid, early_stopping=50, verbose=False)

# 8d. Modèles par cluster
print("\n[8d] Modèles spécialisés par cluster...")
cluster_models = {}

for c in sorted(train_snapshots['cluster'].dropna().unique()):
    mask_train = train_snapshots['cluster'].values == c
    n_cluster = mask_train.sum()

    if n_cluster < 200:
        print(f"   Cluster {c} ignoré (pas assez de snapshots : {n_cluster})")
        continue

    mask_valid = valid_snapshots['cluster'].values == c
    if mask_valid.sum() == 0:
        print(f"   Cluster {c} absent en validation, ignoré")
        continue

    Xc_train = X_train[mask_train]
    yc_train = y_train[mask_train]
    Xc_valid = X_valid[mask_valid]
    yc_valid = y_valid[mask_valid]

    dtrain_c = xgb.DMatrix(Xc_train, label=yc_train)
    dvalid_c = xgb.DMatrix(Xc_valid, label=yc_valid)

    m = train_booster(POINT_PARAMS, dtrain_c, dvalid_c, early_stopping=50, verbose=False)
    cluster_models[c] = m

    print(f"   Cluster {c} : {n_cluster:,} snapshots")

emissions_training = tracker.stop()
print(f"\n💡 Émissions CO₂ pour l'entraînement : {emissions_training:.4f} kg CO₂eq")

# Sauvegarde des modèles (optionnel)
joblib.dump(model_main, 'model_main.pkl')
joblib.dump(model_low, 'model_low.pkl')
joblib.dump(model_high, 'model_high.pkl')
joblib.dump(cluster_models, 'cluster_models.pkl')

# ============================================================
# 9. PRÉDICTIONS VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("9. Prédictions validation")
print("=" * 70)

# Global
preds_main_log = model_main.predict(xgb.DMatrix(X_valid))
preds_main = np.expm1(preds_main_log)

# Quantiles
preds_low_log  = model_low.predict(xgb.DMatrix(X_valid))
preds_high_log = model_high.predict(xgb.DMatrix(X_valid))

preds_low  = np.expm1(preds_low_log)
preds_high = np.expm1(preds_high_log)

# Cluster models
preds_cluster = np.zeros(len(valid_snapshots))
for c, m in cluster_models.items():
    mask = (valid_snapshots['cluster'].values == c)
    if mask.sum() > 0:
        Xc_valid = X_valid.loc[mask]
        preds_cluster[mask] = np.expm1(m.predict(xgb.DMatrix(Xc_valid)))

missing_cluster_mask = preds_cluster <= 0
preds_cluster[missing_cluster_mask] = preds_main[missing_cluster_mask]

# Clipping
preds_main    = np.maximum(preds_main, 0.0)
preds_low     = np.maximum(preds_low, 0.0)
preds_high    = np.maximum(preds_high, preds_low + 1e-6)
preds_cluster = np.maximum(preds_cluster, 0.0)

# ============================================================
# 10. STACKING LEARNED COMBINATION
# ============================================================
print("\n" + "=" * 70)
print("10. Stacking appris")
print("=" * 70)

stack_valid = pd.DataFrame({
    'pred_main': preds_main,
    'pred_cluster': preds_cluster,
    'pred_low': preds_low,
    'pred_high': preds_high,
    'interval': preds_high - preds_low,

    'time_elapsed': valid_snapshots['time_elapsed'].values,
    'ema_rate': valid_snapshots['ema_rate'].values,
    'gap_ratio': valid_snapshots['gap_ratio'].values,
    'time_since_last': valid_snapshots['time_since_last'].values,
    'rate_decay': valid_snapshots['rate_decay'].values,
    'elapsed_vs_cluster_median': valid_snapshots['elapsed_vs_cluster_median'].values,
    'cluster': valid_snapshots['cluster'].values,
})

stack_valid = pd.get_dummies(stack_valid, columns=['cluster'], prefix='cluster')
stack_valid = stack_valid.replace([np.inf, -np.inf], np.nan).fillna(0.0)

stack_model = xgb.XGBRegressor(**STACK_PARAMS)
stack_model.fit(stack_valid, y_valid_raw)

preds_combined = stack_model.predict(stack_valid)
preds_combined = np.maximum(preds_combined, 0.0)
preds_combined = np.minimum(preds_combined, preds_high + 10)

mae_main    = mean_absolute_error(y_valid_raw, preds_main)
mae_cluster = mean_absolute_error(y_valid_raw, preds_cluster)
mae_stack   = mean_absolute_error(y_valid_raw, preds_combined)

print(f"MAE modèle global   : {mae_main:.2f} min")
print(f"MAE modèle cluster  : {mae_cluster:.2f} min")
print(f"MAE stacking final  : {mae_stack:.2f} min")

valid_snapshots['predicted_remaining'] = preds_combined

# ============================================================
# 11. CALIBRATION DU SCORE DE CONFIANCE
# ============================================================
print("\n" + "=" * 70)
print("11. Calibration du score de confiance")
print("=" * 70)

is_safe = (preds_combined >= y_valid_raw).astype(float)
print(f"Taux naturel de sécurité : {is_safe.mean():.3f}")

gap_norm = np.clip(valid_snapshots['gap_ratio'].values / 5.0, 0, 1)
tsl_norm = np.clip(valid_snapshots['time_since_last'].values / 15.0, 0, 1)
rate_norm = 1.0 - np.clip(valid_snapshots['ema_rate'].values / 3.0, 0, 1)

interval = np.maximum(preds_high - preds_low, 1e-6)
tight_norm = 1.0 - np.clip(interval / (np.percentile(interval, 95) + 1e-6), 0, 1)

elapsed_norm = np.clip(valid_snapshots['time_elapsed'].values / 60.0, 0, 1)
cluster_progress = np.clip(valid_snapshots['elapsed_vs_cluster_median'].values / 1.5, 0, 1)

raw_score = (
    0.25 * gap_norm +
    0.22 * tsl_norm +
    0.18 * rate_norm +
    0.15 * tight_norm +
    0.10 * elapsed_norm +
    0.10 * cluster_progress
)

score_min, score_max = raw_score.min(), raw_score.max()
confidence = (raw_score - score_min) / (score_max - score_min + 1e-6)

valid_snapshots['confidence'] = confidence

print(f"Score moyen : {confidence.mean():.3f}")
print(f"Score min   : {confidence.min():.3f}")
print(f"Score max   : {confidence.max():.3f}")

# ============================================================
# 12. GÉNÉRATION predictions_val.csv
# ============================================================
print("\n" + "=" * 70)
print("12. Génération predictions_val.csv")
print("=" * 70)

valid_snapshots['predicted_date_end_alert'] = (
    valid_snapshots['snapshot_time'] +
    pd.to_timedelta(valid_snapshots['predicted_remaining'], unit='minutes')
)

predictions_val = valid_snapshots[[
    'airport', 'alert_id', 'snapshot_time',
    'predicted_date_end_alert', 'confidence'
]].rename(columns={
    'alert_id': 'airport_alert_id',
    'snapshot_time': 'prediction_date',
})

predictions_val.to_csv('predictions_val.csv', index=False)
print(f"predictions_val.csv : {len(predictions_val):,} lignes")

# ============================================================
# 13. SIMULATION ÉVALUATION JURY (validation)
# ============================================================
print("\n" + "=" * 70)
print("13. Simulation évaluation jury (validation)")
print("=" * 70)

tot_lightnings = len(valid_df[valid_df['dist'] < MIN_DIST_KM])
alerts_grouped = valid_df.groupby(['airport', 'airport_alert_id'])

predictions_val_eval = predictions_val.copy()
predictions_val_eval['predicted_date_end_alert'] = pd.to_datetime(
    predictions_val_eval['predicted_date_end_alert'], utc=True
)

thetas = [i / 100 for i in range(1, 100)]
results = {}

print("Scan des thetas...")
for theta in tqdm(thetas):
    pred_over = predictions_val_eval[predictions_val_eval['confidence'] >= theta]

    if len(pred_over) == 0:
        results[theta] = (0, tot_lightnings)
        continue

    pred_min = pred_over.groupby(['airport', 'airport_alert_id'])['predicted_date_end_alert'].min()

    gain = 0
    missed = 0

    for (airport, alert_id), end_pred in pred_min.items():
        try:
            lightnings = alerts_grouped.get_group((airport, alert_id))
        except KeyError:
            continue

        end_baseline = pd.to_datetime(lightnings['date'], utc=True).max() + pd.Timedelta(minutes=MAX_GAP_MINUTES)
        gain += (end_baseline - end_pred).total_seconds()

        missed += sum(
            pd.to_datetime(lightnings.loc[lightnings['dist'] < MIN_DIST_KM, 'date'], utc=True) > end_pred
        )

    results[theta] = (gain, missed)

gains = {t: v[0] / 3600 for t, v in results.items()}
risks = {t: v[1] / tot_lightnings for t, v in results.items()}

print("\nRésultats par theta (extrait)")
print(f"{'θ':>6} | {'Gain (h)':>10} | {'Risque R':>10} | {'OK ?':>6}")
print("-" * 44)
for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    if t in gains:
        ok = "✓" if risks[t] < R_ACCEPT else "✗"
        print(f"{t:>6.2f} | {gains[t]:>10.1f} | {risks[t]:>10.4f} | {ok:>6}")

valid_thetas = [(gains[t], t, risks[t]) for t in thetas if risks[t] < R_ACCEPT and gains[t] > 0]

if valid_thetas:
    best_gain, best_theta, best_risk = max(valid_thetas)
    print("\n✓ MEILLEUR THETA TROUVÉ (validation)")
    print(f"θ optimal  : {best_theta:.2f}")
    print(f"Gain total : {best_gain:.1f} h")
    print(f"Risque R   : {best_risk:.4f} (< {R_ACCEPT})")
else:
    print("\n✗ Aucun theta avec gain > 0 ET R < R_ACCEPT")
    best_theta = 0.99
    best_gain = gains.get(0.99, 0)
    best_risk = risks.get(0.99, 1)

# ============================================================
# 14. GÉNÉRATION ET ÉVALUATION SUR TEST ÉTIQUETÉ
# ============================================================
print("\n" + "=" * 70)
print("14. Génération et évaluation sur test étiqueté")
print("=" * 70)

# Lancer un tracker pour l'inference
tracker_inf = EmissionsTracker(project_name="orages_inference", measure_power_secs=10)
tracker_inf.start()

df_test = pd.read_csv('segment_alerts_all_airports_eval.csv', parse_dates=['date'])
df_test_labeled = df_test[df_test['airport_alert_id'].notna()].copy()

def compute_true_remaining(snapshots, df_alert):
    alert_end = df_alert.groupby(['airport', 'airport_alert_id'])['date'].max().to_dict()
    snapshots['true_remaining_min'] = snapshots.apply(
        lambda row: (alert_end.get((row['airport'], row['alert_id'])) - row['snapshot_time']).total_seconds() / 60,
        axis=1
    )
    return snapshots

print("Calcul des features alertes test...")
test_alerts = compute_alert_features(df_test_labeled)
test_alerts['cluster'] = kmeans.predict(
    scaler.transform(test_alerts[ALERT_FEATURES])
)

print("Génération snapshots test...")
test_snapshots = generate_snapshots(df_test_labeled)
test_snapshots = enrich_snapshots_with_clusters(test_snapshots, test_alerts, cluster_duration_stats)

test_snapshots = test_snapshots[test_snapshots['time_elapsed'] >= MIN_ELAPSED].copy()
test_snapshots = test_snapshots.reset_index(drop=True)
test_snapshots = compute_true_remaining(test_snapshots, df_test_labeled)

X_test = build_X(test_snapshots, ref_columns=TRAIN_COLUMNS)

# Prédictions test
t_main = np.expm1(model_main.predict(xgb.DMatrix(X_test)))
t_low  = np.expm1(model_low.predict(xgb.DMatrix(X_test)))
t_high = np.expm1(model_high.predict(xgb.DMatrix(X_test)))

t_cluster = np.zeros(len(test_snapshots))
for c, m in cluster_models.items():
    mask = (test_snapshots['cluster'].values == c)
    if mask.sum() > 0:
        Xc_test = X_test.loc[mask]
        t_cluster[mask] = np.expm1(m.predict(xgb.DMatrix(Xc_test)))

missing_cluster_mask = t_cluster <= 0
t_cluster[missing_cluster_mask] = t_main[missing_cluster_mask]

t_main    = np.maximum(t_main, 0.0)
t_low     = np.maximum(t_low, 0.0)
t_high    = np.maximum(t_high, t_low + 1e-6)
t_cluster = np.maximum(t_cluster, 0.0)

# Stacking test
stack_test = pd.DataFrame({
    'pred_main': t_main,
    'pred_cluster': t_cluster,
    'pred_low': t_low,
    'pred_high': t_high,
    'interval': t_high - t_low,

    'time_elapsed': test_snapshots['time_elapsed'].values,
    'ema_rate': test_snapshots['ema_rate'].values,
    'gap_ratio': test_snapshots['gap_ratio'].values,
    'time_since_last': test_snapshots['time_since_last'].values,
    'rate_decay': test_snapshots['rate_decay'].values,
    'elapsed_vs_cluster_median': test_snapshots['elapsed_vs_cluster_median'].values,
    'cluster': test_snapshots['cluster'].values,
})

stack_test = pd.get_dummies(stack_test, columns=['cluster'], prefix='cluster')
stack_test = stack_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

for col in stack_valid.columns:
    if col not in stack_test.columns:
        stack_test[col] = 0
stack_test = stack_test[stack_valid.columns]

t_combined = stack_model.predict(stack_test)
t_combined = np.maximum(t_combined, 0.0)
t_combined = np.minimum(t_combined, t_high + 10)

test_snapshots['predicted_remaining'] = t_combined

# Score de confiance test
gap_norm_t = np.clip(test_snapshots['gap_ratio'].values / 5.0, 0, 1)
tsl_norm_t = np.clip(test_snapshots['time_since_last'].values / 15.0, 0, 1)
rate_norm_t = 1.0 - np.clip(test_snapshots['ema_rate'].values / 3.0, 0, 1)

interval_t = np.maximum(t_high - t_low, 1e-6)
tight_norm_t = 1.0 - np.clip(interval_t / (np.percentile(interval_t, 95) + 1e-6), 0, 1)

elapsed_norm_t = np.clip(test_snapshots['time_elapsed'].values / 60.0, 0, 1)
cluster_progress_t = np.clip(test_snapshots['elapsed_vs_cluster_median'].values / 1.5, 0, 1)

raw_t = (
    0.25 * gap_norm_t +
    0.22 * tsl_norm_t +
    0.18 * rate_norm_t +
    0.15 * tight_norm_t +
    0.10 * elapsed_norm_t +
    0.10 * cluster_progress_t
)

test_snapshots['confidence'] = (raw_t - score_min) / (score_max - score_min + 1e-6)

# Évaluation des performances sur test
mae_test_main = mean_absolute_error(test_snapshots['true_remaining_min'], t_main)
mae_test_cluster = mean_absolute_error(test_snapshots['true_remaining_min'], t_cluster)
mae_test_stack = mean_absolute_error(test_snapshots['true_remaining_min'], t_combined)

print(f"\nMAE test global   : {mae_test_main:.2f} min")
print(f"MAE test cluster  : {mae_test_cluster:.2f} min")
print(f"MAE test stacking : {mae_test_stack:.2f} min")

if mae_test_stack > mae_stack * 1.1:
    print("\n⚠️  Surapprentissage potentiel détecté : MAE test > 10% de la MAE validation")
else:
    print("\n✓ Bonne généralisation : MAE test proche de la validation")

# ----------------------------
# Fichier final test (toutes les prédictions)
# ----------------------------
test_snapshots['predicted_date_end_alert'] = (
    test_snapshots['snapshot_time'] +
    pd.to_timedelta(test_snapshots['predicted_remaining'], unit='minutes')
)


# Après avoir calculé test_snapshots et avant de créer predictions_test
test_snapshots['true_end_alert'] = test_snapshots['snapshot_time'] + pd.to_timedelta(test_snapshots['true_remaining_min'], unit='minutes')

predictions_test = test_snapshots[[
    'airport', 'alert_id', 'snapshot_time', 'true_end_alert',
    'predicted_date_end_alert', 'confidence'
]].rename(columns={
    'alert_id': 'airport_alert_id',
    'snapshot_time': 'prediction_date',
})

predictions_test.to_csv('predictions_test.csv', index=False)
print(f"predictions_test.csv : {len(predictions_test):,} lignes")

emissions_inference = tracker_inf.stop()
print(f"\n💡 Émissions CO₂ pour l'inférence sur le test : {emissions_inference:.4f} kg CO₂eq")

emissions_data = {
    "training_kg_co2": emissions_training,
    "inference_kg_co2": emissions_inference,
    "training_energy_kwh": None,
    "inference_energy_kwh": None,
    "description": "Émissions CO₂ pour l'entraînement et l'inférence sur le jeu de test"
}

# Sauvegarde
with open("co2_emissions.json", "w") as f:
    json.dump(emissions_data, f, indent=2)

print(f"💾 Émissions sauvegardées dans co2_emissions.json")
# ============================================================
# 15. ÉVALUATION SUR LE TEST AVEC θ OPTIMAL DE VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("15. Évaluation sur le test avec θ optimal validation")
print("=" * 70)

print(f"Utilisation du theta optimal validation : {best_theta:.2f}")

tot_lightnings_test = len(df_test_labeled[df_test_labeled['dist'] < MIN_DIST_KM])
alerts_grouped_test = df_test_labeled.groupby(['airport', 'airport_alert_id'])

test_eval = predictions_test.copy()
test_eval['predicted_date_end_alert'] = pd.to_datetime(test_eval['predicted_date_end_alert'], utc=True)

pred_over_test = test_eval[test_eval['confidence'] >= best_theta]

if len(pred_over_test) == 0:
    gain_test = 0
    missed_test = tot_lightnings_test
else:
    pred_min_test = pred_over_test.groupby(['airport', 'airport_alert_id'])['predicted_date_end_alert'].min()
    gain_test = 0
    missed_test = 0
    for (airport, alert_id), end_pred in pred_min_test.items():
        try:
            lightnings = alerts_grouped_test.get_group((airport, alert_id))
        except KeyError:
            continue
        end_baseline = pd.to_datetime(lightnings['date'], utc=True).max() + pd.Timedelta(minutes=MAX_GAP_MINUTES)
        gain_test += (end_baseline - end_pred).total_seconds()
        missed_test += sum(
            pd.to_datetime(lightnings.loc[lightnings['dist'] < MIN_DIST_KM, 'date'], utc=True) > end_pred
        )

gain_test_hours = gain_test / 3600
risk_test = missed_test / tot_lightnings_test if tot_lightnings_test > 0 else 0.0

print(f"\nRésultats sur le test (θ = {best_theta:.2f}) :")
print(f"Gain total : {gain_test_hours:.1f} heures")
print(f"Risque R   : {risk_test:.4f} ({'✓' if risk_test < R_ACCEPT else '✗'})")
print(f"Éclairs dangereux totaux dans le test : {tot_lightnings_test:,}")
print(f"Éclairs manqués : {missed_test:,}")

# ============================================================
# 16. RECHERCHE DU THETA OPTIMAL SUR LE TEST (pour information)
# ============================================================
print("\n" + "=" * 70)
print("16. Recherche du theta optimal sur le test (information)")
print("=" * 70)

thetas_test = [i / 100 for i in range(1, 100)]
results_test = {}

for theta in tqdm(thetas_test):
    pred_over = test_eval[test_eval['confidence'] >= theta]
    if len(pred_over) == 0:
        results_test[theta] = (0, tot_lightnings_test)
        continue
    pred_min = pred_over.groupby(['airport', 'airport_alert_id'])['predicted_date_end_alert'].min()
    gain = 0
    missed = 0
    for (airport, alert_id), end_pred in pred_min.items():
        try:
            lightnings = alerts_grouped_test.get_group((airport, alert_id))
        except KeyError:
            continue
        end_baseline = pd.to_datetime(lightnings['date'], utc=True).max() + pd.Timedelta(minutes=MAX_GAP_MINUTES)
        gain += (end_baseline - end_pred).total_seconds()
        missed += sum(pd.to_datetime(lightnings.loc[lightnings['dist'] < MIN_DIST_KM, 'date'], utc=True) > end_pred)
    results_test[theta] = (gain, missed)

gains_test_dict = {t: v[0] / 3600 for t, v in results_test.items()}
risks_test_dict = {t: v[1] / tot_lightnings_test for t, v in results_test.items()}

valid_thetas_test = [(gains_test_dict[t], t, risks_test_dict[t]) for t in thetas_test if risks_test_dict[t] < R_ACCEPT and gains_test_dict[t] > 0]
if valid_thetas_test:
    best_gain_test, best_theta_test, best_risk_test = max(valid_thetas_test)
    print(f"\nTheta optimal sur le test : {best_theta_test:.2f}")
    print(f"Gain test correspondant : {best_gain_test:.1f} h")
    print(f"Risque test correspondant : {best_risk_test:.4f}")
else:
    print("\nAucun theta sur le test ne respecte R < 0.02 avec gain positif.")

# ============================================================
# 17. RÉSUMÉ FINAL
# ============================================================
print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"MAE validation globale    : {mae_main:.2f} min")
print(f"MAE validation cluster    : {mae_cluster:.2f} min")
print(f"MAE validation STACKING   : {mae_stack:.2f} min")
print(f"MAE test STACKING         : {mae_test_stack:.2f} min")

print(f"\nTheta optimal (validation) : {best_theta:.2f}")
print(f"Gain validation           : {best_gain:.1f} heures")
print(f"Risque validation         : {best_risk:.4f} ({'✓' if best_risk < R_ACCEPT else '✗'})")

print(f"\nPerformances sur le test avec θ validation :")
print(f"Gain test                 : {gain_test_hours:.1f} heures")
print(f"Risque test               : {risk_test:.4f} ({'✓' if risk_test < R_ACCEPT else '✗'})")

if 'best_theta_test' in locals():
    print(f"\nTheta optimal sur le test (info) : {best_theta_test:.2f}")
    print(f"Gain test correspondant          : {best_gain_test:.1f} h")
    print(f"Risque test correspondant        : {best_risk_test:.4f}")

# Ajout des métriques environnementales
print(f"\n🌍 Impact environnemental :")
print(f"   Émissions entraînement : {emissions_training:.4f} kg CO₂eq")
print(f"   Émissions inférence    : {emissions_inference:.4f} kg CO₂eq")
print(f"   Total                  : {emissions_training + emissions_inference:.4f} kg CO₂eq")

print(f"\nFichiers produits :")
print(f"  → predictions_val.csv   ({len(predictions_val):,} lignes)")
print(f"  → predictions_test.csv  ({len(predictions_test):,} lignes)")
print("=" * 70)
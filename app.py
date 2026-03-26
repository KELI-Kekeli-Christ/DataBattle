import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Storm Alert Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME CSS (identique)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;600;800&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #060a12; color: #e2e8f0; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0d1525; border-right: 1px solid #1e2d45; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #111d30;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 16px !important;
  }
  [data-testid="metric-container"] label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Space Mono', monospace !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Space Mono', monospace !important; font-size: 28px !important; font-weight: 800 !important; }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0d1525; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e2d45; }
  .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #64748b; font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 600; padding: 8px 16px; border: none; }
  .stTabs [aria-selected="true"] { background: rgba(0,212,255,0.09) !important; color: #00d4ff !important; border: 1px solid rgba(0,212,255,0.25) !important; }

  /* Sliders */
  [data-testid="stSlider"] [data-baseweb="slider"] { color: #00d4ff; }

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div { background: #111d30; border: 1px solid #1e2d45; border-radius: 8px; color: #e2e8f0; }

  /* Headers */
  h1, h2, h3 { color: #e2e8f0 !important; }

  /* Dividers */
  hr { border-color: #1e2d45; }

  /* Dataframes */
  [data-testid="stDataFrame"] { background: #111d30; border-radius: 10px; }

  /* Info boxes */
  .info-card {
    background: #111d30;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .accent-bar {
    display: inline-block;
    width: 3px;
    height: 18px;
    background: #00d4ff;
    border-radius: 2px;
    margin-right: 10px;
    vertical-align: middle;
  }
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #e2e8f0;
    margin-bottom: 16px;
  }
  .co2-display {
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(0,212,255,0.03));
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
  }
  .co2-number {
    font-family: 'Space Mono', monospace;
    font-size: 64px;
    font-weight: 800;
    color: #22c55e;
    line-height: 1;
    letter-spacing: -0.04em;
  }
  .actual-banner {
    background: linear-gradient(135deg, rgba(34,197,94,0.07), rgba(0,212,255,0.03));
    border: 1px solid rgba(34,197,94,0.19);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
  }
  .equiv-card {
    background: #111d30;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }
  .ok-badge { background: rgba(34,197,94,0.13); color: #22c55e; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-family: monospace; }
  .nok-badge { background: rgba(239,68,68,0.13); color: #ef4444; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-family: monospace; }
  .opt-badge { background: rgba(0,212,255,0.13); color: #00d4ff; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES PRÉ-CALCULÉES
# ─────────────────────────────────────────────
@st.cache_data
def load_dashboard_data():
    try:
        # Fichiers obligatoires
        predictions_val = pd.read_csv('predictions_val.csv', parse_dates=['prediction_date', 'predicted_date_end_alert'])
        with open('co2_emissions.json') as f:
            co2 = json.load(f)
        cluster_stats = pd.read_csv('cluster_stats.csv')
        theta_results = pd.read_csv('theta_results.csv')
        feature_importance = pd.read_csv('feature_importance.csv')
        confidence_dist = pd.read_csv('confidence_dist.csv')
        valid_snapshots = pd.read_csv('valid_snapshots_dashboard.csv')

        # Fichiers optionnels
        predictions_test = None
        try:
            predictions_test = pd.read_csv('predictions_test.csv', parse_dates=['prediction_date', 'predicted_date_end_alert'])
        except FileNotFoundError:
            pass

        return (predictions_val, co2, cluster_stats, theta_results,
                feature_importance, confidence_dist, valid_snapshots, predictions_test)
    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e.filename}. Veuillez exécuter model.py avec les modifications pour générer tous les fichiers nécessaires.")
        st.stop()

(predictions_val, co2, cluster_stats, theta_results,
 feature_importance, confidence_dist, valid_snapshots, predictions_test) = load_dashboard_data()

# ─────────────────────────────────────────────
# PRÉPARATION DES MÉTRIQUES
# ─────────────────────────────────────────────
# θ optimal selon la contrainte R < 0.02
R_ACCEPT = 0.02
best_theta_row = theta_results[theta_results['Risque R'] < R_ACCEPT].iloc[0] if len(theta_results[theta_results['Risque R'] < R_ACCEPT]) > 0 else theta_results.iloc[0]

# MAE stacking validation (calculée à partir des données de valid_snapshots)
if 'true_remaining_min' in valid_snapshots.columns and 'predicted_remaining_min' in valid_snapshots.columns:
    mae_val = np.mean(np.abs(valid_snapshots['true_remaining_min'] - valid_snapshots['predicted_remaining_min']))
else:
    mae_val = None

# MAE stacking test (si disponible)
mae_test = None
if predictions_test is not None:
    # On suppose que predictions_test contient les colonnes nécessaires, mais pas le vrai remaining.
    # Il faudrait aussi un fichier test_snapshots_dashboard.csv. Sinon on ne peut pas calculer.
    # On laisse mae_test = None pour l'instant.
    pass

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#111d30",
    plot_bgcolor="#111d30",
    font=dict(family="DM Sans, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    legend=dict(bgcolor="#0d1525", bordercolor="#1e2d45", borderwidth=1, font=dict(size=11)),
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", tickfont=dict(color="#64748b", size=11)),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", tickfont=dict(color="#64748b", size=11)),
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Storm Alert\n**Prediction Dashboard**")
    st.markdown("---")
    st.markdown("**Modèle chargé** · Résultats réels")
    st.markdown("*Données pré-calculées*")
    st.markdown("---")

    # Métriques simples (à partir des fichiers chargés)
    st.markdown("**📋 Données d'entraînement**")
    st.metric("Snapshots valid", f"{len(predictions_val):,}")
    st.metric("Alertes valid", f"{predictions_val['airport_alert_id'].nunique():,}")

    st.markdown("---")
    st.markdown("**🌍 Émissions réelles (CodeCarbon)**")
    st.metric("Entraînement", f"{co2['training_kg_co2']:.4f} kg CO₂")
    st.metric("Inférence",    f"{co2['inference_kg_co2']:.4f} kg CO₂")
    st.metric("Total",        f"{co2['training_kg_co2'] + co2['inference_kg_co2']:.4f} kg CO₂")

    st.markdown("---")
    st.caption("Données générées par model.py")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:24px;">
  <div style="width:10px;height:10px;border-radius:50%;background:#22c55e;box-shadow:0 0 10px #22c55e;"></div>
  <span style="font-family:'Space Mono',monospace;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">
    Modèle actif · Prédictions réelles
  </span>
</div>
<h1 style="margin:0 0 4px;font-size:28px;font-weight:800;letter-spacing:-0.02em;">
  ⚡ Storm Alert Prediction
  <span style="color:#00d4ff;">Dashboard</span>
</h1>
<p style="color:#64748b;margin:0 0 24px;font-size:14px;">
  Visualisation des résultats ML · Simulateur d'empreinte carbone
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_results, tab_clusters, tab_theta, tab_carbon = st.tabs([
    "📊  Résultats", "🔵  Clusters", "🎯  Theta optimal", "🌍  Empreinte CO₂"
])

# ═══════════════════════════════════════════
# TAB 1 — RÉSULTATS
# ═══════════════════════════════════════════
with tab_results:
    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("MAE Stacking (val)", f"{mae_val:.2f} min" if mae_val is not None else "N/A")
    k2.metric("MAE Stacking (test)", f"{mae_test:.2f} min" if mae_test is not None else "N/A")
    k3.metric("Gain validation", f"{best_theta_row['Gain (h)']:.1f} h", delta=f"θ = {best_theta_row['θ']:.2f}")
    k4.metric("Risque R", f"{best_theta_row['Risque R']*100:.2f} %", delta="< 2.0% ✓" if best_theta_row['Risque R'] < R_ACCEPT else "≥ 2.0% ✗")
    k5.metric("Snapshots val.", f"{len(predictions_val):,}", delta="alertes aéroports")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Importance des features
    with col_a:
        st.markdown('<div class="section-title"><span class="accent-bar"></span>Top 10 · Importance des features</div>', unsafe_allow_html=True)
        top10 = feature_importance.head(10)
        fig_fi = go.Figure(go.Bar(
            x=top10["Importance"],
            y=top10["Feature"],
            orientation="h",
            marker=dict(
                color=top10["Importance"],
                colorscale=[[0, "#1e2d45"], [0.5, "#00d4ff"], [1, "#7c3aed"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in top10["Importance"]],
            textposition="outside",
            textfont=dict(color="#00d4ff", size=11, family="Space Mono"),
        ))
        fig_fi.update_layout(**PLOT_LAYOUT, height=320, xaxis_title="Importance (gain)")
        fig_fi.update_xaxes(gridcolor="#1e2d45", tickformat=".3f")
        fig_fi.update_yaxes(tickfont=dict(family="Space Mono", size=11, color="#e2e8f0"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # Distribution des confiances
    with col_b:
        st.markdown('<div class="section-title"><span class="accent-bar"></span>Distribution scores de confiance</div>', unsafe_allow_html=True)
        bar_colors = ["#64748b" if i < 3 else "#00d4ff" if i < 6 else "#22c55e" for i in range(10)]
        fig_conf = go.Figure(go.Bar(
            x=confidence_dist["Plage"], y=confidence_dist["Snapshots"],
            marker_color=bar_colors, marker_line_width=0,
            text=confidence_dist["Snapshots"], textposition="outside",
            textfont=dict(color="#64748b", size=10),
        ))
        fig_conf.update_layout(**PLOT_LAYOUT, height=320, yaxis_title="Nombre de snapshots")
        fig_conf.update_xaxes(showgrid=False, tickangle=-30)
        st.plotly_chart(fig_conf, use_container_width=True)

    # Graphique d'erreur (résidus) à partir de valid_snapshots
    if mae_val is not None:
        st.markdown('<div class="section-title"><span class="accent-bar"></span>Erreur de prédiction (validation)</div>', unsafe_allow_html=True)
        valid_snapshots['error'] = valid_snapshots['predicted_remaining_min'] - valid_snapshots['true_remaining_min']
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=valid_snapshots['error'],
            nbinsx=50,
            marker_color="#00d4ff",
            opacity=0.7
        ))
        fig_err.add_vline(x=0, line_dash="dash", line_color="#22c55e")
        fig_err.update_layout(
            **PLOT_LAYOUT,
            xaxis_title="Erreur (prédit - réel) [min]",
            yaxis_title="Nombre de snapshots",
            height=300
        )
        st.plotly_chart(fig_err, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — CLUSTERS
# ═══════════════════════════════════════════
with tab_clusters:
    CLUSTER_COLORS = ["#00d4ff", "#7c3aed", "#22c55e", "#f97316", "#ef4444"]

    # Cartes des clusters
    cols = st.columns(5)
    for i, (_, row) in enumerate(cluster_stats.iterrows()):
        with cols[i]:
            c = CLUSTER_COLORS[i]
            st.markdown(f"""
            <div style="background:{c}0d;border:1px solid {c}40;border-radius:12px;padding:16px;margin-bottom:12px;">
              <div style="font-family:'Space Mono',monospace;font-size:10px;color:{c};text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:6px;">
                CLUSTER {int(row['cluster'])}
              </div>
              <div style="font-size:13px;color:#e2e8f0;font-weight:600;margin-bottom:10px;">{row['Label']}</div>
              <div style="font-family:'Space Mono',monospace;font-size:11px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                  <span style="color:#64748b;">Médiane</span><span style="color:{c};font-weight:700;">{row['Médiane']:.1f} min</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                  <span style="color:#64748b;">Moyenne</span><span style="color:{c};font-weight:700;">{row['Moyenne']:.1f} min</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                  <span style="color:#64748b;">Std</span><span style="color:{c};font-weight:700;">{row['Écart-type']:.1f} min</span>
                </div>
                <div style="background:#1e2d45;border-radius:4px;padding:2px 6px;text-align:center;font-size:10px;color:#64748b;">
                  {int(row['Count']):,} alertes
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title"><span class="accent-bar"></span>Durée médiane par cluster</div>', unsafe_allow_html=True)
        fig_cl = go.Figure(go.Bar(
            x=cluster_stats["Médiane"], y=cluster_stats["Label"],
            orientation="h",
            marker_color=CLUSTER_COLORS,
            text=[f"{v:.1f} min" for v in cluster_stats["Médiane"]],
            textposition="outside",
            textfont=dict(family="Space Mono", size=11, color="#e2e8f0"),
            error_x=dict(type="data", array=cluster_stats["Écart-type"], color="#1e2d45", thickness=2),
        ))
        fig_cl.update_layout(**PLOT_LAYOUT, height=320, xaxis_title="Durée médiane (min)")
        fig_cl.update_yaxes(tickfont=dict(family="Space Mono", size=11))
        st.plotly_chart(fig_cl, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title"><span class="accent-bar"></span>Volume d’alertes par cluster</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=cluster_stats["Label"],
            values=cluster_stats["Count"],
            hole=0.55,
            marker=dict(colors=CLUSTER_COLORS, line=dict(color="#060a12", width=3)),
            textfont=dict(family="Space Mono", size=11, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>%{value:,} alertes<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(**PLOT_LAYOUT, height=320,
                              annotations=[dict(text="Clusters", x=0.5, y=0.5, showarrow=False,
                                                font=dict(size=14, color="#64748b", family="Space Mono"))])
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title"><span class="accent-bar"></span>Tableau récapitulatif</div>', unsafe_allow_html=True)
    st.dataframe(
        cluster_stats.style.background_gradient(subset=["Médiane", "Moyenne"], cmap="Blues")
                           .format({"Médiane": "{:.1f} min", "Moyenne": "{:.1f} min", "Écart-type": "{:.1f} min", "Count": "{:,}"}),
        use_container_width=True, hide_index=True,
    )

# ═══════════════════════════════════════════
# TAB 3 — THETA
# ═══════════════════════════════════════════
with tab_theta:
    best = theta_results[theta_results["Risque R"] < R_ACCEPT].iloc[0] if len(theta_results[theta_results["Risque R"] < R_ACCEPT]) > 0 else theta_results.iloc[0]
    b1, b2, b3 = st.columns(3)
    b1.metric("θ optimal retenu",  f"{best['θ']:.2f}",         delta="Seuil de confiance")
    b2.metric("Gain validation",   f"{best['Gain (h)']:.1f} h", delta=f"{int(best['Gain (h)']*60)} minutes économisées")
    b3.metric("Risque R",          f"{best['Risque R']*100:.2f} %", delta="< 2.0% ✓" if best['Risque R'] < R_ACCEPT else "≥ 2.0% ✗", delta_color="off")

    st.markdown("---")
    col_g, col_r2 = st.columns(2)

    with col_g:
        st.markdown('<div class="section-title"><span class="accent-bar"></span>Gain (heures) selon θ</div>', unsafe_allow_html=True)
        fig_gain = go.Figure()
        fig_gain.add_trace(go.Scatter(
            x=theta_results["θ"], y=theta_results["Gain (h)"],
            mode="lines+markers",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=8, color="#22c55e", line=dict(color="#060a12", width=2)),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.07)",
            name="Gain (h)",
        ))
        fig_gain.add_vline(x=best["θ"], line_dash="dash", line_color="#00d4ff", annotation_text=f"θ opt = {best['θ']}", annotation_font=dict(color="#00d4ff", size=12))
        fig_gain.update_layout(**PLOT_LAYOUT, height=300, xaxis_title="θ (seuil de confiance)", yaxis_title="Gain (heures)")
        st.plotly_chart(fig_gain, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title"><span class="accent-bar"></span>Risque R selon θ</div>', unsafe_allow_html=True)
        fig_risk = go.Figure()
        fig_risk.add_hline(y=0.02, line_dash="dash", line_color="#ef4444",
                           annotation_text="Limite R = 2%", annotation_font=dict(color="#ef4444", size=12),
                           annotation_position="top left")
        colors_risk = ["#ef4444" if r >= 0.02 else "#22c55e" for r in theta_results["Risque R"]]
        fig_risk.add_trace(go.Scatter(
            x=theta_results["θ"], y=theta_results["Risque R"] * 100,
            mode="lines+markers",
            line=dict(color="#f97316", width=3),
            marker=dict(size=8, color=colors_risk, line=dict(color="#060a12", width=2)),
            name="Risque R (%)",
        ))
        fig_risk.add_vline(x=best["θ"], line_dash="dash", line_color="#00d4ff")
        fig_risk.update_layout(**PLOT_LAYOUT, height=300, xaxis_title="θ (seuil de confiance)", yaxis_title="Risque R (%)")
        st.plotly_chart(fig_risk, use_container_width=True)

    # Table
    st.markdown('<div class="section-title"><span class="accent-bar"></span>Tableau complet θ → Gain / Risque</div>', unsafe_allow_html=True)
    display_df = theta_results.copy()
    display_df["Contrainte R < 2%"] = display_df["Risque R"].apply(lambda r: "✓ Respectée" if r < 0.02 else "✗ Mauvais")
    display_df["Sélection"]         = display_df["θ"].apply(lambda t: "★ Optimal" if t == best["θ"] else "")
    display_df["Risque R %"]        = (display_df["Risque R"] * 100).round(2).astype(str) + "%"
    display_df = display_df.drop(columns=["Risque R"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# TAB 4 — EMPREINTE CO₂ (simulateur inchangé)
# ═══════════════════════════════════════════
with tab_carbon:
    # Affichage des émissions réelles
    train_co2 = co2['training_kg_co2']
    infer_co2 = co2['inference_kg_co2']
    total_co2 = train_co2 + infer_co2

    st.markdown(f"""
    <div class="actual-banner">
      <div style="font-family:'Space Mono',monospace;font-size:11px;color:#22c55e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
        🌿 Émissions réelles mesurées (CodeCarbon · co2_emissions.json)
      </div>
      <div style="display:flex;gap:40px;align-items:center;">
        <div>
          <span style="color:#64748b;font-size:12px;">Entraînement</span><br>
          <span style="font-family:'Space Mono',monospace;font-size:22px;font-weight:800;color:#00d4ff;">{train_co2:.4f} kg CO₂</span>
        </div>
        <div style="color:#1e2d45;font-size:24px;">·</div>
        <div>
          <span style="color:#64748b;font-size:12px;">Inférence</span><br>
          <span style="font-family:'Space Mono',monospace;font-size:22px;font-weight:800;color:#7c3aed;">{infer_co2:.4f} kg CO₂</span>
        </div>
        <div style="color:#1e2d45;font-size:24px;">·</div>
        <div>
          <span style="color:#64748b;font-size:12px;">Total</span><br>
          <span style="font-family:'Space Mono',monospace;font-size:22px;font-weight:800;color:#eab308;">{total_co2:.4f} kg CO₂</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Simulateur GPU — Estimez vos émissions")

    # GPU Database et simulateur (inchangé)
    GPU_DB = {
        "NVIDIA A100 (80GB)": {"tdp": 400, "cost_kwh": 0.12, "tier": "datacenter"},
        "NVIDIA H100 (80GB)": {"tdp": 700, "cost_kwh": 0.12, "tier": "datacenter"},
        "NVIDIA V100 (32GB)": {"tdp": 300, "cost_kwh": 0.12, "tier": "datacenter"},
        "NVIDIA RTX 4090":    {"tdp": 450, "cost_kwh": 0.18, "tier": "consumer"},
        "NVIDIA RTX 3090":    {"tdp": 350, "cost_kwh": 0.18, "tier": "consumer"},
        "NVIDIA T4 (cloud)":  {"tdp":  70, "cost_kwh": 0.10, "tier": "cloud"},
        "AMD MI250":          {"tdp": 500, "cost_kwh": 0.12, "tier": "datacenter"},
        "AMD RX 7900 XTX":    {"tdp": 355, "cost_kwh": 0.18, "tier": "consumer"},
        "Google TPU v4":      {"tdp": 170, "cost_kwh": 0.09, "tier": "cloud"},
        "CPU (multi-cœurs)":  {"tdp": 180, "cost_kwh": 0.15, "tier": "consumer"},
    }
    REGION_FACTORS = {
        "France 🇫🇷":           0.052,
        "Allemagne 🇩🇪":         0.364,
        "USA (Californie) 🇺🇸": 0.197,
        "USA (Moyenne) 🇺🇸":    0.397,
        "Chine 🇨🇳":             0.581,
        "Norvège 🇳🇴":           0.028,
        "Inde 🇮🇳":              0.708,
        "Australie 🇦🇺":         0.490,
        "Canada 🇨🇦":            0.159,
    }

    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        selected_gpu    = st.selectbox("🖥️ GPU", list(GPU_DB.keys()), index=0)
        selected_region = st.selectbox("🌍 Région / mix électrique", list(REGION_FACTORS.keys()), index=0)
        num_gpus        = st.slider("Nombre de GPU",    1, 16, 1)
        utilization     = st.slider("Utilisation GPU (%)", 10, 100, 85, 5)
        train_hours     = st.slider("Durée entraînement (h)", 0.1, 24.0, 1.2, 0.1)
        infer_hours     = st.slider("Durée inférence (h)",    0.05, 8.0, 0.3, 0.05)

        gpu = GPU_DB[selected_gpu]
        region_factor = REGION_FACTORS[selected_region]
        effective_tdp = gpu["tdp"] * num_gpus * utilization / 100 / 1000  # kW

        train_energy = effective_tdp * train_hours
        infer_energy = effective_tdp * infer_hours
        train_co2_sim = train_energy * region_factor
        infer_co2_sim = infer_energy * region_factor
        total_co2_sim = train_co2_sim + infer_co2_sim
        total_cost   = (train_energy + infer_energy) * gpu["cost_kwh"]

        st.markdown(f"""
        <div style="background:#0d1525;border:1px solid #1e2d45;border-radius:10px;padding:14px;font-family:'Space Mono',monospace;font-size:12px;margin-top:8px;">
          <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Specs GPU</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#64748b;">TDP</span><span style="color:#e2e8f0;">{gpu['tdp']}W × {num_gpus}</span></div>
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#64748b;">Puissance eff.</span><span style="color:#e2e8f0;">{effective_tdp*1000:.0f}W</span></div>
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#64748b;">Facteur CO₂</span><span style="color:#e2e8f0;">{region_factor} kg/kWh</span></div>
          <div style="display:flex;justify-content:space-between;"><span style="color:#64748b;">Tier</span><span style="color:#00d4ff;">{gpu['tier']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with ctrl2:
        st.markdown(f"""
        <div class="co2-display">
          <div style="font-family:'Space Mono',monospace;font-size:11px;color:#22c55e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
            Émissions CO₂ simulées · Total
          </div>
          <div class="co2-number">{total_co2_sim*1000:.2f}<span style="font-size:22px;color:rgba(34,197,94,0.6);margin-left:8px;">g CO₂eq</span></div>
          <div style="font-size:16px;color:#64748b;margin-top:10px;">
            soit <strong style="color:#eab308;">{total_co2_sim:.4f} kg</strong> —
            Coût électricité : <strong style="color:#00d4ff;">{total_cost:.4f} €</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        b_col1, b_col2 = st.columns(2)
        pct_train = train_co2_sim / (total_co2_sim + 1e-10) * 100
        pct_infer = infer_co2_sim / (total_co2_sim + 1e-10) * 100
        with b_col1:
            st.metric("🏋️ Entraînement", f"{train_co2_sim*1000:.2f} g CO₂",
                      delta=f"{train_energy:.3f} kWh · {train_hours}h · {pct_train:.0f}% du total")
        with b_col2:
            st.metric("⚡ Inférence", f"{infer_co2_sim*1000:.2f} g CO₂",
                      delta=f"{infer_energy:.3f} kWh · {infer_hours}h · {pct_infer:.0f}% du total")

        fig_stack = go.Figure()
        fig_stack.add_bar(name="Entraînement", x=["Répartition"], y=[train_co2_sim * 1000],
                          marker_color="#00d4ff", text=f"{train_co2_sim*1000:.2f}g", textposition="inside",
                          textfont=dict(color="white", size=12, family="Space Mono"))
        fig_stack.add_bar(name="Inférence",    x=["Répartition"], y=[infer_co2_sim * 1000],
                          marker_color="#7c3aed", text=f"{infer_co2_sim*1000:.2f}g", textposition="inside",
                          textfont=dict(color="white", size=12, family="Space Mono"))
        fig_stack.update_layout(**PLOT_LAYOUT, barmode="stack", height=100, showlegend=True)
        fig_stack.update_layout(margin=dict(l=8, r=8, t=8, b=8))
        fig_stack.update_xaxes(showgrid=False, showticklabels=False)
        fig_stack.update_yaxes(showgrid=False, showticklabels=False)
        st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")
    # Équivalences
    st.markdown('<div class="section-title"><span class="accent-bar"></span>Équivalences</div>', unsafe_allow_html=True)
    equivs = [
        ("🚗", "km en voiture",        total_co2_sim / 0.21),
        ("✈️", "km en avion",          total_co2_sim / 0.255),
        ("🥩", "repas avec bœuf",      total_co2_sim / 6.6),
        ("📺", "h de streaming",       total_co2_sim / 0.036),
        ("🌳", "arbres·an pour absorber", total_co2_sim / 21),
        ("💡", "h d'ampoule LED",      total_co2_sim / 0.009),
        ("📱", "charges smartphone",   total_co2_sim / 0.008),
    ]
    eq_cols = st.columns(7)
    for i, (icon, label, value) in enumerate(equivs):
        with eq_cols[i]:
            fmt = f"{value:.2f}" if value < 10 else f"{value:.0f}"
            st.markdown(f"""
            <div class="equiv-card">
              <div style="font-size:28px;margin-bottom:6px;">{icon}</div>
              <div style="font-family:'Space Mono',monospace;font-size:18px;font-weight:800;color:#e2e8f0;">{fmt}</div>
              <div style="font-size:10px;color:#64748b;margin-top:4px;line-height:1.3;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    # GPU Comparison chart
    st.markdown('<div class="section-title"><span class="accent-bar"></span>Comparaison GPU · Émissions pour la même durée & même région</div>', unsafe_allow_html=True)
    gpu_compare = []
    for gname, gdata in GPU_DB.items():
        eff = gdata["tdp"] * num_gpus * utilization / 100 / 1000
        total = eff * (train_hours + infer_hours) * region_factor * 1000
        gpu_compare.append({"GPU": gname, "CO₂ (g)": round(total, 2), "active": gname == selected_gpu})

    gpu_df = pd.DataFrame(gpu_compare).sort_values("CO₂ (g)")
    fig_gpu = go.Figure(go.Bar(
        x=gpu_df["CO₂ (g)"], y=gpu_df["GPU"],
        orientation="h",
        marker=dict(
            color=["#22c55e" if row["active"] else "#1e2d45" for _, row in gpu_df.iterrows()],
            line=dict(
                color=["#22c55e" if row["active"] else "#1e2d45" for _, row in gpu_df.iterrows()],
                width=1
            ),
            opacity=[1.0 if row["active"] else 0.45 for _, row in gpu_df.iterrows()]
        ),
        text=[f"{v:.2f}g" for v in gpu_df["CO₂ (g)"]],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11, color="#e2e8f0"),
    ))
    fig_gpu.update_layout(**PLOT_LAYOUT, height=360, xaxis_title="CO₂ total (g)",
                          title=dict(text=f"GPU sélectionné : {selected_gpu}", font=dict(size=12, color="#00d4ff")))
    fig_gpu.update_yaxes(tickfont=dict(family="Space Mono", size=11))
    st.plotly_chart(fig_gpu, use_container_width=True)

    # Region comparison
    st.markdown('<div class="section-title"><span class="accent-bar"></span>Impact de la région sur les émissions</div>', unsafe_allow_html=True)
    reg_data = []
    for rname, rfactor in REGION_FACTORS.items():
        eff = gpu["tdp"] * num_gpus * utilization / 100 / 1000
        total = eff * (train_hours + infer_hours) * rfactor * 1000
        reg_data.append({"Région": rname, "CO₂ (g)": round(total, 2), "active": rname == selected_region})
    reg_df = pd.DataFrame(reg_data).sort_values("CO₂ (g)")

    fig_reg = go.Figure(go.Bar(
        x=reg_df["Région"], y=reg_df["CO₂ (g)"],
        marker=dict(
            color=["#22c55e" if row["active"] else "#1e2d45" for _, row in reg_df.iterrows()],
            opacity=[1.0 if row["active"] else 0.5 for _, row in reg_df.iterrows()],
            line_width=0
        ),
        text=[f"{v:.1f}g" for v in reg_df["CO₂ (g)"]],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11, color="#e2e8f0"),
    ))
    fig_reg.update_layout(**PLOT_LAYOUT, height=280, yaxis_title="CO₂ total (g)")
    fig_reg.update_xaxes(showgrid=False)
    st.plotly_chart(fig_reg, use_container_width=True)
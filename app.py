"""
app.py — Immobilien Preisrechner · Rheinland-Pfalz

Tabs:
  🔮 Preis schätzen  — property inputs → price estimate + SHAP explanation
  📊 Daten erkunden  — EDA charts
  🗺️  Karte           — geographic price heatmap (RP)
  🤖 Modell-Insights — metrics + feature importance

Usage:
    streamlit run app.py
"""

import json
import sqlite3
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Immobilien Preisrechner · RLP",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .price-card {
    background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
    border-radius: 16px; padding: 2rem 1.5rem; color: white;
    text-align: center; margin: 1rem 0 1.5rem 0;
    box-shadow: 0 4px 20px rgba(45,106,79,0.35);
  }
  .price-label { font-size: .85rem; opacity: .85; text-transform: uppercase; letter-spacing: .12em; }
  .price-value { font-size: 3.2rem; font-weight: 800; margin: .4rem 0; }
  .price-range { font-size: .88rem; opacity: .75; }
  .sec-header  {
    font-size: .78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: #2d6a4f; margin: 1.2rem 0 .4rem 0;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"]      { height: 46px; padding: 0 22px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH           = Path("data/housing.db")
MODEL_PATH        = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
METRICS_PATH      = Path("models/metrics.json")

TARGET = "obj_purchasePrice"

# ── Feature config (must match train.py) ──────────────────────────────────────
NUMERIC_FEATURES     = ["obj_livingSpace", "obj_noRooms", "obj_lotArea",
                         "obj_yearConstructed", "obj_noParkSpaces",
                         "obj_numberOfFloors", "obj_thermalChar",
                         "obj_pricetrendbuy", "geo_lat", "geo_lng"]
BINARY_FEATURES      = ["obj_newlyConst", "obj_cellar", "obj_barrierFree", "obj_rented"]
ORDERED_CAT_FEATURES = ["obj_condition", "obj_interiorQual"]
ONEHOT_FEATURES      = ["obj_heatingType", "obj_buildingType",
                         "obj_firingTypes", "obj_constructionPhase"]
ALL_FEATURES         = NUMERIC_FEATURES + BINARY_FEATURES + ORDERED_CAT_FEATURES + ONEHOT_FEATURES

FEATURE_LABELS = {
    "obj_livingSpace":    "Wohnfläche (m²)",
    "obj_noRooms":        "Zimmeranzahl",
    "obj_lotArea":        "Grundstücksfläche (m²)",
    "obj_yearConstructed":"Baujahr",
    "obj_noParkSpaces":   "Stellplätze",
    "obj_numberOfFloors": "Etagen gesamt",
    "obj_thermalChar":    "Energieverbrauch (kWh/m²a)",
    "obj_pricetrendbuy":  "Kaufpreistrend (%)",
    "geo_lat": "Breitengrad", "geo_lng": "Längengrad",
    "obj_newlyConst":     "Neubau",
    "obj_cellar":         "Keller",
    "obj_barrierFree":    "Barrierefrei",
    "obj_rented":         "Vermietet",
    "obj_condition":      "Zustand",
    "obj_interiorQual":   "Ausstattung",
    "obj_heatingType":    "Heizungsart",
    "obj_buildingType":   "Gebäudetyp",
    "obj_firingTypes":    "Befeuerungsart",
    "obj_constructionPhase": "Bauphase",
}

RP_LAT, RP_LNG, RP_ZOOM = 49.95, 7.45, 8


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Daten werden geladen…")
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error("⚠️ Datenbank nicht gefunden — bitte `python setup_db.py` ausführen.")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM housing", conn)
    conn.close()
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    for col in ORDERED_CAT_FEATURES + ONEHOT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("no_information").astype(str)
    # Force coordinates to numeric (zip lookup may have inserted "na" strings)
    for col in ["geo_lat", "geo_lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource(show_spinner="Modell wird geladen…")
def load_model():
    if not MODEL_PATH.exists():
        st.error("⚠️ Modell nicht gefunden — bitte `python train.py` ausführen.")
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


def predict(model, preprocessor, inp: pd.DataFrame) -> float:
    X = preprocessor.transform(inp[ALL_FEATURES])
    return float(np.expm1(model.predict(X)[0]))


def col_uniq(df, col, fallback):
    vals = sorted(df[col].dropna().unique().tolist()) if col in df.columns else []
    return vals or fallback


# ── Load ───────────────────────────────────────────────────────────────────────
df            = load_data()
model, scaler = load_model()
metrics       = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

CONDITIONS     = col_uniq(df, "obj_condition",
                           ["well_kept","modernized","need_of_renovation",
                            "refurbished","first_time_use","mint_condition","no_information"])
INTERIORQUALS  = col_uniq(df, "obj_interiorQual",
                           ["simple","normal","sophisticated","luxury","no_information"])
HEATINGTYPES   = col_uniq(df, "obj_heatingType",
                           ["central_heating","gas_heating","oil_heating",
                            "district_heating","heat_pump","no_information"])
BUILDINGTYPES  = col_uniq(df, "obj_buildingType",
                           ["single_family_house","multi_family_house","detached_house",
                            "semi_detached_house","terraced_house","no_information"])
FIRINGTYPES    = col_uniq(df, "obj_firingTypes",
                           ["gas","oil","district_heating","heat_pump","electricity","no_information"])
CONSTR_PHASES  = col_uniq(df, "obj_constructionPhase",
                           ["completed","projected","under_construction","no_information"])

# ── Header ─────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([4, 1])
with h1:
    st.title("🏠 Immobilien Preisrechner")
    st.caption("Rheinland-Pfalz · ImmobilienScout24-Daten · XGBoost + MLflow")
with h2:
    if metrics:
        st.metric("Modell R²", f"{metrics.get('r2', 0):.3f}")

tab_pred, tab_exp, tab_map, tab_mod = st.tabs([
    "🔮  Preis schätzen",
    "📊  Daten erkunden",
    "🗺️   Karte",
    "🤖  Modell-Insights",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<p class="sec-header">📐 Fläche & Zimmer</p>', unsafe_allow_html=True)
        living_space  = st.slider("Wohnfläche (m²)", 20, 500, 130, 5)
        no_rooms      = st.selectbox("Zimmeranzahl",
                                     [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0], index=5)
        lot_area      = st.slider("Grundstücksfläche (m²)", 0, 5_000, 600, 50)
        no_floors     = st.number_input("Etagen gesamt", 1, 10, 2)
        no_park       = st.selectbox("Stellplätze", [0, 1, 2, 3], index=1)

    with c2:
        st.markdown('<p class="sec-header">🏡 Gebäude</p>', unsafe_allow_html=True)
        year_built    = st.slider("Baujahr", 1880, 2024, 1985)
        building_type = st.selectbox("Gebäudetyp",    BUILDINGTYPES)
        constr_phase  = st.selectbox("Bauphase",       CONSTR_PHASES)
        heating_type  = st.selectbox("Heizungsart",    HEATINGTYPES)
        firing_type   = st.selectbox("Befeuerungsart", FIRINGTYPES)
        thermal_char  = st.slider("Energieverbrauch (kWh/m²a)", 0, 400, 130, 5,
                                   help="A+ <30 · B ~75 · D ~130 · H >250")

    with c3:
        st.markdown('<p class="sec-header">⭐ Qualität & Lage</p>', unsafe_allow_html=True)
        condition     = st.selectbox("Zustand",    CONDITIONS)
        interior_qual = st.selectbox("Ausstattung", INTERIORQUALS)
        newly_const   = st.checkbox("Neubau")
        cellar        = st.checkbox("Keller")
        barrier_free  = st.checkbox("Barrierefrei")
        rented        = st.checkbox("Derzeit vermietet")
        price_trend   = st.slider("Kaufpreistrend (%)", -5.0, 20.0, 7.5, 0.1,
                                   help="Lokaler jährlicher Preisanstieg")
        geo_lat = st.number_input("Breitengrad", 49.0, 51.0, RP_LAT, 0.01, format="%.3f")
        geo_lng = st.number_input("Längengrad",   6.0,  8.5, RP_LNG, 0.01, format="%.3f")

    st.divider()

    inp = pd.DataFrame([{
        "obj_livingSpace":    living_space,
        "obj_noRooms":        no_rooms,
        "obj_lotArea":        lot_area,
        "obj_yearConstructed": year_built,
        "obj_noParkSpaces":   no_park,
        "obj_numberOfFloors": no_floors,
        "obj_thermalChar":    thermal_char,
        "obj_pricetrendbuy":  price_trend,
        "geo_lat":            geo_lat,
        "geo_lng":            geo_lng,
        "obj_newlyConst":     int(newly_const),
        "obj_cellar":         int(cellar),
        "obj_barrierFree":    int(barrier_free),
        "obj_rented":         int(rented),
        "obj_condition":      condition,
        "obj_interiorQual":   interior_qual,
        "obj_heatingType":    heating_type,
        "obj_buildingType":   building_type,
        "obj_firingTypes":    firing_type,
        "obj_constructionPhase": constr_phase,
    }])

    pred_price   = predict(model, scaler, inp)
    median_price = df[TARGET].median()
    delta_pct    = (pred_price - median_price) / median_price * 100
    per_m2       = pred_price / living_space
    low, high    = pred_price * 0.88, pred_price * 1.12

    st.markdown(f"""
    <div class="price-card">
      <div class="price-label">Geschätzter Kaufpreis</div>
      <div class="price-value">€{pred_price:,.0f}</div>
      <div class="price-range">Wahrscheinliche Spanne &nbsp;·&nbsp; €{low:,.0f} – €{high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("vs. Median RLP",  f"€{pred_price:,.0f}",  f"{delta_pct:+.1f}%")
    m2.metric("Preis pro m²",    f"€{per_m2:,.0f}")
    m3.metric("Wohnfläche",      f"{living_space} m²")
    m4.metric("Zimmer",          f"{no_rooms}")

    # SHAP
    with st.expander("🔍 Warum dieser Preis? (Erklärung)"):
        try:
            import shap
            bg = scaler.transform(
                df[ALL_FEATURES].fillna(0).sample(min(200, len(df)), random_state=42)
            )
            explainer = shap.TreeExplainer(model, bg)
            sv        = explainer.shap_values(scaler.transform(inp[ALL_FEATURES]))

            try:
                ohe_cols = scaler.named_transformers_["ohe"] \
                               .get_feature_names_out(ONEHOT_FEATURES).tolist()
            except Exception:
                ohe_cols = []

            feat_names = NUMERIC_FEATURES + BINARY_FEATURES + ORDERED_CAT_FEATURES + ohe_cols
            n = min(len(feat_names), len(sv[0]))
            shap_df = (
                pd.DataFrame({"feature": feat_names[:n], "shap": sv[0][:n]})
                .assign(abs_val=lambda x: x["shap"].abs())
                .sort_values("abs_val").tail(10)
            )
            shap_df["feature"] = shap_df["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_df["shap"]]

            fig_s = go.Figure(go.Bar(x=shap_df["shap"], y=shap_df["feature"],
                                     orientation="h", marker_color=colors))
            fig_s.update_layout(title="Top 10 Einflussfaktoren",
                                 xaxis_title="Einfluss (SHAP)", height=360,
                                 margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption("🔴 Rot = treibt Preis hoch · 🔵 Blau = senkt Preis")
        except ImportError:
            st.info("`pip install shap` für Erklärungen installieren.")
        except Exception as e:
            st.warning(f"SHAP nicht verfügbar: {e}")

    # Comparables
    with st.expander("🏘️ Vergleichbare Immobilien"):
        comps = df[df[TARGET].between(low * 0.85, high * 1.15)].copy()
        if "obj_noRooms" in df.columns:
            comps = comps[comps["obj_noRooms"].between(no_rooms - 0.5, no_rooms + 0.5)]
        show_cols = [c for c in [TARGET, "obj_livingSpace", "obj_noRooms",
                                  "obj_yearConstructed", "obj_condition",
                                  "obj_regio2", "obj_regio3"] if c in comps.columns]
        if len(comps) >= 3:
            st.dataframe(
                comps.sample(min(8, len(comps)), random_state=7)[show_cols]
                     .sort_values(TARGET)
                     .style.format({TARGET: "€{:,.0f}", "obj_livingSpace": "{:.0f} m²"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Keine ähnlichen Immobilien gefunden — Zimmeranzahl anpassen.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    df_p = df[df[TARGET].between(50_000, 2_000_000)]

    r1a, r1b = st.columns(2)
    with r1a:
        fig1 = px.histogram(df_p, x=TARGET, nbins=60,
                            title="Kaufpreisverteilung",
                            color_discrete_sequence=["#2d6a4f"],
                            labels={TARGET: "Kaufpreis (€)"})
        fig1.update_layout(showlegend=False,
                           xaxis=dict(tickprefix="€", tickformat=",.0f"))
        st.plotly_chart(fig1, use_container_width=True)

    with r1b:
        if "obj_livingSpace" in df_p.columns:
            fig2 = px.scatter(
                df_p.sample(min(3_000, len(df_p)), random_state=1),
                x="obj_livingSpace", y=TARGET,
                color="obj_noRooms" if "obj_noRooms" in df_p.columns else None,
                title="Preis vs. Wohnfläche (Farbe = Zimmer)",
                color_continuous_scale="Greens", opacity=0.5,
                labels={"obj_livingSpace": "Wohnfläche (m²)",
                        TARGET: "Preis (€)", "obj_noRooms": "Zimmer"},
            )
            fig2.update_layout(yaxis=dict(tickprefix="€", tickformat=",.0f"))
            st.plotly_chart(fig2, use_container_width=True)

    r2a, r2b = st.columns(2)
    with r2a:
        if "obj_buildingType" in df_p.columns:
            bt = (df_p.groupby("obj_buildingType")[TARGET].median()
                      .reset_index().rename(columns={TARGET: "median_price"})
                      .sort_values("median_price", ascending=False).head(8))
            fig3 = px.bar(bt, x="obj_buildingType", y="median_price",
                           title="Median-Preis nach Gebäudetyp",
                           color="median_price", color_continuous_scale="Greens",
                           labels={"median_price": "Median-Preis (€)",
                                   "obj_buildingType": "Gebäudetyp"})
            fig3.update_layout(showlegend=False,
                               yaxis=dict(tickprefix="€", tickformat=",.0f"))
            st.plotly_chart(fig3, use_container_width=True)

    with r2b:
        if "obj_condition" in df_p.columns:
            fig4 = px.box(
                df_p[df_p["obj_condition"] != "no_information"],
                x="obj_condition", y=TARGET,
                title="Preis nach Zustand",
                color="obj_condition",
                color_discrete_sequence=px.colors.sequential.Greens_r,
                labels={TARGET: "Preis (€)", "obj_condition": "Zustand"},
            )
            fig4.update_layout(showlegend=False,
                               yaxis=dict(tickprefix="€", tickformat=",.0f"))
            st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    corr_cols = [c for c in [TARGET, "obj_livingSpace", "obj_noRooms",
                               "obj_yearConstructed", "obj_lotArea",
                               "obj_noParkSpaces", "obj_thermalChar",
                               "obj_pricetrendbuy", "obj_cellar"]
                 if c in df.columns]
    fig5 = px.imshow(df[corr_cols].corr(), text_auto=".2f", aspect="auto",
                     title="Korrelationsmatrix", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1)
    fig5.update_layout(height=420)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Geografische Preisverteilung · Rheinland-Pfalz")

    if "geo_lat" in df.columns and df["geo_lat"].notna().sum() > 10:
        price_cap = st.slider(
            "Max. angezeigter Preis (€)",
            int(df[TARGET].quantile(0.05)),
            int(df[TARGET].quantile(0.99)),
            int(df[TARGET].quantile(0.90)),
            step=10_000, format="€%d",
        )

        extra_cols = [c for c in ["obj_noRooms", "obj_livingSpace",
                                   "obj_condition", "obj_regio2"] if c in df.columns]
        map_df = (
            df[df[TARGET] <= price_cap][["geo_lat", "geo_lng", TARGET] + extra_cols]
            .dropna(subset=["geo_lat", "geo_lng"])
            .copy()
        )

        p_min, p_max = map_df[TARGET].min(), map_df[TARGET].max()
        norm = (map_df[TARGET] - p_min) / (p_max - p_min + 1e-9)
        map_df["r"] = (norm * 210).astype(int)
        map_df["g"] = ((1 - norm) * 140 + 40).astype(int)
        map_df["b"] = ((1 - norm) * 60).astype(int)

        layer = pdk.Layer("ScatterplotLayer", data=map_df,
                           get_position="[geo_lng, geo_lat]",
                           get_color="[r, g, b, 180]",
                           get_radius=400, pickable=True)
        view  = pdk.ViewState(
            latitude=map_df["geo_lat"].mean(),
            longitude=map_df["geo_lng"].mean(),
            zoom=RP_ZOOM, pitch=25,
        )

        tip_html = f"<b>Preis:</b> €{{{TARGET}:,}}"
        if "obj_noRooms"    in map_df.columns: tip_html += "<br/><b>Zimmer:</b> {obj_noRooms}"
        if "obj_livingSpace" in map_df.columns: tip_html += " · <b>Fläche:</b> {obj_livingSpace:.0f} m²"
        if "obj_regio2"     in map_df.columns: tip_html += "<br/><b>Landkreis:</b> {obj_regio2}"

        st.pydeck_chart(pdk.Deck(
            layers=[layer], initial_view_state=view,
            tooltip={"html": tip_html,
                     "style": {"backgroundColor": "white", "color": "#333",
                               "fontSize": "13px", "padding": "8px 12px",
                               "borderRadius": "6px"}},
            map_style="mapbox://styles/mapbox/light-v10",
        ))

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Objekte",      f"{len(map_df):,}")
        mc2.metric("Ø Preis",      f"€{map_df[TARGET].mean():,.0f}")
        mc3.metric("Median",       f"€{map_df[TARGET].median():,.0f}")
        mc4.metric("Max. Preis",   f"€{map_df[TARGET].max():,.0f}")
    else:
        st.warning(
            "Keine Koordinaten in der Datenbank. "
            "Stelle sicher, dass `zip_lat_lang.csv` in `data/` liegt und "
            "`python setup_db.py` erneut ausgeführt wird."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_mod:
    st.subheader("Modell-Performance")

    if metrics:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("R²",   f"{metrics.get('r2',   0):.4f}")
        mc2.metric("RMSE", f"€{metrics.get('rmse', 0):,.0f}")
        mc3.metric("MAE",  f"€{metrics.get('mae',  0):,.0f}")
        mc4.metric("MAPE", f"{metrics.get('mape',  0):.2f}%")
    else:
        st.info("Metriken fehlen — `python train.py` ausführen.")

    st.divider()

    try:
        try:
            ohe_cols = scaler.named_transformers_["ohe"] \
                           .get_feature_names_out(ONEHOT_FEATURES).tolist()
        except Exception:
            ohe_cols = []

        feat_names = NUMERIC_FEATURES + BINARY_FEATURES + ORDERED_CAT_FEATURES + ohe_cols
        n = min(len(feat_names), len(model.feature_importances_))
        fi_df = (
            pd.DataFrame({"Merkmal": feat_names[:n],
                          "Wichtigkeit": model.feature_importances_[:n]})
            .sort_values("Wichtigkeit", ascending=True).tail(14)
        )
        fi_df["Merkmal"] = fi_df["Merkmal"].map(lambda x: FEATURE_LABELS.get(x, x))

        fig_fi = px.bar(fi_df, x="Wichtigkeit", y="Merkmal", orientation="h",
                        title="Merkmalswichtigkeit (XGBoost Gain)",
                        color="Wichtigkeit", color_continuous_scale="Greens")
        fig_fi.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature Importance nicht verfügbar: {e}")

    st.divider()
    if st.button("▶ Tatsächlicher vs. geschätzter Preis (500 Objekte)"):
        with st.spinner("Vorhersagen werden berechnet…"):
            s = df.sample(min(500, len(df)), random_state=42).copy()
            for col in NUMERIC_FEATURES:
                if col in s.columns:
                    s[col] = s[col].fillna(s[col].median())
            for col in BINARY_FEATURES:
                s[col] = s.get(col, pd.Series(0, index=s.index)).fillna(0)
            for col in ORDERED_CAT_FEATURES + ONEHOT_FEATURES:
                s[col] = s.get(col, pd.Series("no_information", index=s.index)) \
                           .fillna("no_information")

            y_pred = np.expm1(model.predict(scaler.transform(s[ALL_FEATURES])))
            max_v  = max(s[TARGET].max(), y_pred.max())

            fig_av = px.scatter(x=s[TARGET], y=y_pred, opacity=0.45,
                                color_discrete_sequence=["#2d6a4f"],
                                labels={"x": "Tatsächlicher Preis (€)",
                                        "y": "Geschätzter Preis (€)"},
                                title="Tatsächlich vs. Geschätzt")
            fig_av.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                             line=dict(color="red", dash="dash", width=1.5))
            fig_av.update_layout(
                xaxis=dict(tickprefix="€", tickformat=",.0f"),
                yaxis=dict(tickprefix="€", tickformat=",.0f"),
            )
            st.plotly_chart(fig_av, use_container_width=True)
            st.caption("Punkte auf der roten Linie = perfekte Vorhersage.")
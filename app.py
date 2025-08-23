import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="🌱 Soil Health Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CONSTANTS =========================
CLASS_LABELS_MAP = {0: "Healthy", 1: "Low", 2: "Moderate", 3: "High"}
SIDEBAR_CARDS = [
    dict(key="Temperature", icon="🌡️", unit="°C", min=0, max=50, default=25, bg="#FFE5DD"),
    dict(key="Humidity",    icon="💧", unit="%",  min=0, max=100, default=60, bg="#E6F2FF"),
    dict(key="Moisture",    icon="💦", unit="%",  min=0, max=100, default=50, bg="#E6FFF7"),
    dict(key="Nitrogen",    icon="🌿", unit="mg/kg", min=0, max=100, default=50, bg="#EFFFF2"),
    dict(key="Potassium",   icon="🪨", unit="mg/kg", min=0, max=100, default=35, bg="#F1E8FF"),
    dict(key="Phosphorous", icon="🧪", unit="mg/kg", min=0, max=100, default=20, bg="#FFF6DE"),
]
SDI_BADGE_IMAGE = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAnCxDd4J8kttbCr2WYwqezNwq2pc2DFCrDHyPrHqPlNasn1sIpvMhoF8&s"

# ========================= CSS =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root{
  --brand:#22C55E;
  --muted:#6b7280;
  --card:#ffffff;
  --ring: 0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.06);
}

.stApp { font-family:'Inter',sans-serif; background:linear-gradient(180deg,#EAF7EA, #F4FBF4); }

.main-header{
  padding:18px 24px; background:linear-gradient(180deg,#d9f3dc,#ecf9ed);
  border-radius:16px; margin-bottom:18px; border:1px solid #d7f0d9;
}
.main-header h1{ margin:0; font-weight:800; color:#1f8f3a; letter-spacing:.2px; }
.main-header p{ margin:.35rem 0 0; color:#4d7a52; }

.stTabs [data-baseweb="tab-list"]{
  background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:6px; box-shadow:var(--ring);
}
.stTabs [data-baseweb="tab"]{ height:40px; padding:0 18px; border-radius:8px; color:#64748b; font-weight:600; }
.stTabs [aria-selected="true"]{ background:#e9f7ee; color:#15803d; }

.section-card{
  background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:18px; box-shadow:var(--ring);
}

.sdi-card{
  position:relative; background:linear-gradient(180deg,#FFF5F5,#FFE7E7);
  border:1px solid #ffd4d4; border-radius:14px; padding:20px 20px 16px 20px; box-shadow:var(--ring);
}
.sdi-meter{ background:#111; height:8px; border-radius:6px; }

.badge-pill{ display:inline-block; padding:.45rem .8rem; border-radius:999px; font-weight:700; font-size:.95rem; }
.badge-high{ background:#ffe3e3; color:#c62828; border:1px solid #ffc6c6; }

.sdi-avatar{
  position:absolute; top:24px; right:24px; width:84px; height:84px; border-radius:999px;
  box-shadow:0 6px 18px rgba(0,0,0,.12); overflow:hidden; border:4px solid #fff;
}
.sdi-avatar img{ width:100%; height:100%; object-fit:cover; }

.quick-tile{
  text-align:center; background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow:var(--ring);
}

.sidebar-title{ display:flex; gap:10px; align-items:center; margin:2px 0 12px; }
.sidebar-title h3{ margin:0; font-weight:800; color:#111827; }
.sidebar-help{ margin:0 0 14px; color:#6b7280; font-size:.925rem; }

.param-card{
  border-radius:16px; padding:14px 14px 12px; border:1px solid #eef2f7; box-shadow:var(--ring); margin-bottom:14px;
}
.param-head{ display:flex; align-items:flex-start; justify-content:space-between; }
.param-left{ display:flex; gap:12px; align-items:center; }
.param-icon{ font-size:22px; line-height:22px; }
.param-name{ font-weight:800; color:#111827; }
.param-value{ font-weight:800; color:#111827; }
.param-sub{ color:#6b7280; margin-top:4px; }

.prog-row{ display:flex; justify-content:space-between; color:#9ca3af; font-size:.85rem; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ========================= HEADER =========================
st.markdown("""
<div class="main-header">
  <h1>🌿 Soil Health Analytics</h1>
  <p>Advanced soil degradation monitoring with real-time analysis and actionable insights for sustainable farming</p>
</div>
""", unsafe_allow_html=True)

# ========================= LOAD MODELS =========================
try:
    rf_reg = joblib.load("rf_reg.pkl")
    xgb_reg = joblib.load("xgb_reg.pkl")
    rf_cls = joblib.load("rf_cls.pkl")
    xgb_cls = joblib.load("xgb_cls.pkl")
except Exception as e:
    st.error(f"⚠️ Model files not found or failed to load: {e}")
    st.stop()

# ========================= SIDEBAR (exact look & feel) =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title"><span>🧪</span><h3>Soil Parameters</h3></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-help">Adjust parameters to analyze soil health</p>', unsafe_allow_html=True)

    # Draw each pretty "card" with a native Streamlit slider inside
    values = {}
    for card in SIDEBAR_CARDS:
        st.markdown(
            f'<div class="param-card" style="background:{card["bg"]};">',
            unsafe_allow_html=True
        )
        colL, colR = st.columns([6,1])
        with colL:
            st.markdown(
                f'''
                <div class="param-head">
                  <div class="param-left">
                    <div class="param-icon">{card["icon"]}</div>
                    <div>
                      <div class="param-name">{card["key"]}</div>
                      <div class="param-sub">{card["default"]} {card["unit"]}</div>
                    </div>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        with colR:
            st.markdown(f'<div class="param-value">{card["default"]}</div>', unsafe_allow_html=True)

        # the actual slider
        values[card["key"]] = st.slider(
            label=f'{card["icon"]} {card["key"]} ({card["unit"]})',
            min_value=card["min"],
            max_value=card["max"],
            value=card["default"],
            label_visibility="collapsed",
            key=f'sld_{card["key"]}'
        )
        # min/max row
        st.markdown(
            f'<div class="prog-row"><span>{card["min"]}</span><span>{card["max"]}</span></div></div>',
            unsafe_allow_html=True
        )

# ========================= DATAFRAME FOR MODELS =========================
input_df = pd.DataFrame({
    'Temperature': [values['Temperature']],
    'Humidity': [values['Humidity']],
    'Moisture': [values['Moisture']],
    'Nitrogen': [values['Nitrogen']],
    'Potassium': [values['Potassium']],
    'Phosphorous': [values['Phosphorous']]
})

# ========================= PREDICTIONS =========================
reg_pred = (rf_reg.predict(input_df) + xgb_reg.predict(input_df)) / 2
sdi_value = float(reg_pred[0])

rf_probs = rf_cls.predict_proba(input_df)
xgb_probs = xgb_cls.predict_proba(input_df)
cls_probs = (rf_probs + xgb_probs) / 2  # shape (1, n_classes)

# ensure labels -> Healthy/Low/Moderate/High
classes_from_model = list(rf_cls.classes_)
display_classes = []
for c in classes_from_model:
    try:
        display_classes.append(CLASS_LABELS_MAP[int(c)])
    except Exception:
        # if model already stores strings like 'Healthy' we keep them
        display_classes.append(str(c))

pred_idx = int(np.argmax(cls_probs[0]))
pred_class_label = display_classes[pred_idx]

# ========================= SHAP =========================
explainer_reg = shap.TreeExplainer(rf_reg)
shap_values_reg = explainer_reg.shap_values(input_df)  # (1, n_features)

explainer_cls = shap.TreeExplainer(rf_cls)
shap_values_cls = explainer_cls.shap_values(input_df)  # may be list/array depending on model

# ========================= HELPERS =========================
def get_level_from_sdi(v: float):
    if v <= 25:    return ("Healthy", "#16a34a", "🟢")
    if v <= 50:    return ("Low Degradation", "#f59e0b", "🟡")
    if v <= 75:    return ("Moderate Degradation", "#fb923c", "🟠")
    return ("High Degradation", "#ef4444", "🔴")

def feature_bar_figure(vals, features, title):
    vals = list(vals)
    fig = go.Figure([
        go.Bar(
            x=features,
            y=vals,
            marker_color=['#EF4444' if v>0 else '#22C55E' for v in vals],
            text=[f'{v:.3f}' for v in vals],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title=title,
        height=380,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=60),
        xaxis=dict(title='Features', tickangle=45),
        yaxis=dict(title='SHAP Value'),
        showlegend=False,
        font=dict(family='Inter', size=12)
    )
    return fig

# ========================= TABS =========================
tab1, tab2, tab3 = st.tabs(["SDI Analysis", "Classification", "Recommendations"])

# ---------- SDI ANALYSIS ----------
with tab1:
    left, right = st.columns([2,1])

    with left:
        level_text, level_color, level_emoji = get_level_from_sdi(sdi_value)

        st.markdown('<div class="sdi-card">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="sdi-avatar"><img src="{SDI_BADGE_IMAGE}" /></div>
            <div style="display:flex; gap:10px; align-items:center; color:#d93025; font-weight:800; font-size:18px;">
                <span>🌿</span><span>Soil Degradation Index</span>
            </div>
            <div style="display:flex; align-items:center; gap:18px; margin:14px 0 6px;">
                <div style="font-size:46px; font-weight:900; color:#d93025;">{sdi_value:.1f}</div>
                <div style="color:#6b7280; font-weight:600">SDI Score</div>
            </div>
            <span class="badge-pill" style="background:{level_color}1a; color:{level_color}; border:1px solid {level_color}55;">
                {level_emoji} {level_text}
            </span>
            <div style="margin:16px 0 8px">
              <div style="display:flex; justify-content:space-between; font-size:12px; color:#6b7280;">
                <span>0 (Healthy)</span><span>50 (Moderate)</span><span>100 (Critical)</span>
              </div>
              <div class="sdi-meter">
                <div style="height:8px; background:{level_color}; width:{min(sdi_value,100)}%; border-radius:6px;"></div>
              </div>
            </div>
            <div style="color:#6b7280; font-size:13px; margin-top:6px;">
              {"Critical condition, urgent intervention required!" if sdi_value>75 else "Condition under observation. Apply recommendations to improve soil health."}
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown('<div class="quick-tile"><div style="font-size:22px;">❌</div><div style="color:#6b7280;font-weight:700;">Crop Ready</div></div>', unsafe_allow_html=True)
        with colB:
            health_score = max(0.0, 100 - sdi_value)  # simple visual
            st.markdown(f'<div class="quick-tile"><div style="font-size:20px;font-weight:800;color:#16a34a;">{health_score:.0f}%</div><div style="color:#6b7280;font-weight:700;">Health Score</div></div>', unsafe_allow_html=True)
        with colC:
            risk = "High" if sdi_value>75 else ("Moderate" if sdi_value>50 else ("Low" if sdi_value>25 else "Low"))
            st.markdown(f'<div class="quick-tile"><div style="font-size:20px;font-weight:800;color:#ef4444;">{risk}</div><div style="color:#6b7280;font-weight:700;">Risk Level</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Contribution
    st.markdown('<div class="section-card"><h3 style="margin:0 0 6px;color:#15803d;">📈 Feature Contribution Analysis</h3><p style="color:#6b7280;margin:0 0 8px;">How each soil parameter influences the SDI score</p>', unsafe_allow_html=True)
    fig_contrib = feature_bar_figure(shap_values_reg[0], input_df.columns.tolist(), "")
    st.plotly_chart(fig_contrib, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        inc = [(f, v) for f, v in zip(input_df.columns, shap_values_reg[0]) if v>0]
        if inc:
            st.markdown('<div class="section-card" style="border-left:4px solid #ef4444;background:#FFF8F8;">'
                        '<h4 style="margin:0 0 6px;color:#ef4444;">⚠️ Increasing Degradation</h4><ul style="margin:0;">'
                        + "".join([f'<li>{f}: +{v:.3f}</li>' for f, v in inc]) +
                        '</ul></div>', unsafe_allow_html=True)
    with colB:
        dec = [(f, v) for f, v in zip(input_df.columns, shap_values_reg[0]) if v<0]
        if dec:
            st.markdown('<div class="section-card" style="border-left:4px solid #22C55E;background:#F6FFF8;">'
                        '<h4 style="margin:0 0 6px;color:#22C55E;">✅ Protecting Soil Health</h4><ul style="margin:0;">'
                        + "".join([f'<li>{f}: {v:.3f}</li>' for f, v in dec]) +
                        '</ul></div>', unsafe_allow_html=True)

# ---------- CLASSIFICATION ----------
with tab2:
    c1, c2 = st.columns([1,1])
    with c1:
        confidence = float(np.max(cls_probs[0]) * 100.0)
        st.markdown(f"""
        <div class="section-card" style="background:linear-gradient(180deg,#f4f8ff,#eef4ff);">
          <h3 style="margin:0;color:#1d4ed8;">🎯 Soil Classification Results</h3>
          <div style="margin:16px 0;">
            <span class="badge-pill" style="background:#ffe3e3; color:#c62828; border:1px solid #ffc6c6;">{ 'High Degradation' if pred_class_label=='High' else (pred_class_label+' Degradation' if pred_class_label!='Healthy' else 'Healthy') }</span>
          </div>
          <div style="font-size:34px;font-weight:900;color:#1d4ed8;">{confidence:.1f}%</div>
          <div style="color:#6b7280;">Confidence</div>
          <div style="margin-top:14px;padding:12px;border:1px dashed #bfd3ff;border-radius:10px;background:#f7faff;">
            <b>Model Prediction</b><br>
            Based on current soil parameters, the model predicts <b>{ 'High Degradation' if pred_class_label=='High' else (pred_class_label+' Degradation' if pred_class_label!='Healthy' else 'Healthy')}</b> with high confidence.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-card" style="background:linear-gradient(180deg,#faf5ff,#f5ecff); text-align:center;">
          <div style="font-weight:800; color:#7e22ce; font-size:18px;">📊 Soil Trend</div>
          <div style="font-size:26px; color:#7e22ce; margin:10px 0;">Declining</div>
          <div style="font-size:22px; font-weight:800; color:#7e22ce;">5/5</div>
          <div style="color:#6b7280;">Reliability Score</div>
        </div>
        """, unsafe_allow_html=True)

    # Probabilities (exact four rows in the given order)
    st.markdown('<div class="section-card"><h3 style="margin:0 0 6px;color:#15803d;">📊 Classification Probabilities</h3>', unsafe_allow_html=True)
    # Reorder to Healthy, Low, Moderate, High if present
    order = ["Healthy", "Low", "Moderate", "High"]
    probs_dict = dict(zip(display_classes, cls_probs[0]))
    for i, name in enumerate(order):
        if name in probs_dict:
            pct = float(probs_dict[name] * 100.0)
            st.markdown(f"""
            <div style="margin:10px 0;">
              <div style="display:flex; justify-content:space-between; font-weight:700;">
                <span>{name}</span><span>{pct:.1f}%</span>
              </div>
              <div style="background:#eef2f7; height:12px; border-radius:999px;">
                <div style="height:12px; width:{pct}%; border-radius:999px; background:{['#22C55E','#F59E0B','#FB923C','#EF4444'][i]};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature contribution for classifier
    st.markdown('<div class="section-card"><h3 style="margin:0 0 6px;color:#15803d;">📈 Feature Contribution Analysis</h3><p style="color:#6b7280;margin:0 0 8px;">How each soil parameter influences the classification</p>', unsafe_allow_html=True)
    if isinstance(shap_values_cls, list):
        # for classifiers, shap returns list per class; pick predicted class
        sv = shap_values_cls[pred_idx][0]
    elif len(np.shape(shap_values_cls)) == 3:
        sv = shap_values_cls[0, :, pred_idx]
    else:
        sv = shap_values_cls[0]
    st.plotly_chart(feature_bar_figure(sv, input_df.columns.tolist(), ""), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- RECOMMENDATIONS ----------
with tab3:
    st.markdown(f"""
    <div class="section-card" style="background:#F1FFF4;">
      <h3 style="margin:0;color:#15803d;">💡 Smart Farming Recommendations</h3>
      <div style="display:flex; gap:10px; margin-top:10px;">
        <span class="badge-pill" style="background:#e8f1ff;color:#1d4ed8;border:1px solid #cfe0ff;">SDI: {sdi_value:.1f}</span>
        <span class="badge-pill badge-high">{'High Degradation' if pred_class_label=='High' else (pred_class_label+' Degradation' if pred_class_label!='Healthy' else 'Healthy')}</span>
        <span class="badge-pill" style="background:#fff0d8;color:#b45309;border:1px solid #ffd9a8;">High Priority Actions</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cL, cR = st.columns([2,1])

    with cL:
        st.markdown('<div class="section-card"><h3 style="margin:0 0 8px;color:#15803d;">⚡ Immediate Actions Required</h3>'
                    '<div style="display:flex;align-items:center;justify-content:space-between;border:1px solid #ffe0e0;padding:12px;border-radius:12px;background:#FFF8F8">'
                    '<div><b>🧪 Reduce Phosphorous Application</b>'
                    '<div style="color:#6b7280;font-size:13px;margin-top:4px;">Current: '
                    f'{values["Phosphorous"]} mg/kg | Optimal: 25–35 mg/kg'
                    '</div></div>'
                    '<span class="badge-pill" style="background:#ef4444;color:#fff;border:1px solid #ef4444;">High ⚠️</span>'
                    '</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><h3 style="margin:0 0 8px;color:#15803d;">📋 Detailed Action Plan</h3>'
                    '<p style="color:#6b7280;margin:0 0 8px;">Prioritized recommendations based on soil analysis</p>', unsafe_allow_html=True)

        actions = [
            {"name":"Phosphorous Application","curr":f'{values["Phosphorous"]} mg/kg',"opt":"25–35 mg/kg","action":"Reduce phosphorous application","priority":"High","impact":abs(float(sv[list(input_df.columns).index("Phosphorous")])) if "Phosphorous" in input_df.columns else 3.0,"icon":"🧪"},
            {"name":"Temperature Control","curr":f'{values["Temperature"]} °C',"opt":"20–30 °C","action":"Monitor temperature control","priority":"Low","impact":0.0,"icon":"🌡️"},
            {"name":"Humidity Management","curr":f'{values["Humidity"]} %',"opt":"60–80%","action":"Monitor humidity management","priority":"Low","impact":0.0,"icon":"💧"},
            {"name":"Irrigation Management","curr":f'{values["Moisture"]} %',"opt":"50–70%","action":"Monitor irrigation management","priority":"Low","impact":0.0,"icon":"💦"},
            {"name":"Nitrogen Fertilization","curr":f'{values["Nitrogen"]} mg/kg',"opt":"40–80 mg/kg","action":"Monitor nitrogen fertilization","priority":"Low","impact":0.0,"icon":"🌿"},
            {"name":"Potassium Treatment","curr":f'{values["Potassium"]} mg/kg',"opt":"30–60 mg/kg","action":"Monitor potassium treatment","priority":"Low","impact":0.0,"icon":"🪨"},
        ]
        for a in actions:
            badge_bg = "#ef4444" if a["priority"]=="High" else "#22C55E"
            st.markdown(
                f"""
                <div class="section-card" style="padding:14px; margin-top:10px;">
                  <div style="display:flex;justify-content:space-between;align-items:start;gap:10px;">
                    <div style="display:flex;gap:10px;align-items:center;">
                      <div style="font-size:20px">{a["icon"]}</div>
                      <div>
                        <div style="font-weight:800">{a["name"]}</div>
                        <div style="color:#6b7280;font-size:13px;">Current: {a["curr"]} | Optimal: {a["opt"]}</div>
                        <div style="font-size:13px;margin-top:4px;">Action: {a["action"]}</div>
                        <div style="font-size:12px;color:#94a3b8;margin-top:4px;">Impact Score: {a["impact"]:.3f}</div>
                      </div>
                    </div>
                    <span class="badge-pill" style="background:{badge_bg}; color:#fff; border:1px solid {badge_bg};">{a["priority"]}{' ⚠️' if a["priority"]=='High' else ' ✓'}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    with cR:
        st.markdown('<div class="section-card"><h3 style="margin:0 0 8px;color:#15803d;">📊 Impact Analysis</h3>', unsafe_allow_html=True)
        # simple impact chart from SHAP |values|
        impacts = [abs(float(x)) for x in (sv if "sv" in locals() else shap_values_reg[0])]
        fig_imp = go.Figure([go.Bar(x=input_df.columns.tolist(), y=impacts,
                                    marker_color=['#22C55E']*5+['#EF4444'])])
        fig_imp.update_layout(height=280, plot_bgcolor='white', paper_bgcolor='white',
                              margin=dict(l=20,r=20,t=10,b=70),
                              xaxis=dict(tickangle=45), yaxis=dict(title="Impact Score"))
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # CSV download
        rec_df = pd.DataFrame({
            "Action":[a["name"] for a in actions],
            "Priority":[a["priority"] for a in actions],
            "Current_Value":[a["curr"] for a in actions],
            "Optimal_Range":[a["opt"] for a in actions],
            "Impact_Score":[round(a["impact"],3) for a in actions],
        })
        st.download_button(
            "📥 Download Recommendations CSV",
            data=rec_df.to_csv(index=False),
            file_name="soil_health_recommendations.csv",
            mime="text/csv",
            use_container_width=True
        )

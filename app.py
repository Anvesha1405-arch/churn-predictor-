import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

# ── Auto-train model if model.pkl doesn't exist (Streamlit Cloud fix) ──────────
def ensure_model():
    model_path = BASE_DIR / "model.pkl"
    if model_path.exists():
        return
    df = pd.read_csv(BASE_DIR / "customer_churn_prediction_dataset.csv")
    df = df.drop(columns=["customerID"], errors="ignore")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    cat_cols = [c for c in df.columns if c != "Churn"
                and df[c].dtype not in [np.float64, np.int64, np.int32, np.float32]]
    df_enc = pd.get_dummies(df, columns=cat_cols, dtype=int)
    X = df_enc.drop(columns=["Churn"]).astype(float)
    y = df_enc["Churn"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    clf.fit(X_s, y)
    bundle = {"scaler": scaler, "clf": clf, "feature_names": X.columns.to_numpy()}
    joblib.dump(bundle, model_path)

ensure_model()

# ── Auto-train model if model.pkl doesn't exist ────────────────────────────────
def ensure_model():
    model_path = BASE_DIR / "model.pkl"
    if model_path.exists():
        return
    df = pd.read_csv(BASE_DIR / "customer_churn_prediction_dataset.csv")
    df = df.drop(columns=["customerID"], errors="ignore")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    cat_cols = [c for c in df.columns if c != "Churn"
                and df[c].dtype not in [np.float64, np.int64, np.int32, np.float32]]
    df_enc = pd.get_dummies(df, columns=cat_cols, dtype=int)
    X = df_enc.drop(columns=["Churn"]).astype(float)
    y = df_enc["Churn"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    clf.fit(X_s, y)
    bundle = {"scaler": scaler, "clf": clf, "feature_names": X.columns.to_numpy()}
    joblib.dump(bundle, model_path)

ensure_model()   # ← runs before anything else

# ── Page Config ────────────────────────────────────────────
st.set_page_config(...)   # rest of your app continues here

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="ChurnScope · AI Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:#0a0e1a; --card:#111827; --card2:#1a2235;
    --border:#1e2d42; --accent:#00d4ff; --accent2:#ff6b6b;
    --green:#00e5a0; --text:#e2e8f0; --muted:#64748b;
}
html,body,[class*="css"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;}
.stApp{background:linear-gradient(135deg,#0a0e1a 0%,#0d1525 50%,#0a0e1a 100%)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem!important;max-width:1400px;}

.hero-container{background:linear-gradient(135deg,#111827 0%,#0f1e33 100%);border:1px solid var(--border);border-radius:20px;padding:3rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero-container::before{content:'';position:absolute;top:-50%;right:-10%;width:400px;height:400px;background:radial-gradient(circle,rgba(0,212,255,0.08) 0%,transparent 70%);pointer-events:none;}
.hero-title{font-family:'DM Serif Display',serif;font-size:3.2rem;letter-spacing:-1px;background:linear-gradient(135deg,#00d4ff,#ffffff 60%,#00e5a0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem;}
.hero-sub{font-size:1.1rem;color:var(--muted);font-weight:300;}
.hero-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);border-radius:50px;padding:4px 14px;font-size:0.75rem;color:var(--accent);font-family:'Space Mono',monospace;margin-bottom:1.5rem;}

.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0;}
.metric-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.4rem 1.8rem;position:relative;overflow:hidden;}
.metric-card::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;border-radius:3px;}
.metric-card.cyan::after{background:var(--accent);}
.metric-card.red::after{background:var(--accent2);}
.metric-card.green::after{background:var(--green);}
.metric-label{font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;font-family:'Space Mono',monospace;}
.metric-val{font-size:2rem;font-weight:600;font-family:'DM Serif Display',serif;}
.metric-val.cyan{color:var(--accent);} .metric-val.red{color:var(--accent2);} .metric-val.green{color:var(--green);}
.metric-sub{font-size:0.78rem;color:var(--muted);margin-top:0.3rem;}

.section-title{font-family:'DM Serif Display',serif;font-size:1.6rem;color:white;margin-bottom:0.3rem;}
.section-divider{height:1px;background:linear-gradient(90deg,var(--accent),transparent);margin:0.8rem 0 1.5rem;}

[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.sidebar-header{font-family:'DM Serif Display',serif;font-size:1.4rem;color:var(--accent)!important;padding:1rem 0 0.5rem;border-bottom:1px solid var(--border);margin-bottom:1rem;}
.sidebar-section{font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted)!important;margin:1.2rem 0 0.5rem;}

.result-card{border-radius:18px;padding:2.5rem;text-align:center;border:1px solid;position:relative;overflow:hidden;}
.result-card.churn{background:linear-gradient(135deg,rgba(255,107,107,0.12),rgba(255,107,107,0.03));border-color:rgba(255,107,107,0.4);}
.result-card.safe{background:linear-gradient(135deg,rgba(0,229,160,0.12),rgba(0,229,160,0.03));border-color:rgba(0,229,160,0.4);}
.result-icon{font-size:4rem;margin-bottom:0.5rem;}
.result-label{font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;}
.result-verdict{font-family:'DM Serif Display',serif;font-size:2.5rem;margin-bottom:1rem;}
.result-verdict.churn{color:var(--accent2);} .result-verdict.safe{color:var(--green);}
.result-prob{font-size:1rem;color:var(--muted);}

.gauge-container{margin:1.5rem 0;}
.gauge-track{background:var(--card2);height:12px;border-radius:6px;overflow:hidden;border:1px solid var(--border);}
.gauge-fill{height:100%;border-radius:6px;}

.stButton>button{width:100%;background:linear-gradient(135deg,var(--accent),#0099cc)!important;color:#000!important;font-weight:700!important;font-family:'DM Sans',sans-serif!important;border:none!important;border-radius:10px!important;padding:0.75rem!important;font-size:0.95rem!important;}

.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border-radius:10px;padding:4px;gap:4px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:#000!important;font-weight:600!important;}

.factor-row{display:flex;justify-content:space-between;align-items:center;padding:0.8rem 1.2rem;border-radius:10px;margin-bottom:0.5rem;background:var(--card2);border:1px solid var(--border);font-size:0.88rem;}
.factor-name{color:var(--text);}
.factor-impact{font-family:'Space Mono',monospace;font-size:0.8rem;}
.factor-impact.neg{color:var(--accent2);} .factor-impact.pos{color:var(--green);}

.info-box{background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);border-radius:10px;padding:1rem 1.2rem;font-size:0.85rem;color:var(--muted);margin:1rem 0;}
.warn-box{background:rgba(255,107,107,0.08);border:1px solid rgba(255,107,107,0.3);border-radius:10px;padding:1rem 1.2rem;font-size:0.85rem;color:#ff9999;margin:1rem 0;}
</style>
""", unsafe_allow_html=True)

# ── Load model bundle & data ────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    path = BASE_DIR / "model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "customer_churn_prediction_dataset.csv")

bundle = load_bundle()
df     = load_data()

# ── Guard: show setup instructions if model not found ──────────────────────────
if bundle is None:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">◉ SETUP REQUIRED</div>
        <div class="hero-title">ChurnScope</div>
        <div class="hero-sub">Run the one-time setup script first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
        ⚠️ <strong>model.pkl not found.</strong><br><br>
        Open a terminal in your project folder and run:<br><br>
        <code style="background:#1a2235;padding:6px 12px;border-radius:6px;font-family:'Space Mono',monospace;">python setup_model.py</code><br><br>
        Then refresh this page.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

scaler        = bundle["scaler"]
clf           = bundle["clf"]
FEATURE_NAMES = bundle["feature_names"]

# ── Preprocessing helper ────────────────────────────────────────────────────────
def build_input_df(inputs: dict) -> pd.DataFrame:
    row = {f: 0 for f in FEATURE_NAMES}
    row['SeniorCitizen']  = inputs['SeniorCitizen']
    row['tenure']         = inputs['tenure']
    row['MonthlyCharges'] = inputs['MonthlyCharges']
    row['TotalCharges']   = inputs['tenure'] * inputs['MonthlyCharges']
    cat_map = {
        'gender': inputs['gender'], 'Partner': inputs['Partner'],
        'Dependents': inputs['Dependents'], 'PhoneService': inputs['PhoneService'],
        'MultipleLines': inputs['MultipleLines'], 'InternetService': inputs['InternetService'],
        'OnlineSecurity': inputs['OnlineSecurity'], 'OnlineBackup': inputs['OnlineBackup'],
        'DeviceProtection': inputs['DeviceProtection'], 'TechSupport': inputs['TechSupport'],
        'StreamingTV': inputs['StreamingTV'], 'StreamingMovies': inputs['StreamingMovies'],
        'Contract': inputs['Contract'], 'PaperlessBilling': inputs['PaperlessBilling'],
        'PaymentMethod': inputs['PaymentMethod'],
    }
    for feat, val in cat_map.items():
        key = f"{feat}_{val}"
        if key in row:
            row[key] = 1
    return pd.DataFrame([row], dtype=float)

def predict(inputs):
    X = build_input_df(inputs)
    X_s = scaler.transform(X)
    prob = clf.predict_proba(X_s)[0]
    return prob[0], prob[1]   # safe_prob, churn_prob

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🔮 ChurnScope</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Fill in the customer profile and the AI will update the prediction in real-time.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">👤 Demographics</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    partner    = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    st.markdown('<div class="sidebar-section">📞 Services</div>', unsafe_allow_html=True)
    phone       = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    adv = ["No internet service"] if internet == "No" else ["Yes", "No"]
    online_sec    = st.selectbox("Online Security",   adv)
    online_bkp    = st.selectbox("Online Backup",     adv)
    device_prot   = st.selectbox("Device Protection", adv)
    tech_sup      = st.selectbox("Tech Support",      adv)
    stream_tv     = st.selectbox("Streaming TV",      adv)
    stream_movies = st.selectbox("Streaming Movies",  adv)

    st.markdown('<div class="sidebar-section">💳 Billing</div>', unsafe_allow_html=True)
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    st.markdown('<div class="sidebar-section">📊 Account Metrics</div>', unsafe_allow_html=True)
    tenure          = st.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)

# ── Compute prediction ──────────────────────────────────────────────────────────
inputs = dict(
    gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents,
    PhoneService=phone, MultipleLines=multi_lines, InternetService=internet,
    OnlineSecurity=online_sec, OnlineBackup=online_bkp, DeviceProtection=device_prot,
    TechSupport=tech_sup, StreamingTV=stream_tv, StreamingMovies=stream_movies,
    Contract=contract, PaperlessBilling=paperless, PaymentMethod=payment_method,
    tenure=tenure, MonthlyCharges=monthly_charges,
)
safe_prob, churn_prob = predict(inputs)
is_churn = churn_prob >= 0.5

# ── HERO ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">◉ &nbsp;LOGISTIC REGRESSION MODEL &nbsp;·&nbsp; v2.0</div>
    <div class="hero-title">Customer Churn Intelligence</div>
    <div class="hero-sub">AI-powered churn risk scoring &amp; retention analytics for telecom customers</div>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ────────────────────────────────────────────────────────────────────
churn_rate  = (df['Churn'] == 'Yes').mean()
avg_tenure  = df['tenure'].mean()
avg_monthly = df['MonthlyCharges'].mean()

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card cyan">
        <div class="metric-label">Total Customers</div>
        <div class="metric-val cyan">{len(df):,}</div>
        <div class="metric-sub">in training dataset</div>
    </div>
    <div class="metric-card red">
        <div class="metric-label">Churn Rate</div>
        <div class="metric-val red">{churn_rate:.1%}</div>
        <div class="metric-sub">historical base rate</div>
    </div>
    <div class="metric-card green">
        <div class="metric-label">Avg Tenure</div>
        <div class="metric-val green">{avg_tenure:.0f}mo</div>
        <div class="metric-sub">avg monthly charge ${avg_monthly:.0f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Prediction Result", "📊  Dataset Analytics", "⚙️  Model Intelligence"])

PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="DM Sans"),
    margin=dict(t=40, b=20, l=10, r=10),
)

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([1, 1.4], gap="large")

    with c1:
        card_cls = "churn" if is_churn else "safe"
        icon     = "⚠️"   if is_churn else "✅"
        verdict  = "LIKELY TO CHURN" if is_churn else "LIKELY TO STAY"
        bar_col  = "#ff6b6b"         if is_churn else "#00e5a0"

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="result-icon">{icon}</div>
            <div class="result-label">AI Verdict</div>
            <div class="result-verdict {card_cls}">{verdict}</div>
            <div class="result-prob">Churn probability: <strong>{churn_prob:.1%}</strong></div>
        </div>
        <div class="gauge-container">
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748b;margin-bottom:6px;font-family:'Space Mono',monospace;">
                <span>SAFE</span><span>CHURN RISK</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{churn_prob*100:.1f}%;background:{bar_col};"></div>
            </div>
            <div style="text-align:right;font-size:0.75rem;color:{bar_col};margin-top:4px;font-family:'Space Mono',monospace;">{churn_prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        fig_d = go.Figure(go.Pie(
            values=[safe_prob, churn_prob], labels=["Will Stay", "Will Churn"],
            hole=0.72, marker=dict(colors=["#00e5a0","#ff6b6b"], line=dict(color="#0a0e1a",width=3)),
            textinfo='none', hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
        ))
        fig_d.add_annotation(
            text=f"<b>{churn_prob:.0%}</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color="#ff6b6b" if is_churn else "#00e5a0", family="DM Serif Display")
        )
        fig_d.update_layout(**PLOTLY, showlegend=True, height=210,
            legend=dict(font=dict(color="#94a3b8",size=11), orientation="h", x=0.15, y=-0.05))
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<div class="section-title">Risk Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        risk_factors = []
        if contract == "Month-to-month":   risk_factors.append(("Month-to-month contract", "HIGH RISK", "neg"))
        elif contract == "Two year":       risk_factors.append(("Two-year contract", "PROTECTIVE", "pos"))
        else:                              risk_factors.append(("One-year contract", "MODERATE", "pos"))
        if internet == "Fiber optic":      risk_factors.append(("Fiber optic internet", "ELEVATED CHURN", "neg"))
        elif internet == "No":             risk_factors.append(("No internet service", "LOWER RISK", "pos"))
        if online_sec == "No":             risk_factors.append(("No online security", "RISK FACTOR", "neg"))
        if tech_sup == "No":               risk_factors.append(("No tech support", "RISK FACTOR", "neg"))
        if payment_method == "Electronic check": risk_factors.append(("Electronic check", "RISK FACTOR", "neg"))
        if tenure < 12:                    risk_factors.append((f"Short tenure ({tenure}mo)", "HIGH RISK", "neg"))
        elif tenure > 36:                  risk_factors.append((f"Long tenure ({tenure}mo)", "LOYAL SIGNAL", "pos"))
        if monthly_charges > 80:           risk_factors.append((f"High charges (${monthly_charges:.0f}/mo)", "RISK FACTOR", "neg"))
        if partner == "Yes":               risk_factors.append(("Has partner", "RETENTION SIGNAL", "pos"))
        if dependents == "Yes":            risk_factors.append(("Has dependents", "RETENTION SIGNAL", "pos"))

        for name, impact, cls in risk_factors[:7]:
            st.markdown(f"""
            <div class="factor-row">
                <span class="factor-name">{name}</span>
                <span class="factor-impact {cls}">{impact}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1.5rem">Retention Actions</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if is_churn:
            recs = []
            if contract == "Month-to-month":   recs.append("🎁 Offer discounted annual contract upgrade")
            if online_sec == "No" or tech_sup == "No": recs.append("🛡️ Bundle security + tech support free (3 months)")
            if monthly_charges > 80:           recs.append("💰 Apply loyalty discount to reduce monthly bill")
            if tenure < 12:                    recs.append("🤝 Assign dedicated onboarding manager")
            recs.append("📞 Schedule proactive check-in call within 7 days")
            for r in recs[:4]:
                st.markdown(f'<div class="info-box">{r}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">✅ <strong>Low Risk Customer</strong> — No immediate retention intervention required.<br><br>Consider upselling premium add-ons to increase ARPU.</div>', unsafe_allow_html=True)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Dataset Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ct = df.groupby(['Contract','Churn']).size().reset_index(name='count')
        fig1 = px.bar(ct, x='Contract', y='count', color='Churn',
                      color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                      barmode='group', title='Churn by Contract Type')
        fig1.update_layout(**PLOTLY, title_font=dict(size=14,color='white'), bargap=0.3,
                           xaxis=dict(gridcolor='#1e2d42'), yaxis=dict(gridcolor='#1e2d42'))
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig2 = go.Figure()
        for v, col in [('Yes','#ff6b6b'),('No','#00d4ff')]:
            fig2.add_trace(go.Histogram(x=df[df['Churn']==v]['tenure'], name=f"Churn: {v}",
                           marker_color=col, opacity=0.75, nbinsx=20))
        fig2.update_layout(**PLOTLY, title="Tenure Distribution by Churn", barmode='overlay',
                           title_font=dict(size=14,color='white'),
                           xaxis=dict(title="Tenure (months)",gridcolor='#1e2d42'),
                           yaxis=dict(gridcolor='#1e2d42'))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        fig3 = go.Figure()
        for v, col in [('Yes','#ff6b6b'),('No','#00e5a0')]:
            fig3.add_trace(go.Box(y=df[df['Churn']==v]['MonthlyCharges'], name=f"Churn: {v}",
                           marker_color=col, line_color=col))
        fig3.update_layout(**PLOTLY, title="Monthly Charges vs Churn",
                           title_font=dict(size=14,color='white'),
                           yaxis=dict(title="Monthly Charges ($)",gridcolor='#1e2d42'))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with c4:
        net_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x=='Yes').mean()).reset_index()
        net_churn.columns = ['Service','ChurnRate']
        fig4 = px.bar(net_churn, x='Service', y='ChurnRate',
                      color='ChurnRate', color_continuous_scale=['#00e5a0','#00d4ff','#ff6b6b'],
                      title='Churn Rate by Internet Service', text_auto='.1%')
        fig4.update_layout(**PLOTLY, title_font=dict(size=14,color='white'),
                           coloraxis_showscale=False,
                           xaxis=dict(gridcolor='#1e2d42'), yaxis=dict(gridcolor='#1e2d42'))
        fig4.update_traces(textfont_color='white')
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    pay = df.groupby(['PaymentMethod','Churn']).size().unstack(fill_value=0)
    pay['ChurnRate'] = pay['Yes'] / (pay['Yes'] + pay['No'])
    fig5 = go.Figure(go.Bar(
        x=pay.index, y=pay['ChurnRate'],
        marker=dict(color=pay['ChurnRate'],
                    colorscale=[[0,'#00e5a0'],[0.5,'#00d4ff'],[1,'#ff6b6b']],
                    showscale=True, colorbar=dict(title="Churn Rate",tickformat='.0%',tickfont=dict(color='#94a3b8'))),
        text=[f"{r:.1%}" for r in pay['ChurnRate']], textposition='outside',
        textfont=dict(color='white',size=12)
    ))
    fig5.update_layout(**PLOTLY, title="Churn Rate by Payment Method",
                       title_font=dict(size=14,color='white'),
                       xaxis=dict(gridcolor='#1e2d42'), yaxis=dict(gridcolor='#1e2d42',tickformat='.0%'))
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    cm1, cm2 = st.columns([1.2, 1], gap="large")
    with cm1:
        coef = clf.coef_[0]
        fc = pd.DataFrame({'Feature': FEATURE_NAMES, 'Coefficient': coef})
        fc = fc.reindex(fc['Coefficient'].abs().sort_values(ascending=False).index).head(16)
        colors = ['#ff6b6b' if c > 0 else '#00e5a0' for c in fc['Coefficient']]
        fig_c = go.Figure(go.Bar(
            x=fc['Coefficient'], y=fc['Feature'], orientation='h', marker_color=colors,
            text=[f"{v:+.3f}" for v in fc['Coefficient']], textposition='outside',
            textfont=dict(color='#94a3b8', size=10)
        ))
        fig_c.update_layout(**PLOTLY, title="Top Feature Coefficients (Red=Churn, Green=Stay)",
                            title_font=dict(size=13,color='white'), height=500,
                            yaxis=dict(autorange='reversed',gridcolor='#1e2d42'),
                            xaxis=dict(gridcolor='#1e2d42',zeroline=True,
                                       zerolinecolor='#2d3f57',zerolinewidth=2))
        st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})

    with cm2:
        st.markdown(f"""
        <div class="metric-card cyan" style="margin-bottom:1rem;">
            <div class="metric-label">Model Type</div>
            <div class="metric-val cyan" style="font-size:1.1rem;font-family:'Space Mono',monospace;">Logistic Regression</div>
        </div>
        <div class="metric-card green" style="margin-bottom:1rem;">
            <div class="metric-label">Features Used</div>
            <div class="metric-val green">{len(FEATURE_NAMES)}</div>
            <div class="metric-sub">one-hot encoded + numerical</div>
        </div>
        <div class="metric-card red" style="margin-bottom:1rem;">
            <div class="metric-label">Output Classes</div>
            <div class="metric-val red">2</div>
            <div class="metric-sub">Churn (1) · Stay (0)</div>
        </div>
        <div class="info-box">
            <strong style="color:#00d4ff;">How it works:</strong><br><br>
            Input features are scaled, then multiplied by learned coefficients.
            The sigmoid function converts the sum into a probability 0–1.<br><br>
            • <strong style="color:#ff6b6b;">Positive coeff</strong> → raises churn risk<br>
            • <strong style="color:#00e5a0;">Negative coeff</strong> → lowers churn risk
        </div>
        <div class="result-card {'churn' if is_churn else 'safe'}" style="margin-top:1rem;padding:1.5rem;">
            <div class="result-label">Live Score</div>
            <div class="result-verdict {'churn' if is_churn else 'safe'}" style="font-size:2rem;">{churn_prob:.1%}</div>
            <div class="result-prob">churn probability</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#334155;font-size:0.75rem;font-family:'Space Mono',monospace;border-top:1px solid #1e2d42;margin-top:2rem;">
    CHURNSCOPE · LOGISTIC REGRESSION · FOR DEMO PURPOSES ONLY
</div>
""", unsafe_allow_html=True)

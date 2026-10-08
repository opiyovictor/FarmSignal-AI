"""FarmSignal AI — Kenyan farm intelligence dashboard.

Run with:
    python -m streamlit run src/dashboard.py

The dashboard consumes the project's existing AI risk outputs and enriches them
with a synthetic operational layer representing Kenyan smallholder farming:
counties, crops, field health, rainfall, soil moisture, market prices and
field-agent actions. All operational data is synthetic and clearly labelled.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

st.set_page_config(
    page_title="FarmSignal AI | Kenya Farm Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --navy:#08233b; --navy2:#0d304d; --green:#20a464; --lime:#72c83e; --amber:#e9a72f; --red:#df4d4d; --muted:#66768a; --line:#dfe7ee; --bg:#f5f8fa; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
.stApp { background:var(--bg); }
.block-container { padding-top:3.8rem; padding-bottom:2rem; max-width:1600px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#071f35 0%,#0b2d49 100%); }
[data-testid="stSidebar"] * { color:#e8f2f8 !important; }
.brand { padding:6px 2px 22px; border-bottom:1px solid rgba(255,255,255,.12); margin-bottom:18px; }
.brand-title { font-family:'Manrope'; font-weight:800; font-size:23px; letter-spacing:-.5px; }
.brand-title span { color:#4fd37e; }
.brand-sub { color:#a9bfce !important; font-size:11px; margin-top:2px; }
.topbar { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:16px; padding:2px 2px 0; }
.topbar-copy { min-width:0; }
.page-title { white-space:normal; overflow:visible; line-height:1.15; }
@media (max-width: 900px) { .block-container { padding-top:3rem; } .topbar { flex-direction:column; } }
.page-title { font-family:'Manrope'; font-size:29px; font-weight:800; color:#0a2942; letter-spacing:-1px; }
.page-sub { color:var(--muted); font-size:13px; }
.badge { display:inline-block; padding:5px 9px; border-radius:20px; font-size:11px; font-weight:700; background:#e8f7ef; color:#187a49; }
.card { background:white; border:1px solid var(--line); border-radius:14px; padding:16px 17px; box-shadow:0 2px 8px rgba(20,49,70,.04); }
.kpi-label { color:#65768a; font-size:12px; font-weight:600; }
.kpi-value { color:#0a2942; font-family:'Manrope'; font-size:28px; font-weight:800; margin-top:3px; }
.kpi-note { font-size:11px; color:#66768a; margin-top:3px; }
.section-title { color:#0b2d49; font-family:'Manrope'; font-weight:800; font-size:16px; }
.section-note { color:#718196; font-size:11px; }
.insight { padding:12px 13px; border-radius:11px; margin-bottom:10px; border:1px solid #e5edf2; background:#fbfcfd; }
.insight strong { color:#12344e; font-size:12px; }
.insight p { margin:4px 0 0; color:#627286; font-size:11px; line-height:1.45; }
.alert-red { border-left:4px solid var(--red); }
.alert-amber { border-left:4px solid var(--amber); }
.alert-green { border-left:4px solid var(--green); }
.small { font-size:11px; color:#718196; }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:13px 15px; box-shadow:0 2px 8px rgba(20,49,70,.04); }
div[data-testid="stMetricLabel"] { color:#66768a; }
div[data-testid="stMetricValue"] { color:#0a2942; }
.stButton button { border-radius:9px; font-weight:700; }
hr { border:0; border-top:1px solid var(--line); margin:12px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data loading / operational enrichment
# -----------------------------------------------------------------------------
@st.cache_data

def load_core():
    farmers = pd.read_csv(DATA / "farmers.csv")
    scores = pd.read_csv(DATA / "risk_scores.csv")
    actions = pd.read_csv(LOGS / "actions_log.csv")
    importance = pd.read_csv(MODELS / "global_feature_importance.csv")
    df = farmers.merge(scores, on=["farmer_id", "region"], how="left")
    return farmers, df, actions, importance


@st.cache_data

def build_operations(farmers: pd.DataFrame, df: pd.DataFrame):
    rng = np.random.default_rng(2026)
    county_meta = {
        "Kakamega": (-0.28, 34.75), "Bungoma": (0.57, 34.56), "Busia": (0.46, 34.11),
        "Vihiga": (0.08, 34.72), "Trans Nzoia": (1.01, 35.01), "Siaya": (0.06, 34.29),
    }
    crops = ["Maize", "Beans", "Sorghum", "Tomatoes", "Kale", "Groundnuts"]
    crop_probs = [0.42, 0.20, 0.11, 0.09, 0.10, 0.08]

    n_fields = 72
    field_counties = rng.choice(list(county_meta), n_fields, p=[.22,.18,.13,.12,.17,.18])
    field_crops = rng.choice(crops, n_fields, p=crop_probs)
    rows = []
    for i, (county, crop) in enumerate(zip(field_counties, field_crops), 1):
        lat, lon = county_meta[county]
        area = round(float(rng.uniform(1.2, 18.0)), 1)
        health_score = round(float(np.clip(rng.normal(76, 16), 28, 98)), 0)
        rainfall = round(float(np.clip(rng.normal(62, 22), 15, 125)), 1)
        moisture = round(float(np.clip(rng.normal(64, 17), 18, 96)), 0)
        pest = round(float(np.clip(rng.beta(2, 8) * 100, 1, 55)), 0)
        yield_t = {"Maize":3.2,"Beans":1.5,"Sorghum":1.9,"Tomatoes":13.5,"Kale":9.5,"Groundnuts":1.3}[crop]
        predicted = round(float(max(.4, rng.normal(yield_t, yield_t*.18))), 1)
        status = "Healthy" if health_score >= 72 else "At Risk" if health_score >= 50 else "Critical"
        rows.append({
            "field_id": f"F-{i:03d}", "county": county, "crop": crop, "area_ha": area,
            "health_score": health_score, "health_status": status, "rainfall_mm_7d": rainfall,
            "soil_moisture_pct": moisture, "pest_pressure_pct": pest,
            "predicted_yield_t_ha": predicted, "lat": lat+rng.normal(0,.055), "lon": lon+rng.normal(0,.055),
            "last_scouted_days": int(rng.integers(1, 12)),
        })
    fields = pd.DataFrame(rows)

    # Market prices: plausible illustrative KES/kg values for portfolio use.
    base = {"Maize":58,"Beans":142,"Sorghum":74,"Tomatoes":96,"Kale":63,"Groundnuts":118}
    markets = []
    for county in county_meta:
        for crop in crops:
            price = max(20, base[crop] * float(rng.normal(1, .08)))
            markets.append({"county":county,"crop":crop,"price_kes_kg":round(price,1),"price_change_pct":round(float(rng.normal(1.5,4)),1)})
    market = pd.DataFrame(markets)

    # Farmer-to-field assignment gives the existing AI scores an operational location.
    df2 = df.copy()
    df2["field_id"] = rng.choice(fields["field_id"], len(df2))
    field_lookup = fields.set_index("field_id")["crop"]
    df2["crop"] = df2["field_id"].map(field_lookup)
    return fields, market, df2


def fmt_int(v):
    return f"{int(v):,}"


farmers, df, actions, importance = load_core()
fields, market, df = build_operations(farmers, df)

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    "<div class='brand'><div class='brand-title'>🌱 FarmSignal <span>AI</span></div><div class='brand-sub'>Smarter data. Healthier farms. Greater yields.</div></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("**PORTFOLIO VIEW**")
page = st.sidebar.radio("", ["Dashboard", "Fields & Crops", "AI Risk", "Market & Weather", "Actions & Monitoring"], label_visibility="collapsed")
st.sidebar.markdown("---")
counties = ["All counties"] + sorted(fields["county"].unique().tolist())
county_filter = st.sidebar.selectbox("County", counties)
seasons = ["Long Rains 2026", "Short Rains 2026"]
season = st.sidebar.selectbox("Season", seasons, index=0)
if county_filter != "All counties":
    fview = fields[fields["county"] == county_filter].copy()
    dview = df[df["region"] == county_filter].copy()
else:
    fview, dview = fields.copy(), df.copy()
st.sidebar.markdown("---")
st.sidebar.markdown("**Data status**")
st.sidebar.success("AI scoring pipeline online")
st.sidebar.caption("Synthetic portfolio data • Kenya\nLast refresh: 06 Oct 2026")

# -----------------------------------------------------------------------------
# Shared header
# -----------------------------------------------------------------------------
st.markdown(
    f"<div class='topbar'><div class='topbar-copy'><div class='page-title'>FarmSignal AI — Farm Intelligence Dashboard</div><div class='page-sub'>Kenyan smallholder portfolio • Western Kenya focus • {season} • {county_filter}</div></div><div><span class='badge'>● AI SYSTEM HEALTHY</span></div></div>",
    unsafe_allow_html=True,
)

if page == "Dashboard":
    # KPIs
    total_area = fview["area_ha"].sum()
    healthy = (fview["health_status"] == "Healthy").mean()
    high_risk = (dview["risk_tier"] == "high").sum()
    avg_yield = fview["predicted_yield_t_ha"].mean()
    cols = st.columns(5)
    cols[0].metric("Active farms", fmt_int(dview["farmer_id"].nunique()), "+4.8% vs last month")
    cols[1].metric("Area monitored", f"{total_area:,.0f} ha", "+6.2%")
    cols[2].metric("Crops monitored", f"{fview['crop'].nunique()}", "6 crop types")
    cols[3].metric("Predicted yield", f"{avg_yield:.1f} t/ha", "+8.4% seasonal outlook")
    cols[4].metric("Critical signals", fmt_int(high_risk), "Needs field action")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    left, mid, right = st.columns([1.45, 1.15, .85], gap="medium")

    with left:
        st.markdown("<div class='card'><div class='section-title'>📍 Kenya Farm Locations & Field Health</div><div class='section-note'>AI-derived field condition across the six Western Kenya counties in the portfolio: Kakamega, Bungoma, Busia, Vihiga, Trans Nzoia and Siaya.</div></div>", unsafe_allow_html=True)
        mapdf = fview.copy()
        color_map = {"Healthy":"#20a464", "At Risk":"#e9a72f", "Critical":"#df4d4d"}
        fig = px.scatter_geo(
            mapdf, lat="lat", lon="lon", color="health_status", size="area_ha", hover_name="field_id",
            hover_data={"county":True,"crop":True,"health_score":True,"soil_moisture_pct":True,"predicted_yield_t_ha":True,"lat":False,"lon":False},
            color_discrete_map=color_map, scope="world", projection="mercator",
        )
        fig.update_geos(
            showland=True, landcolor="#e9f0e8", showcountries=True, countrycolor="#c5d0d7",
            showlakes=True, lakecolor="#dceef5", showcoastlines=True, coastlinecolor="#c5d0d7",
            fitbounds="locations",
        )
        county_labels = mapdf.groupby("county", as_index=False)[["lat", "lon"]].mean()
        if not county_labels.empty:
            fig.add_trace(go.Scattergeo(
                lon=county_labels["lon"], lat=county_labels["lat"],
                text=county_labels["county"], mode="text",
                textfont=dict(size=10, color="#315064"), hoverinfo="skip", showlegend=False,
            ))
        fig.update_layout(height=420, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h",y=-.02))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with mid:
        st.markdown("<div class='card'><div class='section-title'>🌾 Crop Performance</div><div class='section-note'>Predicted yield against programme benchmarks.</div></div>", unsafe_allow_html=True)
        cp = fview.groupby("crop", as_index=False).agg(predicted=("predicted_yield_t_ha","mean"), health=("health_score","mean"))
        target = {"Maize":3.5,"Beans":1.7,"Sorghum":2.1,"Tomatoes":14,"Kale":10,"Groundnuts":1.5}
        cp["target"] = cp["crop"].map(target)
        fig = go.Figure()
        fig.add_bar(x=cp["crop"], y=cp["predicted"], name="Predicted yield", marker_color="#20a464")
        fig.add_bar(x=cp["crop"], y=cp["target"], name="Target", marker_color="#b8c9d5")
        fig.update_layout(barmode="group", height=340, margin=dict(l=0,r=0,t=10,b=0), yaxis_title="t/ha", legend=dict(orientation="h",y=1.08,x=0), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f"<div class='small'>Average field health: <b>{fview['health_score'].mean():.0f}/100</b> • {healthy:.0%} of monitored area classified healthy.</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='section-title'>🧠 AI Insights</div><div class='section-note'>Prioritised signals for field teams.</div></div>", unsafe_allow_html=True)
        top_crop = cp.sort_values("predicted").iloc[0]["crop"]
        risk_count = int((dview["risk_tier"] != "low").sum())
        insights = [
            ("green", "Maize yield opportunity", f"AI projects {max(0, (cp.loc[cp.crop=='Maize','predicted'].mean() / 3.5 - 1)*100):.0f}% vs benchmark. Maintain scouting and timely top-dressing."),
            ("amber", f"{top_crop} needs attention", "Lower-performing crop blocks should be prioritised for field scouting and agronomy follow-up."),
            ("blue", "Rainfall window", "Moderate rainfall is expected across the western cluster; check drainage on low-lying plots."),
            ("red", f"{risk_count:,} farmers flagged", "AI risk scores indicate a medium/high engagement need. Review explanations before field action."),
        ]
        for tone, title, text in insights:
            klass = "alert-green" if tone=="green" else "alert-amber" if tone=="amber" else "alert-red" if tone=="red" else ""
            st.markdown(f"<div class='insight {klass}'><strong>{title}</strong><p>{text}</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    bottom1, bottom2 = st.columns([1.25, .9], gap="medium")
    with bottom1:
        st.markdown("<div class='card'><div class='section-title'>📋 Priority Fields</div><div class='section-note'>Fields ranked by operational need.</div></div>", unsafe_allow_html=True)
        priority = fview.assign(priority_score=(100-fview["health_score"])+fview["pest_pressure_pct"]*.35).sort_values("priority_score", ascending=False).head(8)
        show = priority[["field_id","county","crop","area_ha","health_score","soil_moisture_pct","pest_pressure_pct","last_scouted_days"]].copy()
        show.columns = ["Field","County","Crop","Area (ha)","Health","Soil moisture","Pest pressure","Days since scout"]
        st.dataframe(show, use_container_width=True, hide_index=True, column_config={"Health":st.column_config.ProgressColumn("Health",min_value=0,max_value=100,format="%d")})
    with bottom2:
        st.markdown("<div class='card'><div class='section-title'>🌦️ 7-Day Farm Weather</div><div class='section-note'>Illustrative western Kenya forecast feed.</div></div>", unsafe_allow_html=True)
        days = pd.date_range("2026-10-06", periods=7)
        rain = [8,14,3,22,18,6,2]
        temps = [27,26,28,25,25,27,29]
        wf = pd.DataFrame({"day":days.strftime("%a"),"rain_mm":rain,"temp":temps})
        fig = go.Figure()
        fig.add_bar(x=wf.day,y=wf.rain_mm,name="Rain (mm)",marker_color="#75b7d4")
        fig.add_scatter(x=wf.day,y=wf.temp,name="Temp °C",mode="lines+markers",line=dict(color="#e9a72f",width=3),yaxis="y2")
        fig.update_layout(height=230,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="white",yaxis=dict(title="mm"),yaxis2=dict(title="°C",overlaying="y",side="right"),legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

elif page == "Fields & Crops":
    st.markdown("<div class='card'><div class='section-title'>🌱 Field & Crop Operations</div><div class='section-note'>Monitor crop health, agronomic signals and expected yield at field level.</div></div>", unsafe_allow_html=True)
    crop = st.selectbox("Crop", ["All"] + sorted(fview.crop.unique().tolist()))
    q = fview if crop == "All" else fview[fview.crop == crop]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fields", len(q)); c2.metric("Area", f"{q.area_ha.sum():,.1f} ha"); c3.metric("Healthy", f"{(q.health_status=='Healthy').mean():.0%}"); c4.metric("Avg yield", f"{q.predicted_yield_t_ha.mean():.1f} t/ha")
    st.dataframe(q[["field_id","county","crop","area_ha","health_status","health_score","soil_moisture_pct","rainfall_mm_7d","pest_pressure_pct","predicted_yield_t_ha","last_scouted_days"]].sort_values("health_score"), use_container_width=True, hide_index=True)
    a,b = st.columns(2)
    with a:
        chart = q.groupby("county",as_index=False).health_score.mean().sort_values("health_score")
        fig=px.bar(chart,x="health_score",y="county",orientation="h",title="Average field health by county",labels={"health_score":"Health score"},color="health_score",color_continuous_scale="Greens")
        fig.update_layout(height=380,margin=dict(l=0,r=0,t=45,b=0)); st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.scatter(q,x="soil_moisture_pct",y="predicted_yield_t_ha",size="area_ha",color="health_status",hover_name="field_id",title="Soil moisture vs predicted yield",color_discrete_map={"Healthy":"#20a464","At Risk":"#e9a72f","Critical":"#df4d4d"})
        fig.update_layout(height=380,margin=dict(l=0,r=0,t=45,b=0)); st.plotly_chart(fig,use_container_width=True)

elif page == "AI Risk":
    st.markdown("<div class='card'><div class='section-title'>🧠 AI Farmer Risk Intelligence</div><div class='section-note'>XGBoost risk scores from the original project, joined to operational farm context. Scores are synthetic.</div></div>", unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("High risk", int((dview.risk_tier=="high").sum()))
    b.metric("Medium risk", int((dview.risk_tier=="medium").sum()))
    c.metric("Low risk", int((dview.risk_tier=="low").sum()))
    x,y=st.columns([1,1])
    with x:
        tier=dview.risk_tier.value_counts().reindex(["low","medium","high"]).fillna(0).reset_index(); tier.columns=["tier","farmers"]
        fig=px.bar(tier,x="tier",y="farmers",color="tier",color_discrete_map={"low":"#20a464","medium":"#e9a72f","high":"#df4d4d"},title="Risk distribution")
        fig.update_layout(height=350); st.plotly_chart(fig,use_container_width=True)
    with y:
        imp=importance.head(8).copy(); imp["feature"]=imp.feature.str.replace("_"," ").str.title()
        fig=px.bar(imp.sort_values("mean_abs_shap"),x="mean_abs_shap",y="feature",orientation="h",title="What drives AI risk",labels={"mean_abs_shap":"Mean |SHAP|"})
        fig.update_layout(height=350); st.plotly_chart(fig,use_container_width=True)
    top=dview.sort_values("risk_score",ascending=False).head(30)[["farmer_id","region","crop","risk_score","risk_tier","top_reasons","days_since_last_farm_visit","input_redemption_rate","group_repayment_avg"]]
    st.dataframe(top,use_container_width=True,hide_index=True,column_config={"risk_score":st.column_config.ProgressColumn("Risk score",min_value=0,max_value=1,format="%.2f")})

elif page == "Market & Weather":
    st.markdown("<div class='card'><div class='section-title'>📈 Market & Weather Signals</div><div class='section-note'>Illustrative market intelligence layer for Kenyan farm planning; values are synthetic portfolio data.</div></div>", unsafe_allow_html=True)
    mcrop=st.selectbox("Market crop",sorted(market.crop.unique().tolist()))
    m=market[market.crop==mcrop].sort_values("price_kes_kg",ascending=False)
    fig=px.bar(m,x="county",y="price_kes_kg",text="price_kes_kg",title=f"Indicative {mcrop} market price by county",labels={"price_kes_kg":"KES/kg"})
    fig.update_traces(texttemplate="KES %{text:.0f}",textposition="outside"); fig.update_layout(height=360,margin=dict(t=55)); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(m.rename(columns={"county":"County","crop":"Crop","price_kes_kg":"Price (KES/kg)","price_change_pct":"7-day change (%)"}),use_container_width=True,hide_index=True)

else:
    st.markdown("<div class='card'><div class='section-title'>⚡ Actions, Alerts & Model Monitoring</div><div class='section-note'>Human-in-the-loop operational queue generated by the original FarmSignal orchestration layer.</div></div>", unsafe_allow_html=True)
    delivered = actions.status.isin(["delivered","created","sent"]).mean()
    a,b,c,d=st.columns(4); a.metric("Actions triggered",fmt_int(len(actions))); b.metric("Delivery / creation",f"{delivered:.1%}"); c.metric("SMS nudges",fmt_int((actions.action_type=="sms_nudge").sum())); d.metric("Escalations",fmt_int((actions.action_type=="supervisor_escalation").sum()))
    funnel=actions.groupby(["risk_tier","action_type"]).size().reset_index(name="count")
    fig=px.bar(funnel,x="risk_tier",y="count",color="action_type",barmode="stack",title="Operational action funnel")
    fig.update_layout(height=350); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(actions.sort_values("timestamp",ascending=False).head(80),use_container_width=True,hide_index=True)

st.markdown("<hr><div class='small'>FarmSignal AI • Portfolio prototype • All farmer, field, market and weather records are synthetic. AI recommendations support — not replace — agronomist and field-officer judgement.</div>", unsafe_allow_html=True)

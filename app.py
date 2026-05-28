# =============================================================================
# IBR 2026 — Awareness of Sustainable Finance & Gen Z Portfolio Preferences
# Author  : Aishwarya Birla | SP Jain School of Global Management
# Dashboard: Streamlit · Plotly · 45 Countries · 7 Variables · TPB Framework
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBR 2026 — Sustainable Finance & Gen Z",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLES ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2736 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }
.kpi-value  { font-size: 2.2rem; font-weight: 800; color: #3fb950; line-height: 1.1; }
.kpi-delta  { font-size: .9rem;  color: #f85149; font-weight: 600; }
.kpi-label  { font-size: .78rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; }

/* ── Section Headers ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    border-bottom: 2px solid #238636;
    padding-bottom: 8px; margin-bottom: 20px;
    font-size: 1.25rem; font-weight: 700; color: #e6edf3;
}

/* ── Insight Box ── */
.insight-box {
    background: #1c2736; border-left: 4px solid #3fb950;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    font-size: .88rem; color: #c9d1d9; margin: 12px 0;
    line-height: 1.6;
}
.insight-box strong { color: #3fb950; }

/* ── Warning Box ── */
.warn-box {
    background: #2d1b00; border-left: 4px solid #d29922;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    font-size: .88rem; color: #e3b341; margin: 12px 0;
}

/* ── Hypothesis Badge ── */
.hyp-supported { background:#1a3a1a; color:#3fb950; border:1px solid #238636; border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:700; }
.hyp-rejected  { background:#3a1a1a; color:#f85149; border:1px solid #da3633; border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:700; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: .9rem; }

/* ── Metric override ── */
[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #3fb950 !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    # Validate numeric columns
    num_cols = ["SFA","FL","EC","PR","EFR","ATT","SIB"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()
VARS        = ["SFA","FL","EC","PR","EFR","ATT","SIB"]
VAR_LABELS  = {
    "SFA":"Sustainable Finance Awareness",
    "FL" :"Financial Literacy",
    "EC" :"Environmental Concern",
    "PR" :"Perceived Risk",
    "EFR":"Expected Financial Return",
    "ATT":"Attitude Toward ESG",
    "SIB":"ESG Investment Behaviour"
}
VAR_SHORT   = {
    "SFA":"SFA (Awareness)","FL":"FL (Literacy)","EC":"EC (Concern)",
    "PR":"PR (Risk)","EFR":"EFR (Return)","ATT":"ATT (Attitude)","SIB":"SIB (Behaviour)"
}
REGIONS     = sorted(df["Region"].dropna().unique().tolist())
INCOME_GRP  = ["High Income","Upper Middle","Lower Middle"]
PALETTE     = px.colors.qualitative.Set2
COLOR_SEQ   = ["#3fb950","#58a6ff","#f78166","#d2a8ff","#ffa657","#79c0ff","#7ee787"]

# Colour map for income groups
INC_COLORS  = {"High Income":"#3fb950","Upper Middle":"#f0a500","Lower Middle":"#f85149"}
REG_COLORS  = {r:PALETTE[i % len(PALETTE)] for i,r in enumerate(REGIONS)}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌱 IBR 2026 Dashboard")
    st.markdown("**Sustainable Finance & Gen Z**")
    st.markdown("---")

    st.markdown("#### 🔍 Filters")
    sel_regions = st.multiselect(
        "Region", REGIONS, default=REGIONS, key="reg_filter"
    )
    sel_income = st.multiselect(
        "Income Group", INCOME_GRP, default=INCOME_GRP, key="inc_filter"
    )

    df_f = df[df["Region"].isin(sel_regions) & df["Income_Group"].isin(sel_income)].copy()
    n_countries = len(df_f)

    st.markdown(f"**Countries selected:** `{n_countries}` / 45")
    st.markdown("---")
    st.markdown("#### 📌 About")
    st.markdown("""
    - **Researcher:** Aishwarya Birla  
    - **Institution:** SP Jain School of Global Management  
    - **Year:** IBR 2026  
    - **Framework:** Theory of Planned Behaviour  
    - **Scale:** 1–5 (all variables)  
    - **Method:** OLS + PLS-SEM  
    """)
    st.markdown("---")
    st.caption("Sources: Morgan Stanley 2023 · OECD/INFE 2022 · GSIA 2022 · CFA Institute 2022 · Eurobarometer 2023 · Pew Research 2022")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1117 0%,#1a2332 50%,#0d1117 100%);
            border:1px solid #238636; border-radius:16px; padding:28px 32px; margin-bottom:24px;">
  <div style="font-size:.85rem;color:#8b949e;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px;">
    SP Jain School of Global Management · IBR 2026
  </div>
  <h1 style="color:#e6edf3;margin:0;font-size:2rem;font-weight:800;line-height:1.2;">
    🌱 Awareness of Sustainable Finance
  </h1>
  <h2 style="color:#3fb950;margin:4px 0 12px;font-size:1.25rem;font-weight:600;">
    & Its Impact on Portfolio Preferences of Generation Z
  </h2>
  <div style="display:flex;gap:20px;flex-wrap:wrap;">
    <span style="background:#1c2736;border:1px solid #30363d;border-radius:20px;padding:4px 14px;font-size:.8rem;color:#c9d1d9;">
      📊 45 Countries · 8 Regions
    </span>
    <span style="background:#1c2736;border:1px solid #30363d;border-radius:20px;padding:4px 14px;font-size:.8rem;color:#c9d1d9;">
      🔬 7 Variables · 1–5 Scale
    </span>
    <span style="background:#1c2736;border:1px solid #30363d;border-radius:20px;padding:4px 14px;font-size:.8rem;color:#c9d1d9;">
      📐 OLS + PLS-SEM · 5,000 Bootstraps
    </span>
    <span style="background:#1a3a1a;border:1px solid #238636;border-radius:20px;padding:4px 14px;font-size:.8rem;color:#3fb950;font-weight:700;">
      R² = 0.993 Model Fit
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Executive Summary",
    "🌍 Global Map",
    "📊 Variable Analysis",
    "🏢 Regional & Income",
    "🔗 Correlation & Regression",
    "🧪 PLS-SEM Results",
    "⚖️ Country Comparison",
    "🔎 Data Quality",
    "💡 Key Findings",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">📋 Executive Summary — Research at a Glance</div>', unsafe_allow_html=True)

    # KPI Row 1
    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [
        (k1, "R² = 0.993", "99.3% Variance Explained", None),
        (k2, "β = 0.670",  "ATT → SIB (Dominant Predictor)", None),
        (k3, "2.09 pts",   "Behavioural Gap (EC−SIB)", "↑ EC=3.70 vs SIB=1.61"),
        (k4, "4 of 6",     "Hypotheses Supported", None),
        (k5, "45",         "Countries · Zero Missing Values", None),
    ]
    for col, val, label, delta in kpis:
        with col:
            d_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-value">{val}</div>
              {d_html}
              <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Story + Key stats
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("#### 🔬 The Research Story")
        st.markdown("""
        <div class="insight-box">
        Gen Z globally demonstrates a <strong>2.09-point behavioural gap</strong> between environmental 
        concern (EC = 3.70/5) and actual ESG investing (SIB = 1.61/5). This 45-country study identifies 
        <strong>attitude toward sustainable investment as the master switch</strong> — shaped by expected 
        financial returns (β = 0.756) and suppressed by risk fear (β = −0.414) — while confirming that 
        general financial literacy has <strong>no independent effect</strong> on behaviour.
        </div>
        """, unsafe_allow_html=True)

        findings = [
            ("🎯 Attitude is the Master Switch", "ATT → SIB: β=0.670, p<0.001. The strongest direct predictor. Positive emotional alignment with ESG must precede financial action."),
            ("💰 Expected Returns Unlock Attitude", "EFR → ATT: β=0.756, p<0.001. Sell ESG on financial performance first — return confidence shapes attitude more than any other factor."),
            ("⚠️ Risk Perception is the #1 Barrier", "PR → ATT: β=−0.414, p=0.016. Risk fear destroys attitude before investing can happen. Dismantle the myth with volatility data."),
            ("📚 Financial Literacy ≠ ESG Action", "FL: not significant (p=0.207). Generic financial education doesn't drive ESG investing. Domain-specific literacy is the real gap."),
        ]
        for title, body in findings:
            with st.expander(title):
                st.write(body)

    with col_r:
        st.markdown("#### 📊 Mean Scores — All 7 Variables")
        means = {VAR_SHORT[v]: round(df_f[v].mean(), 3) for v in VARS}
        fig_means = go.Figure(go.Bar(
            x=list(means.values()),
            y=list(means.keys()),
            orientation='h',
            marker=dict(
                color=list(means.values()),
                colorscale=[[0,"#f85149"],[0.5,"#f0a500"],[1,"#3fb950"]],
                showscale=False,
            ),
            text=[f"{v:.2f}" for v in means.values()],
            textposition='outside',
        ))
        fig_means.add_vline(x=2.5, line_dash="dash", line_color="#8b949e", annotation_text="Midpoint 2.5")
        fig_means.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=340,
            margin=dict(l=10,r=60,t=10,b=10),
            xaxis=dict(range=[0,5.2], title="Score (1–5)"),
            yaxis=dict(title=""),
            font=dict(size=12),
        )
        st.plotly_chart(fig_means, use_container_width=True)

        # Behavioral gap callout
        ec_mean = df_f["EC"].mean()
        sib_mean = df_f["SIB"].mean()
        gap = ec_mean - sib_mean
        st.markdown(f"""
        <div style="background:#2d1b00;border:1px solid #d29922;border-radius:10px;padding:14px 18px;text-align:center;">
          <div style="font-size:2.2rem;font-weight:800;color:#f0a500;">{gap:.2f} pts</div>
          <div style="font-size:.8rem;color:#e3b341;">Behavioural Gap (EC {ec_mean:.2f} − SIB {sib_mean:.2f})</div>
          <div style="font-size:.75rem;color:#8b949e;margin-top:4px;">Gen Z cares deeply · but doesn't invest</div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom summary stats table
    st.markdown("---")
    st.markdown("#### 📐 Descriptive Statistics — All 7 Variables")
    desc = df_f[VARS].describe().T.round(3)
    desc.index = [VAR_LABELS[v] for v in VARS]
    desc = desc[["mean","std","min","25%","50%","75%","max"]]
    desc.columns = ["Mean","Std Dev","Min","Q1","Median","Q3","Max"]
    st.dataframe(desc.style.background_gradient(subset=["Mean"], cmap="Greens"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GLOBAL MAP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">🌍 Global Choropleth — Country-Level ESG Landscape</div>', unsafe_allow_html=True)

    map_var = st.selectbox(
        "Variable to map", VARS,
        format_func=lambda v: f"{v} — {VAR_LABELS[v]}",
        key="map_var"
    )

    fig_map = px.choropleth(
        df_f, locations="Country", locationmode="country names",
        color=map_var,
        color_continuous_scale=["#f85149","#ff8700","#f0a500","#3fb950","#238636"],
        range_color=[1,5],
        hover_name="Country",
        hover_data={"Region":True,"Income_Group":True,**{v:True for v in VARS}},
        labels={map_var: VAR_LABELS[map_var]},
        title=f"Global Distribution: {VAR_LABELS[map_var]} (1–5 Scale)",
    )
    fig_map.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=520,
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
                 coastlinecolor="#30363d", showland=True, landcolor="#161b22",
                 showocean=True, oceancolor="#0d1117"),
        coloraxis_colorbar=dict(title="Score", tickvals=[1,2,3,4,5]),
        margin=dict(t=50,b=0,l=0,r=0),
        font=dict(color="#e6edf3"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top & bottom 5
    col_t, col_b = st.columns(2)
    with col_t:
        st.markdown(f"**🏆 Top 5 Countries — {map_var}**")
        top5 = df_f.nlargest(5, map_var)[["Country","Region","Income_Group",map_var]].round(3)
        top5.columns = ["Country","Region","Income Group",VAR_LABELS[map_var]]
        st.dataframe(top5.reset_index(drop=True), use_container_width=True)
    with col_b:
        st.markdown(f"**📉 Bottom 5 Countries — {map_var}**")
        bot5 = df_f.nsmallest(5, map_var)[["Country","Region","Income_Group",map_var]].round(3)
        bot5.columns = ["Country","Region","Income Group",VAR_LABELS[map_var]]
        st.dataframe(bot5.reset_index(drop=True), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Map Reading:</strong> Darker green = higher score. For risk (PR), higher scores = more risk aversion.
    Notice how the North America + Northern Europe cluster dominates ESG Behaviour (SIB), while South Asia and Africa 
    show structural investment barriers despite moderate Environmental Concern (EC).
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VARIABLE ANALYSIS (Country Explorer)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">📊 Interactive Country-Level Variable Explorer</div>', unsafe_allow_html=True)

    col_v1, col_v2 = st.columns([1,2])
    with col_v1:
        sel_var = st.selectbox(
            "Primary variable", VARS,
            format_func=lambda v: f"{v} — {VAR_LABELS[v]}",
            index=6, key="var_sel"
        )
        chart_type = st.radio("Chart type", ["Bar (ranked)","Scatter vs SIB","Bubble Chart"], key="chart_type")

    with col_v2:
        if chart_type == "Bar (ranked)":
            df_sorted = df_f.sort_values(sel_var, ascending=True).copy()
            fig_bar = px.bar(
                df_sorted, y="Country", x=sel_var,
                color="Income_Group", color_discrete_map=INC_COLORS,
                orientation='h',
                labels={sel_var: VAR_LABELS[sel_var]},
                title=f"{sel_var} — {VAR_LABELS[sel_var]} by Country",
                text=sel_var,
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_bar.add_vline(x=2.5, line_dash="dash", line_color="#8b949e")
            fig_bar.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=900,
                margin=dict(l=0,r=60,t=40,b=0),
                legend=dict(orientation="h", y=1.02),
                font=dict(size=11),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        elif chart_type == "Scatter vs SIB":
            if sel_var == "SIB":
                st.info("ℹ️ Select a different variable to plot against SIB. Currently showing ATT vs SIB.")
                x_var = "ATT"
            else:
                x_var = sel_var
            hover_extra = {"Income_Group": True, "SIB": True}
            if x_var != "SIB":
                hover_extra[x_var] = True
            fig_sc = px.scatter(
                df_f, x=x_var, y="SIB",
                color="Region", size_max=16,
                color_discrete_sequence=PALETTE,
                hover_name="Country",
                hover_data=hover_extra,
                labels={x_var: VAR_LABELS[x_var], "SIB":"ESG Investment Behaviour"},
                title=f"{VAR_LABELS[x_var]} vs ESG Investment Behaviour",
                trendline="ols",
                trendline_color_override="#f0a500",
            )
            fig_sc.update_traces(marker=dict(size=12, opacity=0.85))
            fig_sc.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=500,
                font=dict(size=12),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        else:  # Bubble
            fig_bub = px.scatter(
                df_f, x=sel_var, y="SIB", size="EC",
                color="Income_Group", color_discrete_map=INC_COLORS,
                hover_name="Country",
                labels={sel_var: VAR_LABELS[sel_var], "SIB":"ESG Behaviour","EC":"Env. Concern (bubble size)"},
                title=f"Bubble: {VAR_LABELS[sel_var]} vs SIB — bubble = Environmental Concern",
            )
            fig_bub.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=500, font=dict(size=12),
            )
            st.plotly_chart(fig_bub, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Full Data Table")
    df_show = df_f[["No","Country","Region","Income_Group"]+VARS].copy()
    df_show.columns = ["#","Country","Region","Income Group","SFA (Awareness)","FL (Literacy)",
                       "EC (Concern)","PR (Risk)","EFR (Return)","ATT (Attitude)","SIB (Behaviour)"]
    st.dataframe(
        df_show.style.background_gradient(
            subset=["SFA (Awareness)","FL (Literacy)","EC (Concern)","ATT (Attitude)","SIB (Behaviour)"],
            cmap="Greens"
        ).background_gradient(subset=["PR (Risk)"], cmap="Reds_r"),
        use_container_width=True, height=350
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGIONAL & INCOME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">🏢 Regional & Income Group Analysis</div>', unsafe_allow_html=True)

    subtab_r, subtab_i = st.tabs(["🌐 Regional Analysis", "💰 Income Group Analysis"])

    # ── REGIONAL ──
    with subtab_r:
        reg_means = df_f.groupby("Region")[VARS].mean().round(3).reset_index()

        # Grouped bar: SFA vs SIB gap by region
        fig_reg_gap = go.Figure()
        for v, c, name in [("SFA","#58a6ff","SFA (Awareness)"),("SIB","#3fb950","SIB (Behaviour)"),("EC","#d2a8ff","EC (Concern)")]:
            fig_reg_gap.add_trace(go.Bar(
                name=name, x=reg_means["Region"], y=reg_means[v],
                marker_color=c, opacity=0.88,
                text=reg_means[v].round(2), textposition='outside',
            ))
        fig_reg_gap.update_layout(
            barmode='group', template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=420, title="Behavioural Gap by Region: Awareness vs Concern vs Actual Investing",
            xaxis_tickangle=-25, yaxis=dict(range=[0,5.5], title="Score (1–5)"),
            legend=dict(orientation="h", y=1.08), font=dict(size=12),
            margin=dict(t=60,b=80),
        )
        st.plotly_chart(fig_reg_gap, use_container_width=True)

        # Gap column
        reg_means["Behavioural_Gap"] = (reg_means["EC"] - reg_means["SIB"]).round(3)

        col_r1, col_r2 = st.columns([1.4, 1])
        with col_r1:
            # Radar chart by region — top 5 variables
            fig_radar = go.Figure()
            r_vars  = ["SFA","FL","EC","EFR","ATT","SIB"]
            r_labels = [VAR_SHORT[v] for v in r_vars] + [VAR_SHORT[r_vars[0]]]
            for _, row in reg_means.iterrows():
                vals = [row[v] for v in r_vars] + [row[r_vars[0]]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=r_labels, fill='toself', name=row["Region"], opacity=0.65,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,5])),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=400, title="Radar: All Variables by Region",
                font=dict(size=11), legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_r2:
            st.markdown("**📊 Regional Summary Table**")
            reg_show = reg_means[["Region","SFA","FL","EC","PR","EFR","ATT","SIB","Behavioural_Gap"]].copy()
            reg_show = reg_show.sort_values("SIB", ascending=False)
            reg_show.columns = ["Region","SFA","FL","EC","PR","EFR","ATT","SIB","Beh. Gap"]
            st.dataframe(
                reg_show.style.background_gradient(subset=["SIB","ATT","EFR"], cmap="Greens")
                              .background_gradient(subset=["PR","Beh. Gap"], cmap="Oranges"),
                use_container_width=True
            )
            st.markdown("""
            <div class="insight-box">
            <strong>North America</strong> leads in SIB (≈2.25) driven by mature ESG product ecosystems.
            <strong>S. Asia & Africa</strong> show the lowest SIB (≈0.76) despite moderate concern — 
            structural access barriers dominate over attitude.
            </div>
            """, unsafe_allow_html=True)

    # ── INCOME GROUP ──
    with subtab_i:
        inc_means = df_f.groupby("Income_Group")[VARS].mean().round(3).reset_index()
        # Enforce correct ordering
        inc_order = [g for g in INCOME_GRP if g in inc_means["Income_Group"].values]
        inc_means["Income_Group"] = pd.Categorical(inc_means["Income_Group"], categories=inc_order, ordered=True)
        inc_means = inc_means.sort_values("Income_Group")

        fig_inc = make_subplots(
            rows=2, cols=4,
            subplot_titles=[f"{VAR_LABELS[v]}" for v in VARS],
            vertical_spacing=0.18, horizontal_spacing=0.08,
        )
        for idx, v in enumerate(VARS):
            r, c = divmod(idx, 4)
            r += 1; c += 1
            for g in inc_order:
                row_data = inc_means[inc_means["Income_Group"]==g]
                if not row_data.empty:
                    fig_inc.add_trace(
                        go.Bar(
                            name=g, x=[g], y=[row_data[v].values[0]],
                            marker_color=INC_COLORS.get(g,"#888"),
                            showlegend=(idx==0),
                            text=[f"{row_data[v].values[0]:.2f}"],
                            textposition='outside',
                        ),
                        row=r, col=c
                    )

        fig_inc.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=540,
            title_text="All 7 Variables by Income Group",
            barmode='group', font=dict(size=10),
            legend=dict(orientation="h", y=1.06),
            margin=dict(t=80, b=20),
        )
        for i in range(1,9):
            try:
                fig_inc.update_yaxes(range=[0,5.5], row=(i-1)//4+1, col=(i-1)%4+1)
            except:
                pass
        st.plotly_chart(fig_inc, use_container_width=True)

        # Trend lines
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            fig_sib_inc = px.line(
                inc_means, x="Income_Group", y="SIB",
                markers=True, color_discrete_sequence=["#3fb950"],
                title="SIB drops sharply with income",
                labels={"SIB":"ESG Investment Behaviour","Income_Group":""},
            )
            fig_sib_inc.update_traces(line_width=3, marker_size=12)
            fig_sib_inc.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=50,b=10),
            )
            st.plotly_chart(fig_sib_inc, use_container_width=True)

        with col_i2:
            fig_fl_inc = px.line(
                inc_means, x="Income_Group", y="FL",
                markers=True, color_discrete_sequence=["#58a6ff"],
                title="FL most income-correlated variable",
                labels={"FL":"Financial Literacy","Income_Group":""},
            )
            fig_fl_inc.update_traces(line_width=3, marker_size=12)
            fig_fl_inc.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=50,b=10),
            )
            st.plotly_chart(fig_fl_inc, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Key Income Insight:</strong> SIB drops from 1.95 (High Income) → 1.21 (Upper Middle) → 0.81 (Lower Middle). 
        EC stays relatively stable across groups (3.88→3.22) — concern is near-universal. 
        PR rises as income falls (2.84→3.62), confirming risk aversion is highest where financial systems are least developed.
        FL is the most wealth-linked variable (3.06→1.76), pointing to a structural financial education gap.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CORRELATION & OLS REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🔗 Correlation Heatmap & OLS Regression</div>', unsafe_allow_html=True)

    subtab_c, subtab_o = st.tabs(["📊 Correlation Heatmap", "📐 OLS Regression"])

    with subtab_c:
        corr = df_f[VARS].corr().round(3)
        labels_short = [VAR_SHORT[v] for v in VARS]

        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=labels_short, y=labels_short,
            colorscale=[[0,"#f85149"],[0.5,"#e6edf3"],[1,"#3fb950"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr.values.round(3),
            texttemplate="%{text}",
            textfont=dict(size=12, color="black"),
            hoverongaps=False,
        ))
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=480,
            title="Pearson Correlation Matrix — n=45 Countries (all p<0.001)",
            font=dict(size=12),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="warn-box">
        ⚠️ <strong>Multicollinearity Detected:</strong> IVs inter-correlate at r = 0.877–0.993. 
        This inflates OLS standard errors, causing H2–H5 to appear insignificant in OLS. 
        PLS-SEM was specifically selected to resolve individual path contributions — see the PLS-SEM tab.
        </div>
        """, unsafe_allow_html=True)

        # Highlight correlations with SIB
        st.markdown("**Correlation with ESG Investment Behaviour (SIB)**")
        sib_corr = corr["SIB"].drop("SIB").sort_values(key=abs, ascending=False)
        fig_sib_corr = go.Figure(go.Bar(
            x=sib_corr.index,
            y=sib_corr.values,
            marker_color=["#f85149" if v<0 else "#3fb950" for v in sib_corr.values],
            text=sib_corr.values.round(3), textposition='outside',
        ))
        fig_sib_corr.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            yaxis=dict(range=[-1.1,1.1], title="Pearson r"),
            title="Pearson r with SIB (ESG Behaviour) — all significant at p<0.001",
            margin=dict(t=50,b=10), font=dict(size=12),
        )
        st.plotly_chart(fig_sib_corr, use_container_width=True)

    with subtab_o:
        st.markdown("#### OLS Regression — SIB = β₀ + β₁SFA + β₂FL + β₃EC + β₄PR + β₅EFR + β₆ATT + ε")
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;
                    font-family:monospace;font-size:.88rem;color:#3fb950;margin-bottom:16px;">
        SIB = −1.70 + 0.247(SFA) − 0.157(FL) − 0.083(EC) − 0.207(PR) + 0.173(EFR) + 0.929(ATT)
        </div>
        """, unsafe_allow_html=True)

        ols_data = {
            "Variable": ["Intercept","SFA — Awareness","FL — Literacy","EC — Concern","PR — Risk","EFR — Return","ATT — Attitude"],
            "β Coeff.": ["-1.70","0.247","-0.157","-0.083","-0.207","0.173","0.929"],
            "t-stat":   ["—","3.044","-1.563","-1.369","-0.952","0.770","4.914"],
            "p-value":  ["—","0.004","0.126","0.179","0.347","0.446","<0.001"],
            "Significant?": ["—","✅ Yes","❌ No","❌ No","❌ No","❌ No","✅ Yes"],
            "Note": ["","H1 — Direct confirmed","Multicollinear suppresses","Multicollinear suppresses","Multicollinear suppresses","Multicollinear suppresses","Dominant predictor"],
        }
        ols_df = pd.DataFrame(ols_data)
        st.dataframe(ols_df, use_container_width=True, hide_index=True)

        col_ols1, col_ols2 = st.columns(2)
        with col_ols1:
            st.markdown("""
            <div style="background:#1c2736;border-radius:10px;padding:16px;">
              <div style="font-size:1.1rem;color:#8b949e;">Model Fit</div>
              <div style="font-size:2rem;font-weight:800;color:#3fb950;">R² = 0.993</div>
              <div style="color:#8b949e;font-size:.85rem;">99.3% of SIB variance explained</div>
            </div>""", unsafe_allow_html=True)
        with col_ols2:
            st.markdown("""
            <div style="background:#1c2736;border-radius:10px;padding:16px;">
              <div style="font-size:1.1rem;color:#8b949e;">F-Statistic</div>
              <div style="font-size:2rem;font-weight:800;color:#58a6ff;">963.73</div>
              <div style="color:#8b949e;font-size:.85rem;">p < 0.001 — Overall model significant</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Why do H2–H5 appear insignificant in OLS?</strong> The "Witnesses Problem": when 5 variables give 
        nearly identical testimony (r = 0.877–0.993), OLS cannot assign individual credit — standard errors inflate. 
        Only SFA (β=0.247, p=0.004) and ATT (β=0.929, p&lt;0.001) survive. 
        PLS-SEM algorithmically decomposes shared variance to resolve this.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PLS-SEM RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">🧪 PLS-SEM Results — SmartPLS 4.1.1.7 · 5,000 Bootstraps</div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("#### 📊 Direct Path Coefficients")
        direct_data = {
            "Path":["ATT → SIB","EFR → ATT","EC → ATT","PR → ATT","SFA → SIB","SFA → ATT","FL → SIB","PR → SIB"],
            "β":   [0.670, 0.756, 0.137,-0.414, 0.285,-0.194,-0.163,-0.133],
            "t":   [5.354, 4.682, 3.168, 2.413, 2.381, 1.856, 1.261, 1.137],
            "p":   ["0.000","0.000","0.002","0.016","0.017","0.064","0.207","0.256"],
            "Sig": ["✅","✅","✅","✅","✅","❌","❌","❌"],
        }
        direct_df = pd.DataFrame(direct_data)
        st.dataframe(direct_df, use_container_width=True, hide_index=True)

        st.markdown("#### 🔄 Indirect Effects — Full Mediation via ATT")
        indirect_data = {
            "Indirect Path": ["EFR → ATT → SIB","EC → ATT → SIB","PR → ATT → SIB","SFA → ATT → SIB","FL → ATT → SIB"],
            "β":             [0.507,  0.092, -0.278, -0.130, -0.073],
            "p":             ["0.001","0.005","0.022","0.105","0.416"],
            "Mediation":     ["✅ Full","✅ Full","✅ Full","❌ None","❌ None"],
        }
        indirect_df = pd.DataFrame(indirect_data)
        st.dataframe(indirect_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Full Mediation Confirmed:</strong> EC, EFR, and PR do NOT directly change ESG investing — 
        they first shape <strong>attitude (ATT)</strong>, then attitude drives behaviour (SIB). 
        SFA is the exception — it has a direct path to SIB, bypassing attitude entirely.
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown("#### 🗺️ PLS-SEM Path Model (Visual)")

        # Build path diagram using a Sankey-like chart
        node_labels = ["SFA","FL","EC","PR","EFR","ATT","SIB"]
        node_colors = ["#58a6ff","#58a6ff","#58a6ff","#f85149","#58a6ff","#f0a500","#3fb950"]

        # Edges with β
        edges = [
            (0,5, 0.194,"neg"),  # SFA→ATT (ns, neg in PLS)
            (0,6, 0.285,"pos"),  # SFA→SIB
            (1,6,-0.163,"neg"),  # FL→SIB  (ns)
            (2,5, 0.137,"pos"),  # EC→ATT
            (3,5,-0.414,"neg"),  # PR→ATT
            (4,5, 0.756,"pos"),  # EFR→ATT
            (5,6, 0.670,"pos"),  # ATT→SIB
        ]

        # Sankey diagram
        sources = [e[0] for e in edges]
        targets = [e[1] for e in edges]
        values  = [abs(e[2])*10 for e in edges]
        e_colors= ["rgba(248,81,73,0.6)" if e[3]=="neg" else "rgba(63,185,80,0.6)" for e in edges]
        e_labels= [f"β={e[2]:+.3f}" for e in edges]

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20,
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources, target=targets,
                value=values,
                color=e_colors,
                label=e_labels,
            )
        ))
        fig_sankey.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=400, title="TPB Path Model — β Coefficients (width ∝ |β|)",
            font=dict(size=12, color="#e6edf3"),
            margin=dict(t=50,b=10),
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

        # R² boxes
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""
            <div style="background:#1a3a1a;border:1px solid #238636;border-radius:10px;padding:16px;text-align:center;">
              <div style="font-size:1.8rem;font-weight:800;color:#3fb950;">R² = 0.993</div>
              <div style="color:#8b949e;font-size:.8rem;">SIB — 99.3% explained</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown("""
            <div style="background:#1a3a1a;border:1px solid #238636;border-radius:10px;padding:16px;text-align:center;">
              <div style="font-size:1.8rem;font-weight:800;color:#3fb950;">R² = 0.991</div>
              <div style="color:#8b949e;font-size:.8rem;">ATT — 99.1% explained</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Hypothesis Outcomes — Final Verdict")

    hyp_data = [
        ("H1","SFA → SIB","+","✅ Supported","β=+0.285, p=0.017 (PLS direct). Only IV with a significant direct path to SIB."),
        ("H2","FL → SIB", "+","❌ Not Supported","β=−0.163, p=0.207 (direct); indirect p=0.416 (NS). Generic literacy ≠ ESG investing."),
        ("H3","EC → SIB", "+","✅ Via ATT","Direct NS. Indirect via ATT: β=+0.092, p=0.005. Full mediation confirmed."),
        ("H4","PR → SIB", "−","✅ Via ATT","Direct NS (p=0.256). Indirect: β=−0.278, p=0.022. Risk fear suppresses attitude."),
        ("H5","EFR → SIB","+","✅ Via ATT","Direct NS (p=0.464). Indirect: β=+0.507, p=0.001. Strongest indirect effect."),
        ("H6","ATT→SIB",  "Med","✅ Confirmed","β=+0.670, p<0.001. Full mediation for EC, EFR, PR. TPB empirically validated."),
    ]

    cols = st.columns(3)
    for i, (h, path, dir_, result, note) in enumerate(hyp_data):
        badge_cls = "hyp-supported" if "✅" in result else "hyp-rejected"
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-weight:700;color:#e6edf3;font-size:1rem;">{h}: {path}</span>
                <span class="{badge_cls}">{result}</span>
              </div>
              <div style="font-size:.82rem;color:#8b949e;">{note}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — COUNTRY COMPARISON TOOL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">⚖️ Country-to-Country Comparison Tool</div>', unsafe_allow_html=True)

    all_countries = sorted(df["Country"].tolist())

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        country_a = st.selectbox("Country A", all_countries, index=0, key="cmp_a")
    with col_s2:
        country_b = st.selectbox("Country B", all_countries, index=1, key="cmp_b")

    row_a = df[df["Country"] == country_a].iloc[0]
    row_b = df[df["Country"] == country_b].iloc[0]

    # Radar comparison
    r_v = VARS
    vals_a = [row_a[v] for v in r_v] + [row_a[r_v[0]]]
    vals_b = [row_b[v] for v in r_v] + [row_b[r_v[0]]]
    r_theta = [VAR_SHORT[v] for v in r_v] + [VAR_SHORT[r_v[0]]]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatterpolar(r=vals_a, theta=r_theta, fill='toself', name=country_a, line_color="#3fb950", opacity=0.75))
    fig_cmp.add_trace(go.Scatterpolar(r=vals_b, theta=r_theta, fill='toself', name=country_b, line_color="#58a6ff", opacity=0.75))
    fig_cmp.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,5])),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        height=430, title=f"Radar Comparison: {country_a} vs {country_b}",
        font=dict(size=12), legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Metric table
    col_m0, col_m1, col_m2, col_m3 = st.columns([1.2,1,1,1])
    with col_m0: st.markdown("**Variable**")
    with col_m1: st.markdown(f"**{country_a}**")
    with col_m2: st.markdown(f"**{country_b}**")
    with col_m3: st.markdown("**Δ (A−B)**")

    for v in VARS:
        va, vb = row_a[v], row_b[v]
        delta = va - vb
        delta_color = "#3fb950" if delta > 0 else "#f85149"
        col_m0, col_m1, col_m2, col_m3 = st.columns([1.2,1,1,1])
        with col_m0: st.write(VAR_LABELS[v])
        with col_m1: st.write(f"{va:.2f}")
        with col_m2: st.write(f"{vb:.2f}")
        with col_m3:
            st.markdown(f"<span style='color:{delta_color};font-weight:700;'>{delta:+.2f}</span>", unsafe_allow_html=True)

    # Context cards
    st.markdown("---")
    c1, c2 = st.columns(2)
    for col, row, country in [(c1, row_a, country_a),(c2, row_b, country_b)]:
        with col:
            st.markdown(f"""
            <div style="background:#1c2736;border:1px solid #30363d;border-radius:10px;padding:16px;">
              <div style="font-size:1.15rem;font-weight:700;color:#e6edf3;margin-bottom:8px;">🏳️ {country}</div>
              <div style="font-size:.85rem;color:#8b949e;">
                <b>Region:</b> {row['Region']}<br>
                <b>Income Group:</b> {row['Income_Group']}<br>
                <b>SIB (Behaviour):</b> {row['SIB']:.2f} / 5.0<br>
                <b>ATT (Attitude):</b> {row['ATT']:.2f} / 5.0<br>
                <b>Beh. Gap (EC−SIB):</b> {(row['EC']-row['SIB']):.2f} pts
              </div>
            </div>""", unsafe_allow_html=True)

    # Multi-country comparison
    st.markdown("---")
    st.markdown("#### 📊 Multi-Country Bar Comparison")
    multi_countries = st.multiselect(
        "Select up to 10 countries", all_countries,
        default=[country_a, country_b], max_selections=10, key="multi_cmp"
    )
    if multi_countries:
        df_multi = df[df["Country"].isin(multi_countries)].melt(
            id_vars=["Country","Region","Income_Group"],
            value_vars=VARS, var_name="Variable", value_name="Score"
        )
        df_multi["Variable"] = df_multi["Variable"].map(VAR_SHORT)
        fig_multi = px.bar(
            df_multi, x="Variable", y="Score", color="Country",
            barmode="group", title="Multi-Country Variable Comparison",
            labels={"Score":"Score (1–5)"},
        )
        fig_multi.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=400,
            font=dict(size=11), legend=dict(orientation="h", y=1.08),
            margin=dict(t=60, b=10),
        )
        st.plotly_chart(fig_multi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">🔎 Data Quality & Completeness Panel</div>', unsafe_allow_html=True)

    col_q1, col_q2, col_q3 = st.columns(3)
    total_cells = len(df) * len(VARS)
    missing_cells = df[VARS].isna().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100

    with col_q1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{completeness:.1f}%</div>
          <div class="kpi-label">Data Completeness</div>
        </div>""", unsafe_allow_html=True)
    with col_q2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{len(df)}</div>
          <div class="kpi-label">Total Countries</div>
        </div>""", unsafe_allow_html=True)
    with col_q3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{len(VARS)}</div>
          <div class="kpi-label">Variables · All 1–5 Scale</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_dq1, col_dq2 = st.columns(2)

    with col_dq1:
        st.markdown("#### Missing Values per Variable")
        missing_data = df[VARS].isna().sum().reset_index()
        missing_data.columns = ["Variable","Missing"]
        missing_data["Complete"] = len(df) - missing_data["Missing"]
        missing_data["Variable"] = missing_data["Variable"].map(VAR_LABELS)
        fig_miss = px.bar(
            missing_data, x="Variable", y=["Complete","Missing"],
            color_discrete_map={"Complete":"#3fb950","Missing":"#f85149"},
            labels={"value":"Countries","variable":"Status"},
            title="Data Completeness by Variable",
        )
        fig_miss.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=320, xaxis_tickangle=-20,
            barmode="stack", font=dict(size=11), margin=dict(t=50,b=60),
        )
        st.plotly_chart(fig_miss, use_container_width=True)

    with col_dq2:
        st.markdown("#### Value Range Validation (1–5 Scale)")
        range_check = []
        for v in VARS:
            col_data = df[v].dropna()
            range_check.append({
                "Variable": VAR_LABELS[v],
                "Min": col_data.min(),
                "Max": col_data.max(),
                "In Range?": "✅" if col_data.min() >= 0 and col_data.max() <= 5 else "❌",
                "Missing": df[v].isna().sum(),
            })
        range_df = pd.DataFrame(range_check)
        st.dataframe(range_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📦 Variable Distribution (Box Plots)")
    box_data = df[VARS].melt(var_name="Variable", value_name="Score")
    box_data["Variable"] = box_data["Variable"].map(VAR_SHORT)
    fig_box = px.box(
        box_data, x="Variable", y="Score",
        color="Variable",
        color_discrete_sequence=COLOR_SEQ,
        title="Distribution of All 7 Variables (n=45, Scale 1–5)",
        labels={"Score":"Score (1–5)"},
        points="all",
    )
    fig_box.add_hline(y=2.5, line_dash="dash", line_color="#8b949e", annotation_text="Scale Midpoint 2.5")
    fig_box.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=380,
        showlegend=False, font=dict(size=12), margin=dict(t=50,b=10),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### 🗃️ Region & Income Group Distribution")
    col_dq3, col_dq4 = st.columns(2)
    with col_dq3:
        reg_count = df.groupby("Region").size().reset_index(name="Count")
        fig_reg_pie = px.pie(reg_count, names="Region", values="Count",
                            color_discrete_sequence=PALETTE,
                            title="Countries by Region", hole=0.35)
        fig_reg_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, font=dict(size=11))
        st.plotly_chart(fig_reg_pie, use_container_width=True)

    with col_dq4:
        inc_count = df.groupby("Income_Group").size().reset_index(name="Count")
        fig_inc_pie = px.pie(inc_count, names="Income_Group", values="Count",
                            color_discrete_map=INC_COLORS,
                            title="Countries by Income Group", hole=0.35)
        fig_inc_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, font=dict(size=11))
        st.plotly_chart(fig_inc_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — KEY FINDINGS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-header">💡 Key Findings, Insights & Strategic Recommendations</div>', unsafe_allow_html=True)

    # Paradox highlight
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a2332 0%,#2d1b00 100%);
                border:2px solid #d29922;border-radius:14px;padding:22px 28px;margin-bottom:20px;">
      <div style="font-size:1.5rem;font-weight:800;color:#f0a500;margin-bottom:8px;">
        💥 The $30.3 Trillion Paradox
      </div>
      <div style="color:#e3b341;font-size:1rem;">
        <strong>85% of Gen Z</strong> say they want to invest sustainably. 
        Yet only <strong>31%</strong> actually hold ESG products. 
        This 54-percentage-point intention-action gap is the central puzzle — 
        and this research identifies exactly why it exists.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("#### 🔑 6 Business Insights")
        insights = [
            ("🎯 Attitude is the Master Switch", "ATT→SIB β=0.670, p<0.001",
             "Knowledge without emotional conviction fails. Institutions must cultivate positive ESG attitude — not just literacy."),
            ("💰 Sell ESG on Returns First", "EFR→ATT β=0.756, p<0.001",
             "Lead with financial performance data. When Gen Z believes ESG pays, attitude shifts — then investing follows."),
            ("⚠️ Dismantle the Risk Myth", "PR→ATT β=−0.414, p=0.016",
             "Risk perception is the #1 attitude killer. Show volatility comparisons and risk-adjusted returns."),
            ("🌿 Concern Must Become Confidence", "EC→SIB indirect only, p=0.005",
             "EC=3.70 globally but doesn't directly invest. Convert environmental concern into investment confidence."),
            ("📢 Awareness Has a Direct Channel", "SFA→SIB β=0.285, p=0.017",
             "Only IV with a direct path. Awareness campaigns are necessary first steps but insufficient alone."),
            ("📚 General Literacy Is Not Enough", "FL p=0.207 — not significant",
             "Domain-specific sustainable finance literacy — not generic financial education — is the key intervention."),
        ]
        for title, stat, body in insights:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin-bottom:10px;">
              <div style="font-weight:700;color:#e6edf3;font-size:.95rem;">{title}</div>
              <div style="color:#3fb950;font-size:.8rem;margin:3px 0 6px;">{stat}</div>
              <div style="color:#8b949e;font-size:.82rem;line-height:1.5;">{body}</div>
            </div>""", unsafe_allow_html=True)

    with col_f2:
        st.markdown("#### 🏦 Strategic Recommendations")

        recs = {
            "Banks & Asset Managers": [
                "Lead ESG product comms with financial performance data — not green messaging alone",
                "Show risk-adjusted return comparisons vs conventional funds using MSCI ESG data",
                "Design simple, mobile-first ESG products priced for Gen Z retail investors",
                "Use peer testimonials and social proof — Gen Z trusts community over corporate claims",
                "Create ESG onboarding flows that address risk myth before product discussion",
            ],
            "Policymakers & Regulators": [
                "Expand EU SFDR-style disclosure frameworks globally — regulation accelerates adoption",
                "Fund domain-specific sustainable finance literacy programs, not generic financial ed",
                "Mandate standardised ESG performance reporting to reduce information asymmetry",
                "Remove tax friction for retail ESG investors in lower-income markets",
            ],
            "Educators & Universities": [
                "Integrate sustainable finance modules into undergraduate business curricula",
                "Partner with GSIA/CFA Institute to certify ESG-specific financial literacy",
                "Use this cross-national evidence base for comparative policy research",
            ],
        }
        for group, points in recs.items():
            with st.expander(f"📌 {group}"):
                for p in points:
                    st.markdown(f"→ {p}")

        st.markdown("---")
        st.markdown("#### 📌 Research Limitations & Future Work")
        limitations = [
            "Secondary data — no primary survey validation per country",
            "Cross-sectional — no longitudinal ESG adoption tracking",
            "FL not disaggregated into domain-specific ESG literacy",
            "Future: primary Gen Z survey data across same 45 countries",
            "Future: add crypto/DeFi ESG variables for digital-native cohort",
            "Future: time-series analysis as ESG regulation evolves post-2026",
        ]
        for l in limitations:
            st.markdown(f"• {l}")

    # Bottom watermark
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#8b949e;font-size:.8rem;padding:10px;">
      Aishwarya Birla · GCS25GF135 · SP Jain School of Global Management · Global Finance · IBR 2026<br>
      Data sources: Morgan Stanley 2023 · OECD/INFE 2022 · Eurobarometer 2023 · Pew Research 2022 · CFA Institute 2022 · GSIA 2022 · BlackRock 2023
    </div>
    """, unsafe_allow_html=True)

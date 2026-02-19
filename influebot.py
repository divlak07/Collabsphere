import streamlit as st
import pandas as pd
import os
import re
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# 1. CORE ARCHITECTURE: FAIL-SAFE PARSING
# -------------------------------
CSV_PATH = r"C:\Users\user\Desktop\influcerpro\influcer.csv"

@st.cache_data
def load_and_refine_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    
    try:
        # Handling encoding and hidden characters
        df = pd.read_csv(CSV_PATH, encoding='latin-1')
        df.columns = [str(c).strip().replace('_', ' ') for c in df.columns] 
        
        # PRO-MAPPING (Detecting intent, not just string matching)
        mapping_rules = {
            'Name': ['name', 'talent', 'influencer', 'handle', 'id'],
            'Budget': ['bud', 'fee', 'cost', 'inr', 'price', 'money'],
            'Followers': ['fol', 'reach', 'sub', 'fan', 'audience'],
            'Engagement': ['eng', 'rate', 'perf', 'score', 'ratio'],
            'Category': ['cat', 'niche', 'genre', 'type'],
            'Region': ['reg', 'state', 'loc', 'city', 'area']
        }
        
        found_cols = {}
        for target, keys in mapping_rules.items():
            for col in df.columns:
                if any(k in col.lower() for k in keys):
                    found_cols[col] = target
                    break
        
        df = df.rename(columns=found_cols)

        # INDEX-BASED EMERGENCY FALLBACK (Prevents KeyError: 'Followers')
        cols = list(df.columns)
        fallback_map = {0: 'Name', 1: 'Category', 2: 'Followers', 3: 'Budget', 4: 'Engagement', 5: 'Region'}
        for idx, name in fallback_map.items():
            if name not in df.columns and len(cols) > idx:
                df.rename(columns={cols[idx]: name}, inplace=True)

        # DATA SANITIZATION (Numeric Forcing)
        def clean_numeric(val):
            if pd.isna(val): return 0.0
            num_str = "".join(re.findall(r"[\d.]", str(val)))
            return float(num_str) if num_str else 0.0

        for num_col in ['Budget', 'Followers', 'Engagement']:
            if num_col in df.columns:
                df[num_col] = df[num_col].apply(clean_numeric)
            else:
                df[num_col] = 0.0

        return df
    except Exception as e:
        st.error(f"Architectural Failure: {e}")
        return pd.DataFrame()

# -------------------------------
# 2. BRANDING & STYLE ENGINE
# -------------------------------
st.set_page_config(page_title="CollabSphere Imperial", page_icon="⚜️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');
    
    .main { background-color: #06080b; }
    
    /* Header Styling */
    .imperial-header {
        font-family: 'Playfair Display', serif;
        background: linear-gradient(to right, #bf953f, #fcf6ba, #b38728, #fbf5b7, #aa771c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 62px; font-weight: 800; text-align: center; margin-bottom: 0px;
    }
    
    /* Luxury Glass Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 20, 26, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-top: 4px solid #d4af37;
        padding: 30px !important;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] { background-color: #0c0e12; border-right: 1px solid #1c252e; }
    
    /* Dataframe Styling */
    .stDataFrame { border: 1px solid #1c252e; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

df = load_and_refine_data()

# -------------------------------
# 3. EXECUTIVE NAVIGATION
# -------------------------------
with st.sidebar:
    st.markdown("<h1 style='color:#d4af37; text-align:center;'>⚜️ COLLABSPHERE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#5d6d7e; font-size:10px;'>IMPERIAL RESERVE v15.0</p>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Strategic Hub", ["Dashboard", "Talent Acquisition", "ROI Matrix", "AI Strategy Vault"])

if df.empty:
    st.error("System Offline: Database Connection Failed.")
    st.stop()

# -------------------------------
# PAGE: DASHBOARD
# -------------------------------
if menu == "Dashboard":
    st.markdown('<p class="imperial-header">Imperial Insights</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#5d6d7e; letter-spacing:5px; text-transform:uppercase;'>Global Talent Overview</p>", unsafe_allow_html=True)
    
    # Hero Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verified Partners", f"{len(df):,}")
    c2.metric("Portfolio Reach", f"{df['Followers'].sum()/1000000:.1f}M")
    c3.metric("Capital Efficiency", f"{df['Engagement'].mean():.2f}%")
    c4.metric("Avg Allocation", f"₹{df['Budget'].mean():,.0f}")

    

    st.markdown("<br>", unsafe_allow_html=True)
    
    g1, g2 = st.columns([1.2, 0.8])
    with g1:
        # Hierarchical Market Share
        fig_sun = px.sunburst(df, path=['Region', 'Category'], values='Followers',
                             color_continuous_scale="YlOrBr", template="plotly_dark")
        fig_sun.update_layout(title="<b>Market Authority by Region & Niche</b>", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sun, use_container_width=True)
    with g2:
        # Efficiency Matrix
        fig_scatter = px.scatter(df, x="Budget", y="Engagement", size="Followers", color="Region",
                                title="<b>ROI Performance Matrix</b>", template="plotly_dark")
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------
# PAGE: TALENT ACQUISITION
# -------------------------------
elif menu == "Talent Acquisition":
    st.markdown('<p class="imperial-header">Talent Sourcing</p>', unsafe_allow_html=True)
    
    with st.container():
        f1, f2, f3 = st.columns(3)
        with f1: s_reg = st.multiselect("Geographic Targeting", sorted(df['Region'].unique()))
        with f2: s_cat = st.multiselect("Industry Niche", sorted(df['Category'].unique()))
        with f3: 
            max_val = float(df['Budget'].max()) if df['Budget'].max() > 0 else 1000000.0
            s_bud = st.slider("Capital Allocation Limit (₹)", 0.0, max_val, max_val)

    # Filtering Engine
    results = df[df['Budget'] <= s_bud]
    if s_reg: results = results[results['Region'].isin(s_reg)]
    if s_cat: results = results[results['Category'].isin(s_cat)]

    

    st.markdown(f"### ⚜️ {len(results)} Elite Profiles Identified")
    st.dataframe(
        results.sort_values(by='Engagement', ascending=False),
        column_config={
            "Engagement": st.column_config.ProgressColumn("ROI Efficiency", format="%.2f%%", min_value=0, max_value=25),
            "Budget": st.column_config.NumberColumn("Fee (₹)", format="₹%d"),
            "Followers": st.column_config.NumberColumn("Reach", format="%d")
        },
        use_container_width=True, hide_index=True
    )

# -------------------------------
# PAGE: ROI MATRIX
# -------------------------------
elif menu == "ROI Matrix":
    st.markdown('<p class="imperial-header">ROI Matrix</p>', unsafe_allow_html=True)
    targets = st.multiselect("Select Portfolio for Comparative DNA Analysis", df['Name'].unique())
    
    if len(targets) >= 2:
        mdf = df[df['Name'].isin(targets)]
        
        
        
        fig = go.Figure()
        for _, r in mdf.iterrows():
            # Normalized Scoring logic
            eng_score = r['Engagement'] * 4
            reach_score = (r['Followers'] / df['Followers'].max()) * 100 if df['Followers'].max() > 0 else 50
            cost_score = 100 - ((r['Budget'] / df['Budget'].max()) * 100) if df['Budget'].max() > 0 else 50
            
            fig.add_trace(go.Scatterpolar(
                r=[eng_score, reach_score, cost_score],
                theta=['Engagement Performance', 'Market Reach', 'Cost Efficiency'],
                fill='toself', name=r['Name']
            ))
        
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚜️ Select at least two partners to generate performance DNA analysis.")

# -------------------------------
# PAGE: AI STRATEGY VAULT
# -------------------------------
elif menu == "AI Strategy Vault":
    st.markdown('<p class="imperial-header">AI Strategy Vault</p>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Director, how shall we optimize our market penetration today?"}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.write(m["content"])

    if query := st.chat_input("Ex: Best lifestyle influencer in Delhi under 50k"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"): st.write(query)

        with st.chat_message("assistant"):
            matches = df[df.apply(lambda r: query.lower() in str(r.values).lower(), axis=1)]
            if not matches.empty:
                top = matches.sort_values('Engagement', ascending=False).iloc[0]
                resp = f"Strategic Recommendation: **{top['Name']}** in **{top['Region']}**. Current ROI: **{top['Engagement']}%**."
                st.markdown(resp)
                st.table(matches[['Name', 'Region', 'Engagement', 'Budget']].head(3))
            else:
                resp = "Consulting top performers... No exact match found for parameters."
                st.warning(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
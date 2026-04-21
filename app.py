import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os
import re
import time
# Set page config for wide layout and title
st.set_page_config(
    page_title="DeliveryIQ Analytics Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_file_path = "style.css"
    if os.path.exists(css_file_path):
        with open(css_file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session State Initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'mapped' not in st.session_state:
    st.session_state.mapped = False
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'model_state' not in st.session_state:
    st.session_state.model_state = None
if 'alert_time' not in st.session_state:
    st.session_state.alert_time = time.time()
if 'dismiss_alerts' not in st.session_state:
    st.session_state.dismiss_alerts = False
if 'clean_report' not in st.session_state:
    st.session_state.clean_report = []


# Utility Functions
def format_hour(hour_val):
    try:
        h = int(hour_val)
        if h == 0: return "12 AM"
        elif h < 12: return f"{h} AM"
        elif h == 12: return "12 PM"
        else: return f"{h-12} PM"
    except:
        return hour_val

def apply_theme(theme_name):
    if theme_name == "Business Blue":
        st.markdown("""
        <style>
        :root {
            --background-dark: #0B132B !important;
            --card-bg: rgba(28, 37, 65, 0.6) !important;
            --card-border: rgba(255, 255, 255, 0.1) !important;
            --accent-color: #00C9A7 !important;
            --accent-gradient: linear-gradient(135deg, #00C9A7 0%, #00F5D4 50%, #3A86FF 100%) !important;
            --glow-shadow: 0 0 20px rgba(0, 245, 212, 0.2) !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--background-dark) !important;
            background: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='stagger-1'>DeliveryIQ Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='stagger-1' style='font-size: 1.1rem; color: #A0AEC0; margin-bottom: 2rem;'>Complete analytics, dynamic visualization, and machine learning engine.</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎨 Appearance")
theme_selection = st.sidebar.selectbox("Theme Switcher", ["Modern Neon", "Business Blue"])
apply_theme(theme_selection)

st.sidebar.title("📂 Data Input")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])

def auto_detect_columns(columns):
    mapping = {
        'Distance': None,
        'Time': None,
        'Revenue': None,
        'Rating': None,
        'Hour': None
    }
    
    for col in columns:
        if not mapping['Distance'] and re.search(r'(?i)(distance|dist|km|miles)', col):
            mapping['Distance'] = col
        elif not mapping['Time'] and re.search(r'(?i)(duration|mins|minutes|delivery.*time)', col) and not re.search(r'(?i)(date|time of|hour|placed)', col):
            mapping['Time'] = col
        elif not mapping['Revenue'] and re.search(r'(?i)(revenue|value|price|amount|total|sales)', col):
            mapping['Revenue'] = col
        elif not mapping['Rating'] and re.search(r'(?i)(rating|score|stars)', col):
            mapping['Rating'] = col
        elif not mapping['Hour'] and re.search(r'(?i)(hour|time of day|order time|placed)', col):
            mapping['Hour'] = col
            
    return mapping

def clean_and_impute(df):
    """Clean dataset and return a report of what was imputed."""
    report = []  # list of (col, n_filled, method)
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        if df[col].dtype in ['float64', 'int64']:
            fill_val = df[col].median()
            df[col].fillna(fill_val, inplace=True)
            report.append((col, n_missing, f"median ({fill_val:.2f})"))
        else:
            fill_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(fill_val, inplace=True)
            report.append((col, n_missing, f"mode ('{fill_val}')"))

    # Mock Coordinate Generator for Map Visualization
    city_coords = {
        "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
        "Bangalore": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
        "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567)
    }
    city_col = next((c for c in df.columns if re.search(r'(?i)(city)', c)), None)
    if city_col and 'Latitude' not in df.columns:
        lats, lons = [], []
        np.random.seed(42)
        districts = []
        zones = ['Central Commercial', 'North District', 'South Suburbs', 'West End',
                 'East Side', 'Downtown', 'Tech Park Area', 'Residential Hub']
        for city in df[city_col]:
            coords = city_coords.get(city, (20.5937, 78.9629))
            lats.append(coords[0] + np.random.normal(0, 0.05))
            lons.append(coords[1] + np.random.normal(0, 0.05))
            districts.append(f"{city} - {np.random.choice(zones)}")
        df['Latitude'] = lats
        df['Longitude'] = lons
        df['Location'] = districts

    return df, report


# ====== DATA VALIDATION LAYER ======

def validate_dataset(mapping, df):
    """
    Analyse mapping + df to produce structured errors, warnings, and info.
    Returns dict: {'errors': [...], 'warnings': [...], 'info': [...], 'feature_flags': {...}}
    """
    errors   = []
    warnings = []
    info     = []
    flags    = {
        'has_time':     False,
        'has_distance': False,
        'has_hour':     False,
        'has_revenue':  False,
        'has_rating':   False,
        'ml_ready':     False,
    }

    total_rows = len(df)

    def _col_ok(key):
        v = mapping.get(key)
        return v and v not in (None, "None") and v in df.columns

    # --- Mandatory: Delivery Time ---
    if _col_ok('Time'):
        flags['has_time'] = True
        col = mapping['Time']
        # Type check
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Delivery Time column `{col}` contains non-numeric values. Predictions will fail.")
        # Missing rate
        miss = df[col].isnull().sum()
        miss_pct = miss / total_rows * 100
        if miss_pct > 20:
            warnings.append(f"Delivery Time (`{col}`) has {miss_pct:.1f}% missing values — analysis may be skewed.")
    else:
        errors.append("Delivery Time column is NOT mapped. This is required for all analysis.")

    # --- Optional: Distance ---
    if _col_ok('Distance'):
        flags['has_distance'] = True
        col = mapping['Distance']
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"Distance column `{col}` appears non-numeric. Distance analysis may be inaccurate.")
    else:
        warnings.append("Distance column not mapped. Distance-based analysis and route recommendations will be disabled.")

    # --- Optional: Hour ---
    if _col_ok('Hour'):
        flags['has_hour'] = True
        col = mapping['Hour']
        max_h = df[col].dropna().max()
        if pd.api.types.is_numeric_dtype(df[col]) and max_h > 23:
            warnings.append(f"Hour column `{col}` has values > 23 — it may not be an hour-of-day column.")
    else:
        warnings.append("Hour column not mapped. Peak hour analysis and time-based trends will be disabled.")

    # --- Optional: Revenue ---
    if _col_ok('Revenue'):
        flags['has_revenue'] = True
    else:
        info.append("Revenue column not mapped. Revenue charts will be hidden.")

    # --- Optional: Rating ---
    if _col_ok('Rating'):
        flags['has_rating'] = True
        col = mapping['Rating']
        if pd.api.types.is_numeric_dtype(df[col]):
            mx = df[col].dropna().max()
            if mx > 10:
                warnings.append(f"Rating column `{col}` has values > 10. Expected range 1–5.")
    else:
        info.append("Rating column not mapped. Customer satisfaction analysis will be hidden.")

    # --- ML readiness ---
    if flags['has_time'] and (flags['has_distance'] or flags['has_hour']):
        flags['ml_ready'] = True
    else:
        info.append("Prediction module requires at least Delivery Time + Distance or Hour. Currently unavailable.")

    # --- Row count check ---
    if total_rows < 50:
        warnings.append(f"Dataset has only {total_rows} rows. Insights may not be statistically reliable.")

    return {'errors': errors, 'warnings': warnings, 'info': info, 'flags': flags}


def display_validation_report(mapping, df, clean_report=None):
    """
    Render the Dataset Health Check panel.
    Returns the validation result dict (flags etc.) for downstream use.
    """
    result = validate_dataset(mapping, df)

    st.markdown("<div class='section-header'>🩺 Dataset Health Check</div>", unsafe_allow_html=True)

    # ── Column status grid ──
    col_defs = [
        ('Time',     'Delivery Time',  True),
        ('Distance', 'Distance',        False),
        ('Hour',     'Order Hour',      False),
        ('Revenue',  'Revenue',         False),
        ('Rating',   'Customer Rating', False),
    ]
    grid_cols = st.columns(len(col_defs))
    for i, (key, label, required) in enumerate(col_defs):
        v = mapping.get(key)
        detected = v and v not in (None, "None") and v in df.columns
        if detected:
            icon, bg, border, status = "✅", "rgba(16,185,129,0.12)", "#10B981", f"Detected: {v}"
        elif required:
            icon, bg, border, status = "❌", "rgba(239,68,68,0.12)", "#EF4444", "CRITICAL — Not Mapped"
        else:
            icon, bg, border, status = "⚠️", "rgba(245,158,11,0.12)", "#F59E0B", "Not Mapped"
        # Pre-build the required badge OUTSIDE the f-string to avoid quote-escaping issues
        required_badge = "<span style='color:#EF4444; font-size:0.7rem; margin-left:4px;'>★ Required</span>" if required else ""
        with grid_cols[i]:
            st.markdown(
                f"<div style='background:{bg}; border:1px solid {border}; border-radius:10px;"
                f" padding:0.9rem; text-align:center; margin-bottom:0.5rem;'>"
                f"<div style='font-size:1.6rem;'>{icon}</div>"
                f"<div style='font-weight:700; font-size:0.9rem; margin:4px 0;'>{label}{required_badge}</div>"
                f"<div style='font-size:0.75rem; color:#94A3B8;'>{status}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Errors ──
    for msg in result['errors']:
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.12); border-left:4px solid #EF4444;
                    border-radius:8px; padding:0.8rem 1.2rem; margin-bottom:0.5rem;'>
            ❌ <strong>Error:</strong> {msg}
        </div>""", unsafe_allow_html=True)

    # ── Warnings ──
    for msg in result['warnings']:
        st.markdown(f"""
        <div style='background:rgba(245,158,11,0.1); border-left:4px solid #F59E0B;
                    border-radius:8px; padding:0.8rem 1.2rem; margin-bottom:0.5rem;'>
            ⚠️ <strong>Warning:</strong> {msg}
        </div>""", unsafe_allow_html=True)

    # ── Info ──
    for msg in result['info']:
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.1); border-left:4px solid #6366F1;
                    border-radius:8px; padding:0.8rem 1.2rem; margin-bottom:0.5rem;'>
            ℹ️ {msg}
        </div>""", unsafe_allow_html=True)

    # ── Cleaning report ──
    if clean_report:
        with st.expander(f"🧹 Auto-Cleaning Report ({len(clean_report)} column(s) imputed)", expanded=False):
            for col_name, n_filled, method in clean_report:
                st.markdown(f"- Filled **{n_filled}** missing value(s) in `{col_name}` using **{method}**")

    # ── Hard stop ──
    if result['errors']:
        st.error("🚫 Cannot run analysis — please fix the errors above and re-map columns.")
        st.stop()

    # ── Guidance tips ──
    tips = []
    if not result['flags']['has_hour']:     tips.append("Add an **Order Hour** column for time-based peak analysis.")
    if not result['flags']['has_distance']: tips.append("Add a **Distance** column to enable route and SLA analysis.")
    if not result['flags']['has_revenue']:  tips.append("Add a **Revenue** column to track financial performance.")
    if tips:
        with st.expander("💡 Suggestions to improve analysis coverage", expanded=False):
            for t in tips:
                st.markdown(f"- {t}")

    return result


def handle_missing_feature(feature_name, reason=""):
    """Render a consistent 'feature disabled' notice inline."""
    msg = reason or f"Map the **{feature_name}** column to enable this feature."
    st.markdown(f"""
    <div style='background:rgba(99,102,241,0.08); border:1px dashed rgba(99,102,241,0.4);
                border-radius:8px; padding:0.9rem 1.2rem; color:#94A3B8;
                font-size:0.9rem; margin:0.5rem 0;'>
        🔒 <strong>Feature Unavailable</strong> — {msg}
    </div>""", unsafe_allow_html=True)


def generate_alerts(filtered_df, mapping):
    alerts = []
    # 1. Late deliveries alert
    if mapping['Time'] != "None":
        threshold = filtered_df[mapping['Time']].mean() + filtered_df[mapping['Time']].std()
        late_count = len(filtered_df[filtered_df[mapping['Time']] > threshold])
        late_ratio = late_count / len(filtered_df)
        if late_ratio > 0.2:
            alerts.append({"type": "critical", "icon": "🔴", "text": "Critical: Too many late deliveries occurring."})
        elif late_ratio > 0.1:
            alerts.append({"type": "warning", "icon": "🟡", "text": "Warning: Mild increase in late deliveries."})
        else:
            alerts.append({"type": "good", "icon": "🟢", "text": "System Performing Well: Deliveries are mostly on time."})

    # 2. Peak demand alert
    if mapping['Hour'] != "None":
        current_hour_approx = filtered_df[mapping['Hour']].mode()[0] if not filtered_df.empty else None
        if current_hour_approx is not None:
            alerts.append({"type": "warning", "icon": "⚡", "text": f"Peak demand approaching around {format_hour(current_hour_approx)}!"})

    if alerts:
        # Inject fade-out CSS + JS animation (fades after 5 seconds, gone at 8)
        st.markdown("""
        <style>
        @keyframes fadeOutAlert {
            0%   { opacity: 1; transform: translateY(0); }
            70%  { opacity: 1; }
            100% { opacity: 0; transform: translateY(-8px); pointer-events: none; }
        }
        .auto-fade-alert {
            animation: fadeOutAlert 1.2s ease-out 5s forwards;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='alerts-container'>", unsafe_allow_html=True)
        for a in alerts:
            st.markdown(
                f"<div class='manager-alert auto-fade-alert {a['type']}'>"
                f"<span style='font-size:1.4rem;'>{a['icon']}</span>&nbsp;{a['text']}"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

def generate_insight_summary(filtered_df, mapping):
    st.markdown("<div class='section-header'>📊 Today's Summary</div>", unsafe_allow_html=True)
    
    # Calculate values
    peak_h_str, avg_time_str, late_pct_str, key_issue = "N/A", "N/A", "N/A", "No clear issue"
    late_pct_val = 0
    if mapping['Hour'] != "None":
        peak_h = filtered_df.groupby(mapping['Hour']).size().idxmax()
        peak_h_str = format_hour(peak_h)
    
    if mapping['Time'] != "None":
        avg_time = filtered_df[mapping['Time']].mean()
        avg_time_str = f"{avg_time:.1f} mins"
        
        threshold = filtered_df[mapping['Time']].mean() + 10 # 10 mins padding as threshold based on previous logic
        late_count = len(filtered_df[filtered_df[mapping['Time']] > threshold])
        late_pct_val = (late_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        late_pct_str = f"{late_pct_val:.1f}%"
        
        if late_pct_val > 15:
            key_issue = "High delivery delay rate"
        elif mapping['Rating'] != "None" and filtered_df[mapping['Rating']].mean() < 3.5:
             key_issue = "Low customer satisfaction"
        elif mapping['Distance'] != "None" and filtered_df[mapping['Distance']].mean() > 10:
             key_issue = "Excessive delivery distances"
        else:
             key_issue = "System operating optimally"

    # Delivery Efficiency Score
    # Score out of 100 based on mostly time and lates
    score = 100
    score -= late_pct_val * 1.5
    if mapping['Time'] != "None":
        score -= min(30, max(0, avg_time - 25)) # Deduct points if average time is > 25 mins
    
    score = max(0, min(100, score))
    score_color = "#10B981" if score > 75 else "#F59E0B" if score > 50 else "#EF4444"

    sum_col1, sum_col2 = st.columns([2, 1])
    with sum_col1:
        st.markdown(f"""
        <div class='custom-card'>
            <h3>Snapshot</h3>
            <div style='display: flex; gap: 2rem; margin-top: 1rem;'>
                <div><strong>Peak Hour:</strong> {peak_h_str}</div>
                <div><strong>Avg Delivery:</strong> {avg_time_str}</div>
                <div><strong>Late Deliveries:</strong> {late_pct_str}</div>
            </div>
            <div style='margin-top: 1rem; color: var(--warning);'><strong>Key Issue:</strong> {key_issue}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with sum_col2:
        st.markdown(f"""
        <div class='custom-card' style='text-align: center;'>
            <h3 style='margin-bottom: 0.5rem;'>Decision Score</h3>
            <div style='font-size: 3rem; font-weight: 900; color: {score_color};'>{score:.0f}</div>
            <div style='width: 100%; background: rgba(255,255,255,0.1); border-radius: 10px; height: 10px; margin-top: 10px;'>
                 <div style='width: {score}%; background: {score_color}; height: 100%; border-radius: 10px; transition: width 1s;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def generate_kpis(filtered_df, mapping):
    st.markdown("<div class='section-header'>📈 Key Performance Indicators</div>", unsafe_allow_html=True)
    kpi_cols = st.columns(4)
    
    # Simulating trends for visual appeal
    with kpi_cols[0]:
        st.metric("Total Orders", f"{len(filtered_df):,}", delta=f"{np.random.randint(2, 10)}%" if len(filtered_df)>0 else None)
        
    with kpi_cols[1]:
        if mapping['Time'] != "None":
            st.metric("Avg Delivery Time", f"{filtered_df[mapping['Time']].mean():.1f} mins", delta=f"-{np.random.uniform(0.5, 3.0):.1f} mins", delta_color="inverse")
        else:
            st.metric("Avg Delivery Time", "N/A")
            
    with kpi_cols[2]:
        if mapping['Rating'] != "None":
            st.metric("Avg Rating", f"{filtered_df[mapping['Rating']].mean():.1f} ⭐", delta=f"+{np.random.uniform(0.1, 0.4):.1f}")
        else:
            st.metric("Avg Rating", "N/A")
            
    with kpi_cols[3]:
        if mapping['Revenue'] != "None":
            st.metric("Total Revenue", f"${filtered_df[mapping['Revenue']].sum():,.0f}", delta=f"+${np.random.randint(500, 2000):,}")
        else:
            st.metric("Total Revenue", "N/A")

def generate_simple_graphs(filtered_df, mapping):
    st.markdown("<div class='section-header'>📊 Business Overview View</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        if mapping['Hour'] != "None":
            hour_counts = filtered_df.groupby(mapping['Hour']).size().reset_index(name='Orders')
            hour_counts['Hour_Formatted'] = hour_counts[mapping['Hour']].apply(format_hour)
            peak_h = hour_counts.loc[hour_counts['Orders'].idxmax(), 'Hour_Formatted']
            
            fig = px.bar(hour_counts, x='Hour_Formatted', y='Orders', title=f"Orders by Hour", template='plotly_dark', text='Orders')
            fig.update_traces(marker_color='#3b82f6', textposition='outside', hovertemplate='<b>%{x}</b>: %{y} Orders<extra></extra>')
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Hour of Day", yaxis_title="Number of Orders")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"💡 **Business Insight:** Peak order volume occurs at {peak_h}. Kitchen capacity and resource allocation should be optimized for this window.")
        else:
            st.info("Map 'Hour' column to see peak time analysis")

    with c2:
        if mapping['Hour'] != "None" and mapping['Time'] != "None":
            b_hour = filtered_df.groupby(mapping['Hour'])[mapping['Time']].mean().reset_index()
            b_hour['Hour_Formatted'] = b_hour[mapping['Hour']].apply(format_hour)
            slow_h = b_hour.loc[b_hour[mapping['Time']].idxmax(), 'Hour_Formatted']
            
            fig2 = px.bar(b_hour, x='Hour_Formatted', y=mapping['Time'], title=f"Avg Delivery Time by Hour", template='plotly_dark', text=b_hour[mapping['Time']].round(1))
            fig2.update_traces(marker_color='#f59e0b', textposition='outside', hovertemplate='<b>%{x}</b>: %{y:.1f} mins<extra></extra>')
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Hour of Day", yaxis_title="Avg Time (mins)")
            st.plotly_chart(fig2, use_container_width=True)
            st.info(f"💡 **Operational Insight:** Maximum delivery delays are observed at {slow_h}, suggesting potential traffic congestion or operational bottlenecks.")
        else:
             st.info("Map 'Hour' and 'Delivery Time' to see delivery speeds.")
        
    c3, c4 = st.columns(2)
    with c3:
        if mapping['Time'] != "None":
            threshold = filtered_df[mapping['Time']].mean() + 10
            filtered_df['Status'] = np.where(filtered_df[mapping['Time']] > threshold, 'Late', 'On-Time')
            sc = filtered_df['Status'].value_counts().reset_index()
            late_pct = (sc[sc['Status']=='Late']['count'].sum() / len(filtered_df)) * 100 if 'Late' in sc['Status'].values else 0
            
            fig3 = px.pie(sc, names='Status', values='count', title="On-Time vs Late Deliveries", template='plotly_dark', color='Status', color_discrete_map={'On-Time':'#10B981', 'Late':'#EF4444'})
            fig3.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)), hovertemplate='<b>%{label}</b>: %{value} deliveries<extra></extra>')
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
            st.info(f"💡 **Quality Alert:** {late_pct:.1f}% of deliveries exceed the baseline average SLA, creating significant risk to customer satisfaction ratings.")
        else:
             st.info("Map 'Delivery Time' to see late deliveries.")

    with c4:
        if mapping['Hour'] != "None" and mapping['Revenue'] != "None":
            r_hour = filtered_df.groupby(mapping['Hour'])[mapping['Revenue']].sum().reset_index()
            r_hour['Hour_Formatted'] = r_hour[mapping['Hour']].apply(format_hour)
            peak_r = r_hour.loc[r_hour[mapping['Revenue']].idxmax(), 'Hour_Formatted']
            
            fig4 = px.bar(r_hour, x='Hour_Formatted', y=mapping['Revenue'], title=f"Revenue Output per Hour", template='plotly_dark', text=r_hour[mapping['Revenue']].round(0))
            fig4.update_traces(marker_color='#10b981', textposition='outside', hovertemplate='<b>%{x}</b>: $%{y:,.0f}<extra></extra>')
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Hour", yaxis_title="Revenue ($)")
            st.plotly_chart(fig4, use_container_width=True)
            st.info(f"💡 **Financial Insight:** Peak profitability is achieved at {peak_r}, representing the highest revenue-generating hour.")
        else:
             st.info("Map 'Hour' and 'Revenue' to see earning trends.")
        
    st.markdown("<h3 class='stagger-3'>Top 5 Slowest Deliveries</h3>", unsafe_allow_html=True)
    if mapping['Time'] != "None":
        cols_to_show = [mapping['Time']]
        if mapping['Distance'] != "None": cols_to_show.append(mapping['Distance'])
        if mapping['Hour'] != "None": cols_to_show.append(mapping['Hour'])
        slowest = filtered_df.nlargest(5, mapping['Time'])[cols_to_show]
        st.dataframe(slowest, use_container_width=True)

def generate_map_visualization(df, mapping):
    st.markdown("<div class='section-header'>🗺️ Geographic Intelligence Focus</div>", unsafe_allow_html=True)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Determine color and size parameters dynamically based on mapped columns
        color_col = mapping.get('Time') if mapping.get('Time') != "None" else None
        size_col = mapping.get('Revenue') if mapping.get('Revenue') != "None" else None
        
        # Determine hover name explicitly
        hover_target = "Location" if "Location" in df.columns else next((c for c in df.columns if re.search(r'(?i)(city|restaurant)', c)), None)
        
        # Using a scatter mapbox with mock data
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", 
                                color=color_col, size=size_col,
                                hover_name=hover_target, 
                                hover_data=[c for c in df.columns if re.search(r'(?i)(time|revenue|rating)', c)],
                                color_continuous_scale="Viridis", zoom=4, height=500)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        st.info("💡 **Spatial Analysis:** This heatmap visualizes geographic distribution. Color gradients indicate average delivery delays (warmer hues signify higher latency), while marker sizes represent revenue density.")
        st.markdown("<p style='text-align:center; color: var(--text-muted); margin-top: 5px; font-size: 0.85rem;'>📍 <i>Locations are approximated based on city-level data for demonstration purposes</i></p>", unsafe_allow_html=True)

def generate_recommendations(filtered_df, mapping):
    st.markdown("<div class='section-header'>💡 Intelligent Recommendations</div>", unsafe_allow_html=True)
    
    # Rec 1: Staffing
    if mapping['Hour'] != "None" and mapping['Time'] != "None":
        b_hour = filtered_df.groupby(mapping['Hour'])[mapping['Time']].mean().reset_index()
        slow_h = format_hour(b_hour.loc[b_hour[mapping['Time']].idxmax(), mapping['Hour']])
        html_content = f"""
<div class='recommendation-card' style='display: flex; flex-direction: column; gap: 15px;'>
<div style='display: flex; align-items: center; gap: 12px;'>
<span style='font-size: 2.2rem; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 12px;'>🧑‍🍳</span>
<div>
<h4 style='margin: 0; font-size: 1.25rem; font-weight: 700;'>Boost Capacity at {slow_h}</h4>
<span style='color: #F59E0B; font-size: 0.9rem; font-weight: 600;'>System Alert: Operational Bottleneck</span>
</div>
</div>
<div style='background: rgba(0,0,0,0.2); border-radius: 8px; padding: 15px; border-left: 3px solid #F59E0B;'>
<p style='margin: 0; font-size: 0.95rem; line-height: 1.5; color: #E2E8F0;'>
<strong>Why is this happening?</strong><br>
Data shows that {slow_h} consistently suffers the slowest delivery times. This happens when high order volumes overwhelm active kitchen and fleet capacity, leading to staggered dispatch delays.
</p>
</div>
<div style='background: var(--accent-gradient); border-radius: 8px; padding: 2px;'>
<div style='background: #0F172A; border-radius: 6px; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between;'>
<div style='font-size: 0.9rem; color:#94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;'>Suggested Strategy</div>
<div style='font-size: 1.05rem; font-weight: 700; color: #fff;'>Increase kitchen staffing and fleet allocation during this shift</div>
</div>
</div>
</div>
"""
        st.markdown(html_content.replace("\n", ""), unsafe_allow_html=True)

    # Rec 2: Distances
    if mapping['Distance'] != "None" and mapping['Time'] != "None":
        corr = filtered_df[mapping['Distance']].corr(filtered_df[mapping['Time']])
        if corr > 0.4:
            html_content = f"""
<div class='recommendation-card' style='display: flex; flex-direction: column; gap: 15px;'>
<div style='display: flex; align-items: center; gap: 12px;'>
<span style='font-size: 2.2rem; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 12px;'>🗺️</span>
<div>
<h4 style='margin: 0; font-size: 1.25rem; font-weight: 700;'>Re-zone Distant Deliveries</h4>
<span style='color: #EF4444; font-size: 0.9rem; font-weight: 600;'>System Alert: SLA Violation Risk</span>
</div>
</div>
<div style='background: rgba(0,0,0,0.2); border-radius: 8px; padding: 15px; border-left: 3px solid #EF4444;'>
<p style='margin: 0; font-size: 0.95rem; line-height: 1.5; color: #E2E8F0;'>
<strong>Why is this happening?</strong><br>
There is a severe correlation between delivery distance and SLA failures right now. Distant orders are monopolizing your drivers, leaving short-range orders stranded.
</p>
</div>
<div style='background: var(--accent-gradient); border-radius: 8px; padding: 2px;'>
<div style='background: #0F172A; border-radius: 6px; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between;'>
<div style='font-size: 0.9rem; color:#94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;'>Suggested Strategy</div>
<div style='font-size: 1.05rem; font-weight: 700; color: #fff;'>Consider restricting delivery radiuses during high-volume traffic spikes</div>
</div>
</div>
</div>
"""
            st.markdown(html_content.replace("\n", ""), unsafe_allow_html=True)
            
    # Rec 3: Quality
    if mapping['Rating'] != "None" and mapping['Time'] != "None":
        corr2 = filtered_df[mapping['Time']].corr(filtered_df[mapping['Rating']])
        if corr2 < -0.3:
            html_content = f"""
<div class='recommendation-card' style='display: flex; flex-direction: column; gap: 15px;'>
<div style='display: flex; align-items: center; gap: 12px;'>
<span style='font-size: 2.2rem; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 12px;'>⭐</span>
<div>
<h4 style='margin: 0; font-size: 1.25rem; font-weight: 700;'>Expedite VIP Fast-Tracking</h4>
<span style='color: var(--accent-color); font-size: 0.9rem; font-weight: 600;'>System Alert: Churn Probability High</span>
</div>
</div>
<div style='background: rgba(0,0,0,0.2); border-radius: 8px; padding: 15px; border-left: 3px solid var(--accent-color);'>
<p style='margin: 0; font-size: 0.95rem; line-height: 1.5; color: #E2E8F0;'>
<strong>Why is this happening?</strong><br>
Customers are penalizing you severely for delays, hurting your aggregate rating. Late deliveries directly correspond to significant drops in satisfaction on current active orders.
</p>
</div>
<div style='background: var(--accent-gradient); border-radius: 8px; padding: 2px;'>
<div style='background: #0F172A; border-radius: 6px; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between;'>
<div style='font-size: 0.9rem; color:#94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;'>Suggested Strategy</div>
<div style='font-size: 1.05rem; font-weight: 700; color: #fff;'>Prioritize high-rated drivers for delayed VIP orders to retain trust</div>
</div>
</div>
</div>
"""
            st.markdown(html_content.replace("\n", ""), unsafe_allow_html=True)

def render_manager_ml(filtered_df, df, mapping):
    st.markdown("<div class='section-header'>🚚 Delivery Prediction Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    
    if mapping['Time'] != "None":
        target = mapping['Time']
        # Only use Distance as the simulator input — the sole meaningful controllable variable
        distance_col = mapping.get('Distance') if mapping.get('Distance') != "None" else None
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in num_cols if c != target and 'id' not in c.lower() and c not in ['Latitude', 'Longitude']]
        # Simulator sliders: only Distance (and Hour as secondary operational variable)
        simulator_features = [c for c in [mapping.get('Distance'), mapping.get('Hour')] if c and c != "None" and c in num_cols]
        
        if len(features) > 0:
            if st.session_state.model_state is None:
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                X = df[features].fillna(0)
                y = df[target].fillna(0)
                model.fit(X, y)
                st.session_state.model_state = {'model': model, 'features': features, 'accuracy': model.score(X, y), 'avg_y': y.mean()}
            
            model = st.session_state.model_state['model']
            features = st.session_state.model_state['features']
            acc = st.session_state.model_state['accuracy'] * 100
            
            # Single-column simulator layout (no Feature Importance)
            st.markdown("#### What-If Simulator")
            st.write("Adjust variables below to predict delivery duration.")
            inputs = {}
            base_inputs = {}
            for f in features:
                base_inputs[f] = float(df[f].mean())

            # Show sliders only for Distance and Hour
            sim_feats = simulator_features if simulator_features else features[:2]
            hour_col = mapping.get('Hour') if mapping.get('Hour') != "None" else None

            for i, f in enumerate(sim_feats):
                if f == hour_col:
                    # Build 12-hour clock options from the actual unique hour values in dataset
                    hour_vals = sorted(df[f].dropna().unique().astype(int).tolist())
                    hour_labels = [format_hour(h) for h in hour_vals]
                    # Default to hour closest to mean
                    default_hour = int(round(df[f].mean()))
                    default_hour = min(hour_vals, key=lambda x: abs(x - default_hour))
                    default_label = format_hour(default_hour)
                    selected_label = st.select_slider(
                        "Order Hour",
                        options=hour_labels,
                        value=default_label,
                        help="Select the hour the order is placed (12-hour clock)"
                    )
                    # Map label back to numeric
                    selected_idx = hour_labels.index(selected_label)
                    inputs[f] = float(hour_vals[selected_idx])
                else:
                    inputs[f] = st.slider(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()), help=f"Adjust {f}")

            input_df = pd.DataFrame([inputs])
            base_df = pd.DataFrame([base_inputs])
            for missed in set(features) - set(inputs.keys()):
                input_df[missed] = df[missed].mean()
            pred = model.predict(input_df[features])[0]
            base_pred = st.session_state.model_state['avg_y']
            delta = pred - base_pred
            pct_change = (delta / base_pred) * 100 if base_pred > 0 else 0

            avg_time = df[target].mean()

            if pred < avg_time * 0.8:
                badge = "fast"
                label = "Fast"
            elif pred > avg_time * 1.2:
                badge = "slow"
                label = "Slow"
            else:
                badge = "moderate"
                label = "Moderate"

            delta_color = "red" if delta > 0 else "green"
            delta_sign = "+" if delta > 0 else ""

            import textwrap
            html_content = f"""
<div class='highlight-prediction'>
    <p style="color:rgba(255,255,255,0.7)!important; font-size:1rem; letter-spacing:0; margin:0;">PREDICTED DURATION</p>
    <h2>{pred:.0f} <span style="font-size: 1.5rem;">mins</span></h2>
    <div style="color: {delta_color}; font-weight: bold; margin-top: 5px;">
        {delta_sign}{delta:.1f} mins ({delta_sign}{pct_change:.1f}%) compared to avg
    </div>
    <div class='ml-badge {badge}' style="margin-bottom: 5px;">{label}</div>
</div>
            """
            st.markdown(textwrap.dedent(html_content), unsafe_allow_html=True)

        else:
            st.error("Need more numeric columns for prediction.")
    else:
        st.error("Map Delivery Time to activate the Prediction Simulator.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_manager_mode(filtered_df, df, mapping):
    st.markdown("<h2 style='font-size: 2.2rem; font-weight: 900; margin-bottom: 2rem;'>Manager Dashboard</h2>", unsafe_allow_html=True)
    generate_alerts(filtered_df, mapping)
    generate_insight_summary(filtered_df, mapping)
    generate_kpis(filtered_df, mapping)
    generate_simple_graphs(filtered_df, mapping)
    generate_map_visualization(filtered_df, mapping)
    generate_recommendations(filtered_df, mapping)
    render_manager_ml(filtered_df, df, mapping)


# ====== ANALYST MODE — PROFESSIONAL DATA ANALYSIS DASHBOARD ======

def analyst_overview(filtered_df, mapping):
    st.markdown("<div class='section-header'>📊 Overview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Orders", f"{len(filtered_df):,}")
    with c2:
        if mapping['Time'] != "None":
            st.metric("Avg Delivery Time", f"{filtered_df[mapping['Time']].mean():.1f} mins")
        else:
            st.metric("Avg Delivery Time", "N/A")
    with c3:
        if mapping['Time'] != "None":
            st.metric("Max Delivery Time", f"{filtered_df[mapping['Time']].max():.1f} mins")
        else:
            st.metric("Max Delivery Time", "N/A")
    with c4:
        if mapping['Revenue'] != "None":
            st.metric("Total Revenue", f"₹{filtered_df[mapping['Revenue']].sum():,.0f}")
        else:
            st.metric("Total Revenue", "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        if mapping['Time'] != "None":
            fig_hist = px.histogram(
                filtered_df, x=mapping['Time'], nbins=30,
                title="Delivery Time Distribution",
                template='plotly_dark',
                color_discrete_sequence=['#8b5cf6']
            )
            avg_t = filtered_df[mapping['Time']].mean()
            fig_hist.add_vline(x=avg_t, line_dash="dash", line_color="#ec4899",
                               annotation_text=f"Avg: {avg_t:.1f}m",
                               annotation_position="top right")
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   xaxis_title="Delivery Time (mins)", yaxis_title="Count",
                                   bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)

    with g2:
        if mapping['Time'] != "None":
            threshold = filtered_df[mapping['Time']].mean() + 10
            filtered_df['_status'] = np.where(filtered_df[mapping['Time']] > threshold, 'Late', 'On-Time')
            sc = filtered_df['_status'].value_counts().reset_index()
            fig_pie = px.pie(sc, names='_status', values='count',
                             title="On-Time vs Late Deliveries",
                             template='plotly_dark',
                             color='_status',
                             color_discrete_map={'On-Time': '#10B981', 'Late': '#EF4444'},
                             hole=0.45)
            fig_pie.update_traces(textinfo='percent+label',
                                  marker=dict(line=dict(color='#000', width=2)))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        if mapping['Hour'] != "None" and mapping['Time'] != "None":
            hour_avg = filtered_df.groupby(mapping['Hour'])[mapping['Time']].mean().reset_index()
            hour_avg['Hour_Fmt'] = hour_avg[mapping['Hour']].apply(format_hour)
            fig_ha = px.area(hour_avg, x='Hour_Fmt', y=mapping['Time'],
                             title="Avg Delivery Time by Hour",
                             template='plotly_dark',
                             color_discrete_sequence=['#f59e0b'])
            fig_ha.update_traces(line_color='#f59e0b', fillcolor='rgba(245,158,11,0.15)')
            fig_ha.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="Hour", yaxis_title="Avg Time (mins)")
            st.plotly_chart(fig_ha, use_container_width=True)

    with g4:
        if mapping['Rating'] != "None":
            rating_counts = filtered_df[mapping['Rating']].value_counts().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            rating_counts = rating_counts.sort_values('Rating')
            fig_rat = px.bar(rating_counts, x='Rating', y='Count',
                             title="Customer Rating Distribution",
                             template='plotly_dark',
                             color='Rating',
                             color_continuous_scale='RdYlGn')
            fig_rat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_rat, use_container_width=True)
        elif mapping['Distance'] != "None":
            fig_dist = px.histogram(filtered_df, x=mapping['Distance'], nbins=25,
                                    title="Delivery Distance Distribution",
                                    template='plotly_dark',
                                    color_discrete_sequence=['#06b6d4'])
            fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_dist, use_container_width=True)




def analyst_root_cause(filtered_df, mapping):
    st.markdown("<div class='section-header'>🧠 Root Cause Analysis</div>", unsafe_allow_html=True)
    findings = []

    if mapping['Hour'] != "None" and mapping['Time'] != "None":
        hour_avg = filtered_df.groupby(mapping['Hour'])[mapping['Time']].mean()
        peak_delay_hour = hour_avg.idxmax()
        peak_label = format_hour(peak_delay_hour)
        findings.append({
            "icon": "⏰",
            "title": f"Delays are highest at {peak_label}",
            "detail": f"Average delivery time at {peak_label} is {hour_avg[peak_delay_hour]:.1f} mins, compared to overall avg of {filtered_df[mapping['Time']].mean():.1f} mins.",
            "severity": "critical"
        })

    if mapping['Hour'] != "None":
        hour_vol = filtered_df.groupby(mapping['Hour']).size()
        peak_vol_hour = hour_vol.idxmax()
        findings.append({
            "icon": "📦",
            "title": f"Peak order volume occurs at {format_hour(peak_vol_hour)}",
            "detail": f"{hour_vol[peak_vol_hour]:,} orders placed at this hour — {((hour_vol[peak_vol_hour]/len(filtered_df))*100):.1f}% of total volume.",
            "severity": "warning"
        })

    if mapping['Distance'] != "None" and mapping['Time'] != "None":
        dist_median = filtered_df[mapping['Distance']].median()
        short = filtered_df[filtered_df[mapping['Distance']] <= dist_median][mapping['Time']].mean()
        long_  = filtered_df[filtered_df[mapping['Distance']] >  dist_median][mapping['Time']].mean()
        diff = long_ - short
        corr = filtered_df[mapping['Distance']].corr(filtered_df[mapping['Time']])
        if corr > 0.35:
            findings.append({
                "icon": "🗺️",
                "title": f"Long-distance deliveries (>{dist_median:.1f} km) are adding {diff:.1f} mins of delay",
                "detail": f"Short routes avg {short:.1f} mins vs long routes avg {long_:.1f} mins. Distance is a significant contributing factor.",
                "severity": "critical"
            })
        else:
            findings.append({
                "icon": "✅",
                "title": "Distance has minimal impact on delivery time",
                "detail": f"Short routes avg {short:.1f} mins, long routes avg {long_:.1f} mins. Delays are more likely operational.",
                "severity": "good"
            })

    if mapping['Rating'] != "None" and mapping['Time'] != "None":
        corr_r = filtered_df[mapping['Time']].corr(filtered_df[mapping['Rating']])
        if corr_r < -0.3:
            findings.append({
                "icon": "⭐",
                "title": "Late deliveries are dragging down customer ratings",
                "detail": f"There is a negative relationship between delivery time and rating. Faster deliveries consistently earn higher scores.",
                "severity": "warning"
            })

    if not findings:
        st.info("Map more columns (Hour, Time, Distance) to enable root cause analysis.")
        return

    color_map = {
        "critical": ("rgba(239,68,68,0.15)", "#EF4444"),
        "warning":  ("rgba(245,158,11,0.15)", "#F59E0B"),
        "good":     ("rgba(16,185,129,0.15)", "#10B981"),
    }
    rc_col, rc_cards = st.columns([2, 1])
    with rc_cards:
        for f in findings:
            bg, border = color_map.get(f["severity"], color_map["warning"])
            st.markdown(f"""
            <div style='background:{bg}; border-left: 4px solid {border}; border-radius: 10px;
                        padding: 0.8rem 1.2rem; margin-bottom: 0.8rem;'>
                <div style='font-size:1rem; font-weight:700; margin-bottom:3px;'>{f["icon"]} {f["title"]}</div>
                <div style='font-size:0.85rem; color:#CBD5E1;'>{f["detail"]}</div>
            </div>
            """, unsafe_allow_html=True)

    with rc_col:
        if mapping['Hour'] != "None" and mapping['Time'] != "None":
            hour_avg = filtered_df.groupby(mapping['Hour'])[mapping['Time']].mean().reset_index()
            hour_avg['Hour_Fmt'] = hour_avg[mapping['Hour']].apply(format_hour)
            overall_avg = filtered_df[mapping['Time']].mean()
            hour_avg['Color'] = hour_avg[mapping['Time']].apply(
                lambda x: '#EF4444' if x > overall_avg * 1.1 else '#10B981'
            )
            fig_rc = px.bar(hour_avg, x='Hour_Fmt', y=mapping['Time'],
                            title="Avg Delivery Time by Hour (Red = Above Avg)",
                            template='plotly_dark')
            fig_rc.update_traces(marker_color=hour_avg['Color'].tolist())
            fig_rc.add_hline(y=overall_avg, line_dash='dot', line_color='#ec4899',
                             annotation_text=f'Overall Avg: {overall_avg:.1f}m')
            fig_rc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="Hour", yaxis_title="Avg Time (mins)")
            st.plotly_chart(fig_rc, use_container_width=True)

        if mapping['Distance'] != "None" and mapping['Time'] != "None":
            sample = filtered_df.sample(min(500, len(filtered_df)), random_state=42)
            fig_sc = px.scatter(sample, x=mapping['Distance'], y=mapping['Time'],
                                title="Distance vs Delivery Time",
                                template='plotly_dark',
                                opacity=0.6,
                                color=mapping['Time'],
                                color_continuous_scale='RdYlGn_r')
            fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="Distance (km)", yaxis_title="Delivery Time (mins)",
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)


def analyst_segmentation(filtered_df, mapping):
    st.markdown("<div class='section-header'>📊 Segmentation Analysis</div>", unsafe_allow_html=True)
    segments = []

    # Distance segmentation
    if mapping['Distance'] != "None" and mapping['Time'] != "None":
        dist_median = filtered_df[mapping['Distance']].median()
        short_df = filtered_df[filtered_df[mapping['Distance']] <= dist_median]
        long_df  = filtered_df[filtered_df[mapping['Distance']] >  dist_median]
        avg_time = filtered_df[mapping['Time']].mean() + 10
        for label, seg_df in [("Short Distance", short_df), ("Long Distance", long_df)]:
            late_pct = (seg_df[mapping['Time']] > avg_time).mean() * 100
            segments.append({
                "Segment": label,
                "Orders": f"{len(seg_df):,}",
                "Avg Delivery (mins)": f"{seg_df[mapping['Time']].mean():.1f}",
                "Delay Rate": f"{late_pct:.1f}%"
            })

    # Peak/off-peak segmentation
    if mapping['Hour'] != "None" and mapping['Time'] != "None":
        hour_vol = filtered_df.groupby(mapping['Hour']).size()
        peak_hours = hour_vol[hour_vol >= hour_vol.quantile(0.75)].index.tolist()
        peak_df    = filtered_df[filtered_df[mapping['Hour']].isin(peak_hours)]
        offpeak_df = filtered_df[~filtered_df[mapping['Hour']].isin(peak_hours)]
        avg_time = filtered_df[mapping['Time']].mean() + 10
        for label, seg_df in [("Peak Hours", peak_df), ("Off-Peak Hours", offpeak_df)]:
            if len(seg_df) == 0: continue
            late_pct = (seg_df[mapping['Time']] > avg_time).mean() * 100
            segments.append({
                "Segment": label,
                "Orders": f"{len(seg_df):,}",
                "Avg Delivery (mins)": f"{seg_df[mapping['Time']].mean():.1f}",
                "Delay Rate": f"{late_pct:.1f}%"
            })

    if segments:
        seg_df_display = pd.DataFrame(segments)
        st.markdown("**Segment Comparison Table**")
        st.dataframe(seg_df_display, use_container_width=True, hide_index=True)

        # Visual bar chart comparison
        st.markdown("<br>", unsafe_allow_html=True)
        chart_data = pd.DataFrame({
            'Segment': [s['Segment'] for s in segments],
            'Avg Delivery (mins)': [float(s['Avg Delivery (mins)']) for s in segments],
            'Delay Rate (%)': [float(s['Delay Rate'].replace('%','')) for s in segments]
        })
        sv1, sv2 = st.columns(2)
        with sv1:
            fig_seg1 = px.bar(chart_data, x='Segment', y='Avg Delivery (mins)',
                             title='Avg Delivery Time by Segment',
                             template='plotly_dark',
                             color='Segment',
                             color_discrete_sequence=['#8b5cf6','#ec4899','#f59e0b','#06b6d4'])
            fig_seg1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   showlegend=False)
            st.plotly_chart(fig_seg1, use_container_width=True)
        with sv2:
            fig_seg2 = px.bar(chart_data, x='Segment', y='Delay Rate (%)',
                             title='Delay Rate (%) by Segment',
                             template='plotly_dark',
                             color='Delay Rate (%)',
                             color_continuous_scale='RdYlGn_r')
            fig_seg2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_seg2, use_container_width=True)
    else:
        st.info("Map Distance, Time, and Hour columns to enable segmentation analysis.")


def analyst_trends(filtered_df, mapping):
    st.markdown("<div class='section-header'>📈 Trend Analysis</div>", unsafe_allow_html=True)

    if mapping['Hour'] == "None" and mapping['Time'] == "None":
        st.info("Map Hour and Time columns for trend analysis.")
        return

    c1, c2 = st.columns(2)

    with c1:
        if mapping['Hour'] != "None" and mapping['Time'] != "None":
            hour_avg = filtered_df.groupby(mapping['Hour'])[mapping['Time']].mean().reset_index()
            hour_avg['Hour_Fmt'] = hour_avg[mapping['Hour']].apply(format_hour)
            # Detect trend: compare first half vs second half of hours
            mid = len(hour_avg) // 2
            first_half_avg = hour_avg[mapping['Time']].iloc[:mid].mean()
            second_half_avg = hour_avg[mapping['Time']].iloc[mid:].mean()
            trend_dir = "↑ Rising" if second_half_avg > first_half_avg else "↓ Falling"
            trend_color = "#EF4444" if second_half_avg > first_half_avg else "#10B981"

            fig = px.line(hour_avg, x='Hour_Fmt', y=mapping['Time'],
                         title="Delivery Time by Hour", template='plotly_dark', markers=True)
            fig.update_traces(line=dict(color='#f59e0b', width=2.5), marker=dict(size=6))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis_title="Hour", yaxis_title="Avg Delivery Time (mins)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<div style='text-align:center; font-size:1.1rem; font-weight:700; color:{trend_color};'>"
                       f"Delivery Time Trend: {trend_dir}</div>", unsafe_allow_html=True)

    with c2:
        if mapping['Hour'] != "None" and mapping['Revenue'] != "None":
            rev_hour = filtered_df.groupby(mapping['Hour'])[mapping['Revenue']].sum().reset_index()
            rev_hour['Hour_Fmt'] = rev_hour[mapping['Hour']].apply(format_hour)
            mid = len(rev_hour) // 2
            r1 = rev_hour[mapping['Revenue']].iloc[:mid].sum()
            r2 = rev_hour[mapping['Revenue']].iloc[mid:].sum()
            rev_trend = "↑ Rising" if r2 > r1 else "↓ Falling"
            rev_color = "#10B981" if r2 > r1 else "#EF4444"

            fig2 = px.bar(rev_hour, x='Hour_Fmt', y=mapping['Revenue'],
                         title="Revenue by Hour", template='plotly_dark')
            fig2.update_traces(marker_color='#10b981')
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              xaxis_title="Hour", yaxis_title="Revenue")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(f"<div style='text-align:center; font-size:1.1rem; font-weight:700; color:{rev_color};'>"
                       f"Revenue Trend: {rev_trend}</div>", unsafe_allow_html=True)
        elif mapping['Hour'] != "None":
            hour_vol = filtered_df.groupby(mapping['Hour']).size().reset_index(name='Orders')
            hour_vol['Hour_Fmt'] = hour_vol[mapping['Hour']].apply(format_hour)
            fig2 = px.bar(hour_vol, x='Hour_Fmt', y='Orders',
                         title="Order Volume by Hour", template='plotly_dark')
            fig2.update_traces(marker_color='#8b5cf6')
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)


def analyst_outliers(filtered_df, mapping):
    st.markdown("<div class='section-header'>🚨 Outlier Detection — Top 10 Slowest Deliveries</div>", unsafe_allow_html=True)

    if mapping['Time'] == "None":
        st.info("Map Delivery Time column to detect outliers.")
        return

    cols_to_show = [mapping['Time']]
    rename_map = {mapping['Time']: "Delivery Time (mins)"}

    if mapping['Distance'] != "None":
        cols_to_show.append(mapping['Distance'])
        rename_map[mapping['Distance']] = "Distance (km)"
    if mapping['Hour'] != "None":
        cols_to_show.append(mapping['Hour'])
        rename_map[mapping['Hour']] = "Order Hour"
    if mapping['Rating'] != "None":
        cols_to_show.append(mapping['Rating'])
        rename_map[mapping['Rating']] = "Customer Rating"

    top_slow = filtered_df.nlargest(10, mapping['Time'])[cols_to_show].rename(columns=rename_map).copy()

    # Format hour column with 12h clock if present
    if "Order Hour" in top_slow.columns:
        top_slow["Order Hour"] = top_slow["Order Hour"].apply(lambda x: format_hour(int(x)) if pd.notna(x) else "N/A")

    overall_avg = filtered_df[mapping['Time']].mean()
    st.markdown(f"<p style='color:#94A3B8; margin-bottom:0.5rem;'>Overall average delivery time: "
               f"<strong style='color:#fff;'>{overall_avg:.1f} mins</strong> &nbsp;|&nbsp; "
               f"<strong style='color:#EF4444;'>Outlier threshold: {overall_avg + 1.5 * filtered_df[mapping['Time']].std():.1f} mins</strong></p>",
               unsafe_allow_html=True)

    ot1, ot2 = st.columns([1, 1])
    with ot1:
        st.dataframe(top_slow.reset_index(drop=True), use_container_width=True, hide_index=True)
    with ot2:
        if mapping['Distance'] != "None":
            plot_df = filtered_df[[mapping['Time'], mapping['Distance']]].copy()
            iqr_upper = filtered_df[mapping['Time']].quantile(0.75) + 1.5 * (filtered_df[mapping['Time']].quantile(0.75) - filtered_df[mapping['Time']].quantile(0.25))
            plot_df['Category'] = np.where(plot_df[mapping['Time']] > iqr_upper, 'Outlier', 'Normal')
            sample = plot_df.sample(min(600, len(plot_df)), random_state=42)
            fig_out = px.scatter(sample, x=mapping['Distance'], y=mapping['Time'],
                                 color='Category',
                                 color_discrete_map={'Outlier': '#EF4444', 'Normal': '#8b5cf6'},
                                 title="Outlier Scatter: Distance vs Delivery Time",
                                 template='plotly_dark', opacity=0.7)
            fig_out.add_hline(y=iqr_upper, line_dash='dash', line_color='#EF4444',
                              annotation_text='Outlier Threshold')
            fig_out.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis_title="Distance (km)", yaxis_title="Delivery Time (mins)")
            st.plotly_chart(fig_out, use_container_width=True)
        else:
            # Box plot showing distribution around outliers
            fig_box = px.box(filtered_df, y=mapping['Time'],
                             title="Delivery Time Box Plot",
                             template='plotly_dark',
                             color_discrete_sequence=['#8b5cf6'])
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, use_container_width=True)


def analyst_factor_impact(filtered_df, mapping):
    st.markdown("<div class='section-header'>🔍 Factor Impact Analysis</div>", unsafe_allow_html=True)

    if mapping['Time'] == "None":
        st.info("Map Delivery Time to analyse factor impact.")
        return

    target = mapping['Time']
    candidate_cols = {
        mapping.get('Distance'):  "Delivery Distance",
        mapping.get('Hour'):      "Order Hour",
        mapping.get('Revenue'):   "Order Value",
        mapping.get('Rating'):    "Customer Rating",
    }

    impact_rows = []
    for col, label in candidate_cols.items():
        if col and col != "None" and col in filtered_df.columns:
            corr = abs(filtered_df[col].corr(filtered_df[target]))
            if corr >= 0.5:
                level, color, bar_w = "Strong Impact",  "#EF4444", int(corr * 100)
            elif corr >= 0.25:
                level, color, bar_w = "Medium Impact",  "#F59E0B", int(corr * 100)
            else:
                level, color, bar_w = "Low Impact",     "#10B981", int(corr * 100)
            impact_rows.append({"label": label, "corr": corr, "level": level, "color": color, "bar_w": bar_w})

    if not impact_rows:
        st.info("Map more columns to see factor impact.")
        return

    impact_rows.sort(key=lambda x: x['corr'], reverse=True)

    fi_left, fi_right = st.columns([1, 1])
    with fi_left:
        for row in impact_rows:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border-radius:10px; padding:1rem 1.2rem;
                        margin-bottom:0.6rem; border:1px solid rgba(255,255,255,0.07);'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                    <span style='font-weight:700; font-size:1rem;'>{row["label"]}</span>
                    <span style='color:{row["color"]}; font-weight:600; font-size:0.9rem;'>{row["level"]}</span>
                </div>
                <div style='background:rgba(255,255,255,0.08); border-radius:6px; height:8px;'>
                    <div style='width:{row["bar_w"]}%; background:{row["color"]}; height:100%;
                                border-radius:6px; transition:width 0.8s;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with fi_right:
        imp_chart = pd.DataFrame({
            'Factor': [r['label'] for r in impact_rows],
            'Correlation': [round(r['corr'], 3) for r in impact_rows],
            'Level': [r['level'] for r in impact_rows]
        }).sort_values('Correlation', ascending=True)
        color_map_fi = {'Strong Impact': '#EF4444', 'Medium Impact': '#F59E0B', 'Low Impact': '#10B981'}
        fig_fi = px.bar(imp_chart, x='Correlation', y='Factor', orientation='h',
                       title='Factor Correlation Strength',
                       template='plotly_dark',
                       color='Level',
                       color_discrete_map=color_map_fi)
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis_title="Abs. Correlation with Delivery Time",
                             yaxis_title="", legend_title="Impact Level")
        st.plotly_chart(fig_fi, use_container_width=True)


def analyst_scenario_testing(filtered_df, df, mapping):
    st.markdown("<div class='section-header'>🔮 Scenario Testing — Delivery Predictor</div>", unsafe_allow_html=True)

    if mapping['Time'] == "None":
        st.info("Map Delivery Time to enable scenario testing.")
        return

    target = mapping['Time']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c != target and 'id' not in c.lower() and c not in ['Latitude', 'Longitude']]

    if len(features) == 0:
        st.error("No numeric feature columns available.")
        return

    # Use shared model state, train if needed
    if st.session_state.model_state is None:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        X = df[features].fillna(0)
        y = df[target].fillna(0)
        model.fit(X, y)
        st.session_state.model_state = {'model': model, 'features': features,
                                        'accuracy': model.score(X, y), 'avg_y': y.mean()}

    model    = st.session_state.model_state['model']
    features = st.session_state.model_state['features']
    avg_y    = st.session_state.model_state['avg_y']

    import textwrap

    col_in, col_out = st.columns([1, 1])
    inputs = {}
    hour_col = mapping.get('Hour') if mapping.get('Hour') != "None" else None
    dist_col = mapping.get('Distance') if mapping.get('Distance') != "None" else None
    sim_feats = [c for c in [dist_col, hour_col] if c and c in features]

    with col_in:
        st.markdown("**Adjust inputs to simulate a delivery scenario:**")
        for f in sim_feats:
            if f == hour_col:
                hour_vals   = sorted(df[f].dropna().unique().astype(int).tolist())
                hour_labels = [format_hour(h) for h in hour_vals]
                default_h   = min(hour_vals, key=lambda x: abs(x - int(round(df[f].mean()))))
                sel_label   = st.select_slider("Order Hour", options=hour_labels,
                                               value=format_hour(default_h),
                                               help="Select order time in 12-hour format")
                inputs[f] = float(hour_vals[hour_labels.index(sel_label)])
            else:
                inputs[f] = st.slider(
                    f"Distance (km)" if f == dist_col else f,
                    float(df[f].min()), float(df[f].max()), float(df[f].mean()),
                    help=f"Adjust {f}"
                )

    # Build prediction input
    input_df = pd.DataFrame([inputs])
    for missed in set(features) - set(inputs.keys()):
        input_df[missed] = df[missed].mean()

    pred  = model.predict(input_df[features])[0]
    delta = pred - avg_y
    pct   = (delta / avg_y) * 100 if avg_y > 0 else 0
    avg_t = df[target].mean()

    if pred < avg_t * 0.8:
        badge, badge_color = "Fast",     "#10B981"
    elif pred > avg_t * 1.2:
        badge, badge_color = "Slow",     "#EF4444"
    else:
        badge, badge_color = "Moderate", "#F59E0B"

    delta_color = "#EF4444" if delta > 0 else "#10B981"
    delta_sign  = "+" if delta > 0 else ""

    with col_out:
        st.markdown(f"""
        <div class='highlight-prediction'>
            <p style="color:rgba(255,255,255,0.7)!important; font-size:0.9rem; letter-spacing:1px; margin:0;">
                PREDICTED DURATION
            </p>
            <h2 style="margin:0.4rem 0;">{pred:.0f} <span style="font-size:1.4rem;">mins</span></h2>
            <div style="color:{delta_color}; font-weight:700; font-size:0.95rem; margin-bottom:0.5rem;">
                {delta_sign}{delta:.1f} mins ({delta_sign}{pct:.1f}%) vs. average
            </div>
            <span style="display:inline-block; padding:0.4rem 1.4rem; border-radius:20px;
                         background:{badge_color}; color:#fff; font-weight:700; font-size:1rem;">
                {badge}
            </span>
        </div>
        """, unsafe_allow_html=True)


def analyst_insights_panel(filtered_df, mapping):
    st.markdown("<div class='section-header'>🧩 Key Findings & Suggested Actions</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    findings_html = ""
    actions_html  = ""

    peak_h_label = "N/A"
    main_issue   = "No significant issue detected"
    risk         = ("Low", "#10B981")

    if mapping['Hour'] != "None":
        hour_vol   = filtered_df.groupby(mapping['Hour']).size()
        peak_h_label = format_hour(hour_vol.idxmax())

    if mapping['Time'] != "None":
        avg_t  = filtered_df[mapping['Time']].mean()
        threshold = avg_t + 10
        late_pct  = (filtered_df[mapping['Time']] > threshold).mean() * 100
        if late_pct > 25:
            main_issue = f"{late_pct:.0f}% of orders exceed delivery SLA"
            risk = ("High", "#EF4444")
            actions_html += "<li>Increase driver fleet or restrict delivery radius during peak hours</li>"
            actions_html += "<li>Implement order batching to reduce dispatch pressure</li>"
        elif late_pct > 10:
            main_issue = f"Mild delay rate of {late_pct:.0f}% — monitor closely"
            risk = ("Medium", "#F59E0B")
            actions_html += "<li>Review staffing levels at peak hours</li>"
        else:
            actions_html += "<li>Maintain current operational setup — performance is stable</li>"

    if mapping['Distance'] != "None" and mapping['Time'] != "None":
        corr = filtered_df[mapping['Distance']].corr(filtered_df[mapping['Time']])
        if corr > 0.4:
            actions_html += "<li>Reduce delivery radius for long-distance orders during peak hours</li>"

    if not actions_html:
        actions_html = "<li>Map more columns for detailed action recommendations</li>"

    findings_html = f"""
    <li>Peak order hour: <strong>{peak_h_label}</strong></li>
    <li>Main issue: <strong>{main_issue}</strong></li>
    <li>Risk level: <strong style='color:{risk[1]};'>{risk[0]}</strong></li>
    """

    with c1:
        st.markdown(f"""
        <div class='custom-card'>
            <h3>Key Findings</h3>
            <ul style='margin-top:0.8rem; padding-left:1.2rem; line-height:2;'>{findings_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='custom-card'>
            <h3>Suggested Actions</h3>
            <ul style='margin-top:0.8rem; padding-left:1.2rem; line-height:2;'>{actions_html}</ul>
        </div>
        """, unsafe_allow_html=True)


def analyst_data_explorer(filtered_df, mapping):
    st.markdown("<div class='section-header'>📋 Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Browse, sort, and inspect the active filtered dataset below.</p>",
                unsafe_allow_html=True)

    # Column selector
    all_cols  = filtered_df.columns.tolist()
    key_cols  = [v for v in [mapping.get('Time'), mapping.get('Distance'),
                              mapping.get('Hour'), mapping.get('Revenue'), mapping.get('Rating')]
                 if v and v != "None" and v in all_cols]
    shown_cols = st.multiselect("Select columns to display", all_cols,
                                default=key_cols if key_cols else all_cols[:6])
    if shown_cols:
        display_df = filtered_df[shown_cols].copy()
        # Format hour column
        hour_col = mapping.get('Hour')
        if hour_col and hour_col != "None" and hour_col in display_df.columns:
            display_df[hour_col] = display_df[hour_col].apply(
                lambda x: format_hour(int(x)) if pd.notna(x) else "N/A"
            )
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=400)
        st.caption(f"Showing {len(display_df):,} rows × {len(shown_cols)} columns")
    else:
        st.warning("Select at least one column to display.")


def render_analyst_mode(filtered_df, df, mapping):
    st.markdown("<h2 style='font-size:2.2rem; font-weight:900; margin-bottom:0.5rem;'>Analyst Dashboard</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; margin-bottom:2rem;'>Structured data analysis environment — "
                "understand patterns, causes, and actionable signals in your delivery data.</p>",
                unsafe_allow_html=True)

    tab_list = [
        "Overview",
        "Root Cause",
        "Segmentation",
        "Trends",
        "Outliers",
        "Factor Impact",
        "Scenario Testing",
        "Insights",
        "Data Explorer",
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        analyst_overview(filtered_df, mapping)
    with tabs[1]:
        analyst_root_cause(filtered_df, mapping)
    with tabs[2]:
        analyst_segmentation(filtered_df, mapping)
    with tabs[3]:
        analyst_trends(filtered_df, mapping)
    with tabs[4]:
        analyst_outliers(filtered_df, mapping)
    with tabs[5]:
        analyst_factor_impact(filtered_df, mapping)
    with tabs[6]:
        analyst_scenario_testing(filtered_df, df, mapping)
    with tabs[7]:
        analyst_insights_panel(filtered_df, mapping)
    with tabs[8]:
        analyst_data_explorer(filtered_df, mapping)



# ====== MAIN EXECUTION BLOCK ======
if uploaded_file is not None:
    if st.session_state.data is None:
        with st.spinner("Analyzing and validating data..."):
            time.sleep(0.5)
            try:
                df = pd.read_csv(uploaded_file)
                if len(df) == 0:
                    st.error("Uploaded CSV is empty. Please upload a valid dataset.")
                    st.stop()
                df, clean_report = clean_and_impute(df)
                st.session_state.data = df
                st.session_state.clean_report = clean_report
                st.session_state.column_mapping = auto_detect_columns(df.columns.tolist())
                st.session_state.alert_time = time.time()  # reset alert timer for new dataset
                st.rerun()
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                
    df = st.session_state.data
    columns = df.columns.tolist()
    
    st.sidebar.success(f"Dataset Loaded: {len(df):,} rows")
    
    # Validation Mapping Tab
    if not st.session_state.mapped:
        st.markdown("<div class='stagger-2 validation-card'>", unsafe_allow_html=True)
        st.markdown("<div class='validation-header'>⚠️ Validation: Map Dataset Columns</div>", unsafe_allow_html=True)
        st.write("Review our auto-detection below. Map your columns to generate the optimal analytics layout.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.column_mapping['Distance'] = st.selectbox("Distance", ["None"] + columns, index=(columns.index(st.session_state.column_mapping.get('Distance')) + 1) if st.session_state.column_mapping.get('Distance') in columns else 0)
            st.session_state.column_mapping['Revenue'] = st.selectbox("Revenue / Order Value", ["None"] + columns, index=(columns.index(st.session_state.column_mapping.get('Revenue')) + 1) if st.session_state.column_mapping.get('Revenue') in columns else 0)
        with col2:
            st.session_state.column_mapping['Time'] = st.selectbox("Delivery Duration/Time", ["None"] + columns, index=(columns.index(st.session_state.column_mapping.get('Time')) + 1) if st.session_state.column_mapping.get('Time') in columns else 0)
            st.session_state.column_mapping['Rating'] = st.selectbox("Customer Rating", ["None"] + columns, index=(columns.index(st.session_state.column_mapping.get('Rating')) + 1) if st.session_state.column_mapping.get('Rating') in columns else 0)
        with col3:
            st.session_state.column_mapping['Hour'] = st.selectbox("Order Hour / Temporal", ["None"] + columns, index=(columns.index(st.session_state.column_mapping.get('Hour')) + 1) if st.session_state.column_mapping.get('Hour') in columns else 0)
        
        mapped_count = sum([1 for k, v in st.session_state.column_mapping.items() if v != "None" and v is not None])
        score = mapped_count / 5.0
        st.write(f"**Analytics Engine Readiness Score:** {int(score * 100)}%")
        st.progress(score)
        
        if st.button("Confirm Mapping & Launch", use_container_width=True):
            st.session_state.mapped = True
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # Dashboard is active
        mapping = st.session_state.column_mapping
        st.sidebar.markdown("---")
        mode = st.sidebar.radio("View Mode", ["Manager Mode", "Analyst Mode"])

        if st.sidebar.button("Reset Dataset"):
            st.session_state.data = None
            st.session_state.mapped = False
            st.session_state.column_mapping = {}
            st.session_state.model_state = None
            st.session_state.clean_report = []
            st.rerun()

        # ── Dataset Health Check (shown once after mapping, always accessible via expander) ──
        validation_result = display_validation_report(
            mapping, df,
            clean_report=st.session_state.get('clean_report', [])
        )
        flags = validation_result['flags']
        st.markdown("<br>", unsafe_allow_html=True)
        st.sidebar.markdown("### 🎛️ Interactive Filters")
        filtered_df = df.copy()

        # ── Build active filter set ──
        _auto_cols = {'Location', 'Latitude', 'Longitude'}
        total_rows = len(df)

        # All categorical columns worth filtering (not auto-generated, not pure IDs, not near-unique)
        cat_cols = [
            c for c in df.select_dtypes(include=['object', 'category']).columns.tolist()
            if c not in _auto_cols
            and not re.search(r'(?i)(^id$|_id$)', c)
            and df[c].nunique() <= max(50, total_rows * 0.15)
        ]

        active_filters = 0

        # ── Section: Categorical Filters (one multiselect per column) ──
        if cat_cols:
            st.sidebar.markdown("**🏷️ Category Filters**")
            for col in cat_cols:
                options = sorted(filtered_df[col].dropna().unique().tolist())
                if not options:
                    continue
                selected = st.sidebar.multiselect(
                    col.replace('_', ' ').title(),
                    options,
                    default=[],
                    key=f"cat_filter_{col}"
                )
                if selected:
                    filtered_df = filtered_df[filtered_df[col].isin(selected)]
                    active_filters += 1

        st.sidebar.markdown("---")

        # ── Section: Hour Filter ──
        if flags['has_hour'] and mapping.get('Hour') not in (None, "None"):
            hour_col = mapping['Hour']
            all_hours = sorted(df[hour_col].dropna().unique().astype(int).tolist())
            hour_labels = {h: format_hour(h) for h in all_hours}
            st.sidebar.markdown("**🕐 Order Hour**")
            selected_hours = st.sidebar.multiselect(
                "Select hours",
                options=all_hours,
                format_func=lambda h: hour_labels.get(h, str(h)),
                default=[],
                key="hour_filter"
            )
            if selected_hours:
                filtered_df = filtered_df[filtered_df[hour_col].isin(selected_hours)]
                active_filters += 1

        # ── Section: Delivery Time Range ──
        if flags['has_time'] and mapping.get('Time') not in (None, "None"):
            min_t = float(df[mapping['Time']].min())
            max_t = float(df[mapping['Time']].max())
            if min_t < max_t:
                st.sidebar.markdown("**⏱️ Delivery Time (mins)**")
                time_range = st.sidebar.slider(
                    "Range", min_t, max_t, (min_t, max_t),
                    format="%.0f min", key="time_slider"
                )
                if time_range != (min_t, max_t):
                    filtered_df = filtered_df[
                        (filtered_df[mapping['Time']] >= time_range[0]) &
                        (filtered_df[mapping['Time']] <= time_range[1])
                    ]
                    active_filters += 1

        # ── Section: Distance Range ──
        if flags['has_distance'] and mapping.get('Distance') not in (None, "None"):
            min_d = float(df[mapping['Distance']].min())
            max_d = float(df[mapping['Distance']].max())
            if min_d < max_d:
                st.sidebar.markdown("**📍 Distance (km)**")
                dist_range = st.sidebar.slider(
                    "Range", min_d, max_d, (min_d, max_d),
                    format="%.1f km", key="dist_slider"
                )
                if dist_range != (min_d, max_d):
                    filtered_df = filtered_df[
                        (filtered_df[mapping['Distance']] >= dist_range[0]) &
                        (filtered_df[mapping['Distance']] <= dist_range[1])
                    ]
                    active_filters += 1

        # ── Section: Rating Filter ──
        if flags['has_rating'] and mapping.get('Rating') not in (None, "None"):
            rating_col = mapping['Rating']
            min_r = float(df[rating_col].min())
            max_r = float(df[rating_col].max())
            if min_r < max_r:
                st.sidebar.markdown("**⭐ Customer Rating**")
                rat_range = st.sidebar.slider(
                    "Min – Max", min_r, max_r, (min_r, max_r),
                    format="%.1f", key="rating_slider"
                )
                if rat_range != (min_r, max_r):
                    filtered_df = filtered_df[
                        (filtered_df[rating_col] >= rat_range[0]) &
                        (filtered_df[rating_col] <= rat_range[1])
                    ]
                    active_filters += 1

        # ── Reset notice ──
        if active_filters > 0:
            st.sidebar.markdown(
                f"<div style='background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3);"
                f"border-radius:8px; padding:0.5rem 0.8rem; font-size:0.82rem; color:#c4b5fd;"
                f"margin-top:0.5rem;'>🔽 {active_filters} filter(s) active — "
                f"{len(filtered_df):,} of {total_rows:,} rows shown</div>",
                unsafe_allow_html=True
            )

        if len(filtered_df) == 0:
            st.warning("Filters resulted in 0 rows. Please adjust sidebar filters.")
            st.stop()


        st.markdown("<br>", unsafe_allow_html=True)
        if mode == "Manager Mode":
            render_manager_mode(filtered_df, df, mapping)
        else:
            render_analyst_mode(filtered_df, df, mapping)



else:
    st.markdown("""
        <div class='upload-prompt stagger-2'>
            <span style='font-size: 4rem;'>📁</span>
            <h2>Ingest Analytics Data</h2>
            <p style='color: #A0AEC0;'>Drag and drop the system CSV package left via the sidebar.</p>
        </div>
    """, unsafe_allow_html=True)

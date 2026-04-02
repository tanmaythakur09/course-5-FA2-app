import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from scipy import stats
import os

# Set page config
st.set_page_config(
    page_title="FA-2: ATM Intelligence Dashboard",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tech-themed, advanced design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0f3460;
        --secondary: #16213e;
        --accent: #00d4ff;
        --danger: #ff006e;
        --success: #00ff88;
        --warning: #ffbe0b;
    }
    
    /* Global styles */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a1929 0%, #0f3460 50%, #132f4c 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        border-right: 2px solid #00d4ff;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(15, 52, 96, 0.8), rgba(22, 33, 62, 0.8));
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.05));
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #0a1929;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ff88 0%, #00dd77 100%);
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.4);
        transform: scale(1.05);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div {
        background: rgba(22, 33, 62, 0.7);
        border: 2px solid rgba(0, 212, 255, 0.3);
        color: #e0e0e0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stMultiSelect > div > div > div:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        color: #00d4ff;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.2), rgba(255, 0, 110, 0.1));
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(15, 52, 96, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(0, 212, 255, 0.3);
        margin: 20px 0;
    }
    
    /* Success messages */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.5);
        border-radius: 8px;
        color: #00ff88;
    }
    
    /* Error messages */
    .stError {
        background: rgba(255, 0, 110, 0.1);
        border: 1px solid rgba(255, 0, 110, 0.5);
        border-radius: 8px;
        color: #ff006e;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: rgba(0, 212, 255, 0.2);
    }
    
    /* Animations */
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 10px rgba(0, 212, 255, 0.4); }
        50% { text-shadow: 0 0 20px rgba(0, 212, 255, 0.8); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .glow-text {
        animation: glow 3s ease-in-out infinite;
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }

    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Set style for visualizations
plt.style.use('dark_background')
sns.set_palette("husl")

# Title with animation
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;' class='slide-in'>
    <h1 style='font-size: 2.5em; background: linear-gradient(90deg, #00d4ff, #ff006e, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;'>
        🏧 ATM INTELLIGENCE DASHBOARD
    </h1>
    <p style='color: #00d4ff; font-size: 1.1em; letter-spacing: 2px;'>
        AI-Powered Demand Forecasting & Commerce Analytics
    </p>
    <div style='height: 2px; background: linear-gradient(90deg, transparent, #00d4ff, transparent); margin: 20px 0; border-radius: 1px;'></div>
</div>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv('atm_cash_management_dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Pre-compute anomalies for all data
        z_scores = np.abs(stats.zscore(df['Total_Withdrawals']))
        z_anomalies = z_scores > 3
        
        Q1 = df['Total_Withdrawals'].quantile(0.25)
        Q3 = df['Total_Withdrawals'].quantile(0.75)
        IQR = Q3 - Q1
        iqr_anomalies = (df['Total_Withdrawals'] < (Q1 - 1.5 * IQR)) | (df['Total_Withdrawals'] > (Q3 + 1.5 * IQR))
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_anomalies = iso_forest.fit_predict(df[['Total_Withdrawals']]) == -1
        
        df['Anomaly'] = z_anomalies | iqr_anomalies | iso_anomalies
        df['Anomaly_Score'] = z_scores
        
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'atm_cash_management_dataset.csv' exists.")
        return None

# Load the data
df = load_data()

if df is not None:
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.05)); 
                border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 10px; padding: 15px;'>
        <p style='color: #00ff88; font-size: 1em; margin: 0;'>
            ✓ <strong>Dataset Status:</strong> {len(df):,} records | {df['Date'].min().date()} to {df['Date'].max().date()} | {df['ATM_ID'].nunique()} ATMs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Sidebar Navigation
    st.sidebar.markdown("""
    <h2 style='color: #00d4ff; text-align: center; letter-spacing: 2px;'>📊 NAVIGATION</h2>
    <hr style='border-color: rgba(0, 212, 255, 0.3);'>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Analysis Stage:",
        [
            "🏠 Overview",
            "📈 Stage 3: EDA",
            "🎯 Stage 4: Clustering",
            "⚠️ Stage 5: Anomaly Detection",
            "🔧 Stage 6: Interactive Tools"
        ],
        label_visibility="collapsed"
    )
    
    # ============================================================================
    # PAGE: OVERVIEW
    # ============================================================================
    if page == "🏠 Overview":
        st.markdown("""
        <h2 style='text-align: center; color: #00d4ff; margin-top: 20px;'>📊 EXECUTIVE DASHBOARD</h2>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Total Records", f"{len(df):,}", delta="Complete Dataset")
        with col2:
            st.metric("🏧 Unique ATMs", f"{df['ATM_ID'].nunique()}", delta="Monitored")
        with col3:
            st.metric("📅 Days Analyzed", f"{(df['Date'].max() - df['Date'].min()).days}", delta="2-Year Period")
        with col4:
            st.metric("📍 Locations", f"{df['Location_Type'].nunique()}", delta="Diverse Network")
        
        st.markdown("---")
        
        # Quick Statistics
        st.markdown("""
        <h3 style='color: #00d4ff; margin-top: 30px;'>💰 FINANCIAL METRICS</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(15, 52, 96, 0.8), rgba(22, 33, 62, 0.8)); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #00d4ff;'>
                <p style='color: #00d4ff; font-size: 0.9em; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Average Daily Withdrawals</p>
                <h2 style='color: #00ff88; margin: 10px 0 0 0; font-size: 1.8em;'>₹{df['Total_Withdrawals'].mean():,.0f}</h2>
                <p style='color: #888; font-size: 0.85em; margin: 5px 0 0 0;'>Std Dev: ₹{df['Total_Withdrawals'].std():,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(15, 52, 96, 0.8), rgba(22, 33, 62, 0.8)); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #ff006e;'>
                <p style='color: #ff006e; font-size: 0.9em; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Average Daily Deposits</p>
                <h2 style='color: #ffbe0b; margin: 10px 0 0 0; font-size: 1.8em;'>₹{df['Total_Deposits'].mean():,.0f}</h2>
                <p style='color: #888; font-size: 0.85em; margin: 5px 0 0 0;'>Std Dev: ₹{df['Total_Deposits'].std():,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(15, 52, 96, 0.8), rgba(22, 33, 62, 0.8)); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #00ff88;'>
                <p style='color: #00ff88; font-size: 0.9em; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Next-Day Avg Demand</p>
                <h2 style='color: #00ff88; margin: 10px 0 0 0; font-size: 1.8em;'>₹{df['Cash_Demand_Next_Day'].mean():,.0f}</h2>
                <p style='color: #888; font-size: 0.85em; margin: 5px 0 0 0;'>Std Dev: ₹{df['Cash_Demand_Next_Day'].std():,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Preview
        st.markdown("""
        <h3 style='color: #00d4ff;'>📋 RECENT TRANSACTIONS</h3>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(15), use_container_width=True)
    
    # ============================================================================
    # STAGE 3: EXPLORATORY DATA ANALYSIS
    # ============================================================================
    elif page == "📈 Stage 3: EDA":
        st.markdown("""
        <h2 style='text-align: center; color: #00d4ff;'>📊 EXPLORATORY DATA ANALYSIS</h2>
        <p style='text-align: center; color: #888;'>Understanding data distributions, trends, and behavioral patterns</p>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Section: Distribution Analysis
        st.markdown("""
        <h3 style='color: #00d4ff;'>📈 Section 3.1: Distribution Analysis</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.hist(df['Total_Withdrawals'], bins=40, color='#00d4ff', edgecolor='#00ff88', alpha=0.7, linewidth=1.5)
            ax.axvline(df['Total_Withdrawals'].mean(), color='#ff006e', linestyle='--', linewidth=2.5, 
                      label=f'Mean: ₹{df["Total_Withdrawals"].mean():,.0f}')
            ax.set_title('💸 Withdrawal Distribution', fontweight='bold', fontsize=12, color='#00d4ff', pad=15)
            ax.set_xlabel('Amount (₹)', color='#888')
            ax.set_ylabel('Frequency', color='#888')
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='#00d4ff')
            ax.grid(alpha=0.2, color='#00d4ff')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.hist(df['Total_Deposits'], bins=40, color='#00ff88', edgecolor='#ffbe0b', alpha=0.7, linewidth=1.5)
            ax.axvline(df['Total_Deposits'].mean(), color='#ff006e', linestyle='--', linewidth=2.5,
                      label=f'Mean: ₹{df["Total_Deposits"].mean():,.0f}')
            ax.set_title('💰 Deposit Distribution', fontweight='bold', fontsize=12, color='#00ff88', pad=15)
            ax.set_xlabel('Amount (₹)', color='#888')
            ax.set_ylabel('Frequency', color='#888')
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='#00ff88')
            ax.grid(alpha=0.2, color='#00ff88')
            st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Section: Time-Based Trends
        st.markdown("""
        <h3 style='color: #00d4ff;'>⏰ Section 3.2: Time-Based Trends</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_data = df.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(day_order)
            
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            colors = ['#00d4ff' if day not in ['Saturday', 'Sunday'] else '#ff006e' for day in day_order]
            bars = ax.bar(range(len(day_data)), day_data.values, color=colors, edgecolor='#00ff88', linewidth=1.5, alpha=0.8)
            ax.set_xticks(range(len(day_data)))
            ax.set_xticklabels([d[:3] for d in day_order], color='#888')
            ax.set_title('📅 Daily Withdrawal Patterns', fontweight='bold', fontsize=12, color='#00d4ff', pad=15)
            ax.set_ylabel('Amount (₹)', color='#888')
            ax.grid(alpha=0.2, axis='y', color='#00d4ff')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            time_data = df.groupby('Time_of_Day')['Total_Withdrawals'].mean().reindex(time_order)
            
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            colors_time = ['#FFD700', '#FFA500', '#FF6347', '#191970']
            ax.bar(range(len(time_data)), time_data.values, color=colors_time, edgecolor='#00ff88', linewidth=1.5, alpha=0.8)
            ax.set_xticks(range(len(time_data)))
            ax.set_xticklabels(time_order, color='#888')
            ax.set_title('🕐 Hourly Withdrawal Patterns', fontweight='bold', fontsize=12, color='#ffbe0b', pad=15)
            ax.set_ylabel('Amount (₹)', color='#888')
            ax.grid(alpha=0.2, axis='y', color='#ffbe0b')
            st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Section: Holiday & Event Impact
        st.markdown("""
        <h3 style='color: #00d4ff;'>🎉 Section 3.3: Holiday & Event Impact</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            holiday_data = df.groupby('Holiday_Flag')['Total_Withdrawals'].mean()
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            bars = ax.bar(['Normal Days', 'Holiday Days'], [holiday_data[0], holiday_data[1]], 
                         color=['#00d4ff', '#ff006e'], edgecolor='#00ff88', linewidth=2, width=0.6, alpha=0.8)
            ax.set_title('🏖️ Holiday Impact Analysis', fontweight='bold', fontsize=12, color='#ff006e', pad=15)
            ax.set_ylabel('Average Withdrawals (₹)', color='#888')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'₹{height:,.0f}', 
                       ha='center', va='bottom', fontweight='bold', color='#00ff88')
            ax.grid(alpha=0.2, axis='y', color='#ff006e')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            event_data = df.groupby('Special_Event_Flag')['Total_Withdrawals'].mean()
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            bars = ax.bar(['Normal Days', 'Event Days'], [event_data[0], event_data[1]], 
                         color=['#00ff88', '#ffbe0b'], edgecolor='#00d4ff', linewidth=2, width=0.6, alpha=0.8)
            ax.set_title('🎪 Special Event Impact', fontweight='bold', fontsize=12, color='#ffbe0b', pad=15)
            ax.set_ylabel('Average Withdrawals (₹)', color='#888')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'₹{height:,.0f}', 
                       ha='center', va='bottom', fontweight='bold', color='#0a1929')
            ax.grid(alpha=0.2, axis='y', color='#ffbe0b')
            st.pyplot(fig, use_container_width=True)
    
    # ============================================================================
    # STAGE 4: CLUSTERING ANALYSIS
    # ============================================================================
    elif page == "🎯 Stage 4: Clustering":
        st.markdown("""
        <h2 style='text-align: center; color: #00d4ff;'>🎯 K-MEANS CLUSTERING ANALYSIS</h2>
        <p style='text-align: center; color: #888;'>Identifying distinct ATM demand segments and behavioral clusters</p>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Feature Selection and Standardization
        clustering_features = ['Total_Withdrawals', 'Total_Deposits', 'Nearby_Competitor_ATMs']
        location_encoded = pd.factorize(df['Location_Type'])[0]
        
        X_clustering = df[clustering_features].copy()
        X_clustering['Location_Type_Encoded'] = location_encoded
        X_clustering = X_clustering.fillna(X_clustering.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clustering)
        
        # Determine Optimal K
        st.markdown("""
        <h3 style='color: #00d4ff;'>🔍 Section 4.2: Optimal Cluster Determination</h3>
        """, unsafe_allow_html=True)
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))
        
        # Plot metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.plot(K_range, inertias, marker='o', color='#00d4ff', linewidth=2.5, markersize=10, markerfacecolor='#ff006e', markeredgecolor='#00d4ff', markeredgewidth=2)
            ax.fill_between(K_range, inertias, alpha=0.2, color='#00d4ff')
            ax.set_title('📉 Elbow Method', fontweight='bold', fontsize=12, color='#00d4ff', pad=15)
            ax.set_xlabel('Number of Clusters (k)', color='#888')
            ax.set_ylabel('Inertia', color='#888')
            ax.grid(alpha=0.2, color='#00d4ff')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.plot(K_range, silhouette_scores, marker='s', color='#00ff88', linewidth=2.5, markersize=10, markerfacecolor='#ffbe0b', markeredgecolor='#00ff88', markeredgewidth=2)
            ax.fill_between(K_range, silhouette_scores, alpha=0.2, color='#00ff88')
            ax.set_title('⭐ Silhouette Score', fontweight='bold', fontsize=12, color='#00ff88', pad=15)
            ax.set_xlabel('Number of Clusters (k)', color='#888')
            ax.set_ylabel('Score', color='#888')
            ax.grid(alpha=0.2, color='#00ff88')
            st.pyplot(fig, use_container_width=True)
        
        with col3:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.plot(K_range, davies_bouldin_scores, marker='^', color='#ff006e', linewidth=2.5, markersize=10, markerfacecolor='#00d4ff', markeredgecolor='#ff006e', markeredgewidth=2)
            ax.fill_between(K_range, davies_bouldin_scores, alpha=0.2, color='#ff006e')
            ax.set_title('📊 Davies-Bouldin Index', fontweight='bold', fontsize=12, color='#ff006e', pad=15)
            ax.set_xlabel('Number of Clusters (k)', color='#888')
            ax.set_ylabel('Index (Lower Better)', color='#888')
            ax.grid(alpha=0.2, color='#ff006e')
            st.pyplot(fig, use_container_width=True)
        
        # Optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1)); 
                    border: 2px solid #00ff88; border-radius: 10px; padding: 15px; text-align: center;'>
            <h3 style='color: #00ff88; margin: 0;'>✓ OPTIMAL k = {optimal_k} CLUSTERS</h3>
            <p style='color: #00d4ff; margin: 5px 0 0 0; font-size: 1.1em;'>Silhouette Score: <strong>{max(silhouette_scores):.3f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Apply K-Means
        st.markdown("""
        <h3 style='color: #00d4ff;'>📊 Section 4.3 & 4.4: Cluster Characteristics</h3>
        """, unsafe_allow_html=True)
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Display cluster profiles as expandable sections
        cols = st.columns(2)
        col_idx = 0
        
        for cluster in range(optimal_k):
            cluster_df = df[df['Cluster'] == cluster]
            avg_withdrawal = cluster_df['Total_Withdrawals'].mean()
            avg_demand = cluster_df['Cash_Demand_Next_Day'].mean()
            location_mode = cluster_df['Location_Type'].mode()[0]
            
            with cols[col_idx % 2]:
                with st.expander(f"🎯 Cluster {cluster} · {len(cluster_df):,} records ({len(cluster_df)/len(df)*100:.1f}%)", expanded=(cluster==0)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💸 Avg Withdrawal", f"₹{avg_withdrawal:,.0f}")
                    with col2:
                        st.metric("💰 Next-Day Demand", f"₹{avg_demand:,.0f}")
                    with col3:
                        st.metric("📍 Location", location_mode)
                    with col4:
                        st.metric("🏪 Competitors", f"{cluster_df['Nearby_Competitor_ATMs'].mean():.1f}")
            
            col_idx += 1
    
    # ============================================================================
    # STAGE 5: ANOMALY DETECTION
    # ============================================================================
    elif page == "⚠️ Stage 5: Anomaly Detection":
        st.markdown("""
        <h2 style='text-align: center; color: #ff006e;'>⚠️ ANOMALY DETECTION & RISK ANALYSIS</h2>
        <p style='text-align: center; color: #888;'>Advanced detection of unusual patterns and high-risk behaviors</p>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Display metrics
        anomaly_count = df['Anomaly'].sum()
        anomaly_rate = anomaly_count / len(df) * 100
        z_count = (df['Anomaly_Score'] > 3).sum()
        iqr_threshold_high = df['Total_Withdrawals'].quantile(0.75) + 1.5 * (df['Total_Withdrawals'].quantile(0.75) - df['Total_Withdrawals'].quantile(0.25))
        iqr_count = (df['Total_Withdrawals'] > iqr_threshold_high).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚨 Z-Score Anomalies", f"{z_count}", "Statistical")
        with col2:
            st.metric("📊 IQR Anomalies", f"{iqr_count}", "Range-Based")
        with col3:
            st.metric("🤖 ML-Detected", f"{anomaly_count - z_count - iqr_count}", "Isolation Forest")
        with col4:
            st.metric("⚠️ Total Risk", f"{anomaly_count}", f"{anomaly_rate:.1f}%")
        
        st.markdown("---")
        
        # Visualization
        st.markdown("""
        <h3 style='color: #ff006e;'>📻 Anomaly Timeline & Distribution</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            normal_data = df[~df['Anomaly']]
            anomaly_data = df[df['Anomaly']]
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.scatter(normal_data['Date'], normal_data['Total_Withdrawals'], 
                      alpha=0.4, s=15, color='#00d4ff', label='✓ Normal')
            ax.scatter(anomaly_data['Date'], anomaly_data['Total_Withdrawals'], 
                      alpha=0.9, s=80, color='#ff006e', marker='X', label='⚠️ Anomaly', edgecolors='#ffbe0b', linewidth=1.5)
            ax.set_title('🔴 Anomalies Over Time', fontweight='bold', fontsize=12, color='#ff006e', pad=15)
            ax.set_xlabel('Date', color='#888')
            ax.set_ylabel('Total Withdrawals (₹)', color='#888')
            ax.legend(loc='upper left', framealpha=0.9, edgecolor='#ff006e')
            ax.grid(alpha=0.2, color='#ff006e')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.hist(df['Anomaly_Score'], bins=50, color='#00d4ff', edgecolor='#00ff88', alpha=0.7, linewidth=1.5)
            ax.axvline(3, color='#ff006e', linestyle='--', linewidth=2.5, label='Threshold (±3σ)')
            ax.axvline(-3, color='#ff006e', linestyle='--', linewidth=2.5)
            ax.set_title('📈 Z-Score Distribution', fontweight='bold', fontsize=12, color='#00d4ff', pad=15)
            ax.set_xlabel('Z-Score', color='#888')
            ax.set_ylabel('Frequency', color='#888')
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='#00d4ff')
            ax.grid(alpha=0.2, color='#00d4ff')
            st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Holiday vs Normal
        st.markdown("""
        <h3 style='color: #ff006e;'>🏖️ Holiday vs Normal Day Analysis</h3>
        """, unsafe_allow_html=True)
        
        holiday_anomalies = df[df['Holiday_Flag'] == 1]['Anomaly'].sum()
        normal_anomalies = df[df['Holiday_Flag'] == 0]['Anomaly'].sum()
        holiday_total = df[df['Holiday_Flag'] == 1].shape[0]
        normal_total = df[df['Holiday_Flag'] == 0].shape[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏖️ Holiday Anomalies", holiday_anomalies, f"{holiday_anomalies/holiday_total*100:.1f}%" if holiday_total > 0 else "N/A")
        with col2:
            st.metric("📅 Normal Day Anomalies", normal_anomalies, f"{normal_anomalies/normal_total*100:.1f}%" if normal_total > 0 else "N/A")
    
    # ============================================================================
    # STAGE 6: INTERACTIVE TOOLS
    # ============================================================================
    elif page == "🔧 Stage 6: Interactive Tools":
        st.markdown("""
        <h2 style='text-align: center; color: #00ff88;'>🔧 ADVANCED INTERACTIVE ANALYTICS</h2>
        <p style='text-align: center; color: #888;'>Custom filters and intelligent analysis of ATM performance</p>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Ensure clustering is done
        if 'Cluster' not in df.columns:
            clustering_features = ['Total_Withdrawals', 'Total_Deposits', 'Nearby_Competitor_ATMs']
            location_encoded = pd.factorize(df['Location_Type'])[0]
            X_clustering = df[clustering_features].copy()
            X_clustering['Location_Type_Encoded'] = location_encoded
            X_clustering = X_clustering.fillna(X_clustering.mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clustering)
            
            silhouette_scores = []
            for k in range(2, 11):
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_temp.fit(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
            
            optimal_k = 2 + np.argmax(silhouette_scores)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Interactive Filtering
        st.markdown("""
        <h3 style='color: #00ff88;'>🎛️ Advanced Filtering Engine</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            day_filter = st.multiselect("📅 Day of Week:", sorted(df['Day_of_Week'].unique()), 
                                        default=sorted(df['Day_of_Week'].unique()), key="day_multiselect")
        
        with col2:
            time_filter = st.multiselect("⏰ Time of Day:", sorted(df['Time_of_Day'].unique()), 
                                         default=sorted(df['Time_of_Day'].unique()), key="time_multiselect")
        
        with col3:
            location_filter = st.multiselect("📍 Location Type:", sorted(df['Location_Type'].unique()), 
                                             default=sorted(df['Location_Type'].unique()), key="loc_multiselect")
        
        # Apply filters
        filtered_df = df.copy()
        if day_filter:
            filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(day_filter)]
        if time_filter:
            filtered_df = filtered_df[filtered_df['Time_of_Day'].isin(time_filter)]
        if location_filter:
            filtered_df = filtered_df[filtered_df['Location_Type'].isin(location_filter)]
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1)); 
                    border: 2px solid #00ff88; border-radius: 10px; padding: 15px; text-align: center;'>
            <p style='color: #00ff88; margin: 0; font-size: 1.1em;'>✓ <strong>{len(filtered_df):,}</strong> records ({len(filtered_df)/len(df)*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display filtered data statistics
        st.markdown("""
        <h3 style='color: #00ff88;'>📊 Filtered Dataset Analytics</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💸 Avg Withdrawals", f"₹{filtered_df['Total_Withdrawals'].mean():,.0f}", 
                     f"{((filtered_df['Total_Withdrawals'].mean() / df['Total_Withdrawals'].mean() - 1) * 100):+.1f}%")
        with col2:
            st.metric("💰 Avg Deposits", f"₹{filtered_df['Total_Deposits'].mean():,.0f}", 
                     f"{((filtered_df['Total_Deposits'].mean() / df['Total_Deposits'].mean() - 1) * 100):+.1f}%")
        with col3:
            st.metric("📊 Avg Demand", f"₹{filtered_df['Cash_Demand_Next_Day'].mean():,.0f}", 
                     f"{((filtered_df['Cash_Demand_Next_Day'].mean() / df['Cash_Demand_Next_Day'].mean() - 1) * 100):+.1f}%")
        with col4:
            anomaly_rate_filtered = filtered_df['Anomaly'].sum()/len(filtered_df)*100 if len(filtered_df) > 0 else 0
            st.metric("⚠️ Anomaly Rate", f"{anomaly_rate_filtered:.1f}%", 
                     f"{filtered_df['Anomaly'].sum()} cases")
        
        st.markdown("---")
        
        # High-Risk ATM Identification
        st.markdown("""
        <h3 style='color: #ff006e;'>🚨 High-Risk ATM Detection</h3>
        """, unsafe_allow_html=True)
        
        threshold_percentile = st.slider("Risk Threshold (Top % by anomaly rate):", 0, 100, 80, step=5)
        
        atm_anomaly_rate = filtered_df.groupby('ATM_ID')['Anomaly'].agg(['sum', 'count'])
        atm_anomaly_rate['rate'] = (atm_anomaly_rate['sum'] / atm_anomaly_rate['count'] * 100)
        atm_anomaly_rate = atm_anomaly_rate.sort_values('rate', ascending=False)
        
        threshold = np.percentile(atm_anomaly_rate['rate'], threshold_percentile)
        high_risk = atm_anomaly_rate[atm_anomaly_rate['rate'] >= threshold].head(20)
        
        st.write(f"**Top At-Risk ATMs** (Top {100-threshold_percentile}% by anomaly rate):")
        st.dataframe(high_risk, use_container_width=True)
        
        st.markdown("---")
        
        # Filtered data preview
        st.markdown("""
        <h3 style='color: #00d4ff;'>📋 Filtered Transaction Data</h3>
        """, unsafe_allow_html=True)
        
        st.dataframe(filtered_df.head(25), use_container_width=True)

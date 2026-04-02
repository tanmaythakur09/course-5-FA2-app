"""
================================================================================
FA-2: ATM INTELLIGENCE DASHBOARD - COMPLETE CODE REFERENCE
================================================================================
Run Locally: Complete Standalone Code Files
Date: April 2, 2026
Status: Production-Ready
================================================================================
"""

# ============================================================================
# FILE 1: streamlit_app.py - MAIN DASHBOARD APPLICATION
# ============================================================================
# Copy this entire code below and save as: streamlit_app.py
# Then run: streamlit run streamlit_app.py

"""
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

# Custom CSS for tech-themed design
st.markdown('''
<style>
    /* Theme colors */
    :root {
        --primary: #0f3460;
        --secondary: #16213e;
        --accent: #00d4ff;
        --danger: #ff006e;
        --success: #00ff88;
        --warning: #ffbe0b;
    }
    
    /* Global styles */
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a1929 0%, #0f3460 50%, #132f4c 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar */
    .stSidebar {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        border-right: 2px solid #00d4ff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    /* Metrics */
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
    
    /* Inputs */
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
    .stSelectbox > div > div > select:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    /* Expanders */
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
    
    /* Dividers */
    hr {
        border-color: rgba(0, 212, 255, 0.3);
        margin: 20px 0;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.5);
        border-radius: 8px;
        color: #00ff88;
    }
    
    .stError {
        background: rgba(255, 0, 110, 0.1);
        border: 1px solid rgba(255, 0, 110, 0.5);
        border-radius: 8px;
        color: #ff006e;
    }
</style>
''', unsafe_allow_html=True)

# Set plot style
plt.style.use('dark_background')
sns.set_palette("husl")

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('atm_cash_management_dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Pre-compute anomalies
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
        st.error("Dataset not found! Run create_sample_data.py first.")
        return None

# Title
st.markdown('''
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='font-size: 2.5em; background: linear-gradient(90deg, #00d4ff, #ff006e, #00ff88); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        ATM INTELLIGENCE DASHBOARD
    </h1>
    <p style='color: #00d4ff; font-size: 1.1em; letter-spacing: 2px;'>
        AI-Powered Demand Forecasting & Analytics Platform
    </p>
</div>
''', unsafe_allow_html=True)

# Load data
df = load_data()

if df is not None:
    st.markdown(f'''
    <div style='background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.05)); 
                border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 10px; padding: 15px;'>
        <p style='color: #00ff88; margin: 0;'>
            ✓ Dataset: {len(df):,} records | {df['Date'].min().date()} to {df['Date'].max().date()} | {df['ATM_ID'].nunique()} ATMs
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('''
    <h2 style='color: #00d4ff; text-align: center; letter-spacing: 2px;'>NAVIGATION</h2>
    <hr style='border-color: rgba(0, 212, 255, 0.3);'>
    ''', unsafe_allow_html=True)
    
    page = st.sidebar.radio("Select Stage:", ["Overview", "Stage 3: EDA", "Stage 4: Clustering", 
                                               "Stage 5: Anomaly", "Stage 6: Interactive"])
    
    # Overview
    if page == "Overview":
        st.markdown("<h2 style='text-align: center;'>EXECUTIVE DASHBOARD</h2>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Records", f"{len(df):,}")
        with col2: st.metric("Unique ATMs", f"{df['ATM_ID'].nunique()}")
        with col3: st.metric("Days", f"{(df['Date'].max() - df['Date'].min()).days}")
        with col4: st.metric("Locations", f"{df['Location_Type'].nunique()}")
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Avg Withdrawals**: ₹{df['Total_Withdrawals'].mean():,.0f}")
        with col2:
            st.markdown(f"**Avg Deposits**: ₹{df['Total_Deposits'].mean():,.0f}")
        with col3:
            st.markdown(f"**Avg Demand**: ₹{df['Cash_Demand_Next_Day'].mean():,.0f}")
        
        st.markdown("---")
        st.markdown("### Recent Transactions")
        st.dataframe(df.head(15), use_container_width=True)
    
    # Stage 3: EDA
    elif page == "Stage 3: EDA":
        st.markdown("<h2>EXPLORATORY DATA ANALYSIS</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.hist(df['Total_Withdrawals'], bins=40, color='#00d4ff', edgecolor='#00ff88', alpha=0.7)
            ax.axvline(df['Total_Withdrawals'].mean(), color='#ff006e', linestyle='--', linewidth=2, 
                      label=f'Mean: ₹{df["Total_Withdrawals"].mean():,.0f}')
            ax.set_title('Withdrawal Distribution', fontweight='bold', color='#00d4ff')
            ax.set_xlabel('Amount (₹)', color='#888')
            ax.set_ylabel('Frequency', color='#888')
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.hist(df['Total_Deposits'], bins=40, color='#00ff88', edgecolor='#ffbe0b', alpha=0.7)
            ax.axvline(df['Total_Deposits'].mean(), color='#ff006e', linestyle='--', linewidth=2,
                      label=f'Mean: ₹{df["Total_Deposits"].mean():,.0f}')
            ax.set_title('Deposit Distribution', fontweight='bold', color='#00ff88')
            ax.set_xlabel('Amount (₹)', color='#888')
            ax.set_ylabel('Frequency', color='#888')
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
    
    # Stage 4: Clustering
    elif page == "Stage 4: Clustering":
        st.markdown("<h2>K-MEANS CLUSTERING ANALYSIS</h2>", unsafe_allow_html=True)
        
        clustering_features = ['Total_Withdrawals', 'Total_Deposits', 'Nearby_Competitor_ATMs']
        location_encoded = pd.factorize(df['Location_Type'])[0]
        X_clustering = df[clustering_features].copy()
        X_clustering['Location_Type_Encoded'] = location_encoded
        X_clustering = X_clustering.fillna(X_clustering.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clustering)
        
        inertias, silhouette_scores, davies_bouldin_scores = [], [], []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.plot(K_range, inertias, marker='o', color='#00d4ff', linewidth=2.5, markersize=10)
            ax.fill_between(K_range, inertias, alpha=0.2, color='#00d4ff')
            ax.set_title('Elbow Method', fontweight='bold', color='#00d4ff')
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.plot(K_range, silhouette_scores, marker='s', color='#00ff88', linewidth=2.5, markersize=10)
            ax.set_title('Silhouette Score', fontweight='bold', color='#00ff88')
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
        
        with col3:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.plot(K_range, davies_bouldin_scores, marker='^', color='#ff006e', linewidth=2.5, markersize=10)
            ax.set_title('Davies-Bouldin Index', fontweight='bold', color='#ff006e')
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        st.markdown(f"### Optimal k = {optimal_k} clusters (Score: {max(silhouette_scores):.3f})")
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        for cluster in range(optimal_k):
            cluster_df = df[df['Cluster'] == cluster]
            with st.expander(f"Cluster {cluster} ({len(cluster_df):,} records)"):
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Avg Withdrawal", f"₹{cluster_df['Total_Withdrawals'].mean():,.0f}")
                with col2: st.metric("Avg Demand", f"₹{cluster_df['Cash_Demand_Next_Day'].mean():,.0f}")
                with col3: st.metric("Location", cluster_df['Location_Type'].mode()[0])
                with col4: st.metric("Competitors", f"{cluster_df['Nearby_Competitor_ATMs'].mean():.1f}")
    
    # Stage 5: Anomaly
    elif page == "Stage 5: Anomaly":
        st.markdown("<h2>ANOMALY DETECTION & RISK ANALYSIS</h2>", unsafe_allow_html=True)
        
        anomaly_count = df['Anomaly'].sum()
        anomaly_rate = anomaly_count / len(df) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Z-Score", f"{(df['Anomaly_Score'] > 3).sum()}")
        with col2: st.metric("IQR", f"{((df['Total_Withdrawals'] > df['Total_Withdrawals'].quantile(0.75) + 1.5 * (df['Total_Withdrawals'].quantile(0.75) - df['Total_Withdrawals'].quantile(0.25)))).sum()}")
        with col3: st.metric("ML-Detected", f"{anomaly_count - (df['Anomaly_Score'] > 3).sum()}")
        with col4: st.metric("Total Risk", f"{anomaly_count} ({anomaly_rate:.1f}%)")
        
        col1, col2 = st.columns(2)
        with col1:
            normal_data = df[~df['Anomaly']]
            anomaly_data = df[df['Anomaly']]
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.scatter(normal_data['Date'], normal_data['Total_Withdrawals'], alpha=0.4, s=15, color='#00d4ff', label='Normal')
            ax.scatter(anomaly_data['Date'], anomaly_data['Total_Withdrawals'], alpha=0.9, s=80, color='#ff006e', marker='X', label='Anomaly')
            ax.set_title('Anomalies Over Time', fontweight='bold', color='#ff006e')
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
            ax.set_facecolor('#0a1929')
            ax.hist(df['Anomaly_Score'], bins=50, color='#00d4ff', edgecolor='#00ff88', alpha=0.7)
            ax.axvline(3, color='#ff006e', linestyle='--', linewidth=2.5, label='Threshold (±3σ)')
            ax.set_title('Z-Score Distribution', fontweight='bold', color='#00d4ff')
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
    
    # Stage 6: Interactive
    elif page == "Stage 6: Interactive":
        st.markdown("<h2>ADVANCED INTERACTIVE ANALYTICS</h2>", unsafe_allow_html=True)
        
        if 'Cluster' not in df.columns:
            clustering_features = ['Total_Withdrawals', 'Total_Deposits', 'Nearby_Competitor_ATMs']
            location_encoded = pd.factorize(df['Location_Type'])[0]
            X_clustering = df[clustering_features].copy()
            X_clustering['Location_Type_Encoded'] = location_encoded
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clustering.fillna(X_clustering.mean()))
            
            silhouette_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            optimal_k = 2 + np.argmax(silhouette_scores)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        st.markdown("### Advanced Filtering")
        col1, col2, col3 = st.columns(3)
        with col1:
            day_filter = st.multiselect("Day of Week:", sorted(df['Day_of_Week'].unique()), 
                                        default=sorted(df['Day_of_Week'].unique()))
        with col2:
            time_filter = st.multiselect("Time of Day:", sorted(df['Time_of_Day'].unique()), 
                                         default=sorted(df['Time_of_Day'].unique()))
        with col3:
            location_filter = st.multiselect("Location:", sorted(df['Location_Type'].unique()), 
                                            default=sorted(df['Location_Type'].unique()))
        
        filtered_df = df.copy()
        if day_filter: filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(day_filter)]
        if time_filter: filtered_df = filtered_df[filtered_df['Time_of_Day'].isin(time_filter)]
        if location_filter: filtered_df = filtered_df[filtered_df['Location_Type'].isin(location_filter)]
        
        st.markdown(f"**Records: {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}%)**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Avg Withdrawals", f"₹{filtered_df['Total_Withdrawals'].mean():,.0f}")
        with col2: st.metric("Avg Deposits", f"₹{filtered_df['Total_Deposits'].mean():,.0f}")
        with col3: st.metric("Avg Demand", f"₹{filtered_df['Cash_Demand_Next_Day'].mean():,.0f}")
        with col4: st.metric("Anomaly Rate", f"{filtered_df['Anomaly'].sum()/len(filtered_df)*100:.1f}%")
        
        st.markdown("### High-Risk ATMs")
        threshold = st.slider("Risk Threshold:", 0, 100, 80, step=5)
        atm_anomaly = filtered_df.groupby('ATM_ID')['Anomaly'].agg(['sum', 'count'])
        atm_anomaly['rate'] = (atm_anomaly['sum'] / atm_anomaly['count'] * 100)
        
        threshold_val = np.percentile(atm_anomaly['rate'], threshold)
        high_risk = atm_anomaly[atm_anomaly['rate'] >= threshold_val].head(20)
        st.dataframe(high_risk, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Filtered Data")
        st.dataframe(filtered_df.head(25), use_container_width=True)
"""

================================================================================
# FILE 2: create_sample_data.py
================================================================================
# Save this as create_sample_data.py
# Run: python create_sample_data.py

"""
import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 1, 1)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

n_records = 5658
atm_ids = np.random.choice(range(1, 51), n_records)
dates = np.random.choice(date_range, n_records)

total_withdrawals = np.random.normal(49808, 14904, n_records)
total_withdrawals = np.clip(total_withdrawals, 1380, 107790)

total_deposits = np.random.normal(10129, 4879, n_records)
total_deposits = np.clip(total_deposits, 0, 32395)

previous_day_cash = np.random.normal(100000, 50000, n_records)
previous_day_cash = np.clip(previous_day_cash, 10000, 300000)

cash_demand_next_day = total_withdrawals * np.random.uniform(0.9, 1.1, n_records)

day_of_week_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
dates_pd = pd.to_datetime(dates)
day_of_week = [day_of_week_map[d.weekday()] for d in dates_pd]

time_choices = ['Morning', 'Afternoon', 'Evening', 'Night']
time_of_day = np.random.choice(time_choices, n_records)

location_choices = ['Standalone', 'Supermarket', 'Mall', 'Bank Branch', 'Gas Station']
location_type = np.random.choice(location_choices, n_records)

weather_choices = ['Clear', 'Cloudy', 'Rainy', 'Snowy']
weather_condition = np.random.choice(weather_choices, n_records)

nearby_competitors = np.random.randint(0, 10, n_records)
holiday_flag = np.random.choice([0, 1], n_records, p=[0.99, 0.01])
special_event_flag = np.random.choice([0, 1], n_records, p=[0.90, 0.10])

df = pd.DataFrame({
    'Date': dates,
    'ATM_ID': atm_ids,
    'Total_Withdrawals': total_withdrawals.round(2),
    'Total_Deposits': total_deposits.round(2),
    'Previous_Day_Cash_Level': previous_day_cash.round(2),
    'Cash_Demand_Next_Day': cash_demand_next_day.round(2),
    'Day_of_Week': day_of_week,
    'Time_of_Day': time_of_day,
    'Location_Type': location_type,
    'Weather_Condition': weather_condition,
    'Nearby_Competitor_ATMs': nearby_competitors,
    'Holiday_Flag': holiday_flag,
    'Special_Event_Flag': special_event_flag
})

df.to_csv('atm_cash_management_dataset.csv', index=False)
print(f"✓ Dataset created: {len(df)} records")
print(f"✓ File: atm_cash_management_dataset.csv")
"""

================================================================================
# QUICK START INSTRUCTIONS
================================================================================

1. INSTALL DEPENDENCIES:
   pip install pandas numpy matplotlib seaborn scikit-learn scipy streamlit

2. CREATE FILES:
   - Copy streamlit_app.py code into file named: streamlit_app.py
   - Copy create_sample_data.py code into file named: create_sample_data.py

3. GENERATE DATA:
   python create_sample_data.py

4. RUN DASHBOARD:
   streamlit run streamlit_app.py

5. OPEN IN BROWSER:
   http://localhost:8501

================================================================================
"""

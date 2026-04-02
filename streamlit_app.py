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

# Set page config
st.set_page_config(
    page_title="FA-2: ATM Intelligence Dashboard",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1929 0%, #0f3460 50%, #132f4c 100%);
        color: #e0e0e0;
    }
    .stSidebar {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        border-right: 2px solid #00d4ff;
    }
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        font-weight: 700;
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(15, 52, 96, 0.8), rgba(22, 33, 62, 0.8));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #0a1929;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ff88 0%, #00dd77 100%);
    }
</style>
''', unsafe_allow_html=True)

# Set plot style
plt.style.use('dark_background')
sns.set_palette("husl")

def generate_sample_dataset():
    """Generate sample ATM dataset"""
    np.random.seed(42)
    
    n_records = 5658
    atm_ids = [f'ATM_{i:03d}' for i in range(1, 51)]
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', periods=n_records)
    
    data = {
        'Date': dates,
        'ATM_ID': np.random.choice(atm_ids, n_records),
        'Day_of_Week': [d.strftime('%A') for d in dates],
        'Time_of_Day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_records),
        'Location_Type': np.random.choice(['Urban', 'Suburban', 'Rural'], n_records),
        'Total_Withdrawals': np.random.normal(50000, 15000, n_records).clip(10000, 150000),
        'Total_Deposits': np.random.normal(10000, 5000, n_records).clip(1000, 50000),
        'Weather_Condition': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], n_records),
        'Nearby_Competitor_ATMs': np.random.randint(0, 10, n_records),
        'Holiday_Flag': np.random.choice([0, 1], n_records, p=[0.9, 0.1]),
        'Special_Event': np.random.choice([0, 1], n_records, p=[0.95, 0.05]),
    }
    
    df = pd.DataFrame(data)
    df['Cash_Demand_Next_Day'] = df['Total_Withdrawals'].shift(-1).fillna(df['Total_Withdrawals'].mean())
    
    return df

@st.cache_data
def load_data():
    """Load or generate dataset"""
    try:
        df = pd.read_csv('atm_cash_management_dataset.csv')
    except FileNotFoundError:
        df = generate_sample_dataset()
    
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

# Title
st.markdown('''
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='font-size: 2.5em;'>ATM INTELLIGENCE DASHBOARD</h1>
    <p style='color: #00d4ff;'>AI-Powered Demand Forecasting & Analytics</p>
</div>
''', unsafe_allow_html=True)

# Load data
df = load_data()

st.markdown(f'''
<div style='background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.3); 
            border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
    <p style='color: #00ff88; margin: 0;'>
        [OK] Dataset: {len(df):,} records | {df['Date'].min().date()} to {df['Date'].max().date()} | {df['ATM_ID'].nunique()} ATMs
    </p>
</div>
''', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown('''
<h2 style='color: #00d4ff; text-align: center;'>NAVIGATION</h2>
<hr style='border-color: rgba(0, 212, 255, 0.3);'>
''', unsafe_allow_html=True)

page = st.sidebar.radio("Select Stage:", ["Overview", "Stage 3: EDA", "Stage 4: Clustering", 
                                          "Stage 5: Anomaly", "Stage 6: Interactive"])

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.markdown("<h2 style='text-align: center;'>EXECUTIVE DASHBOARD</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: st.metric("Total Records", f"{len(df):,}")
    with col2: st.metric("Unique ATMs", f"{df['ATM_ID'].nunique()}")
    with col3: st.metric("Days", f"{(df['Date'].max() - df['Date'].min()).days}")
    with col4: st.metric("Locations", f"{df['Location_Type'].nunique()}")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f"**Avg Withdrawals**: Rs{df['Total_Withdrawals'].mean():,.0f}")
    with col2: st.markdown(f"**Avg Deposits**: Rs{df['Total_Deposits'].mean():,.0f}")
    with col3: st.markdown(f"**Avg Demand**: Rs{df['Cash_Demand_Next_Day'].mean():,.0f}")
    
    st.markdown("---")
    st.markdown("### Recent Transactions")
    st.dataframe(df.head(15), use_container_width=True)

# ============================================================================
# PAGE: STAGE 3 - EDA
# ============================================================================
elif page == "Stage 3: EDA":
    st.markdown("<h2>EXPLORATORY DATA ANALYSIS</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
        ax.set_facecolor('#0a1929')
        ax.hist(df['Total_Withdrawals'], bins=40, color='#00d4ff', edgecolor='#00ff88', alpha=0.7)
        ax.axvline(df['Total_Withdrawals'].mean(), color='#ff006e', linestyle='--', linewidth=2)
        ax.set_title('Withdrawal Distribution', fontweight='bold', color='#00d4ff')
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
        ax.set_facecolor('#0a1929')
        ax.hist(df['Total_Deposits'], bins=40, color='#00ff88', edgecolor='#ffbe0b', alpha=0.7)
        ax.axvline(df['Total_Deposits'].mean(), color='#ff006e', linestyle='--', linewidth=2)
        ax.set_title('Deposit Distribution', fontweight='bold', color='#00ff88')
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE: STAGE 4 - CLUSTERING
# ============================================================================
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
        ax.plot(K_range, inertias, marker='o', color='#00d4ff', linewidth=2.5)
        ax.set_title('Elbow Method', fontweight='bold', color='#00d4ff')
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
        ax.set_facecolor('#0a1929')
        ax.plot(K_range, silhouette_scores, marker='s', color='#00ff88', linewidth=2.5)
        ax.set_title('Silhouette Score', fontweight='bold', color='#00ff88')
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
    
    with col3:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
        ax.set_facecolor('#0a1929')
        ax.plot(K_range, davies_bouldin_scores, marker='^', color='#ff006e', linewidth=2.5)
        ax.set_title('Davies-Bouldin Index', fontweight='bold', color='#ff006e')
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    st.markdown(f"### Optimal k = {optimal_k} clusters (Score: {max(silhouette_scores):.3f})")

# ============================================================================
# PAGE: STAGE 5 - ANOMALY
# ============================================================================
elif page == "Stage 5: Anomaly":
    st.markdown("<h2>ANOMALY DETECTION & RISK ANALYSIS</h2>", unsafe_allow_html=True)
    
    anomaly_count = df['Anomaly'].sum()
    anomaly_rate = anomaly_count / len(df) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Z-Score", f"{(df['Anomaly_Score'] > 3).sum()}")
    with col2: st.metric("Total Anomalies", f"{anomaly_count}")
    with col3: st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    with col4: st.metric("Normal Records", f"{len(df) - anomaly_count:,}")
    
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
        ax.axvline(3, color='#ff006e', linestyle='--', linewidth=2.5, label='Threshold (3 sigma)')
        ax.set_title('Z-Score Distribution', fontweight='bold', color='#00d4ff')
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE: STAGE 6 - INTERACTIVE
# ============================================================================
elif page == "Stage 6: Interactive":
    st.markdown("<h2>ADVANCED INTERACTIVE ANALYTICS</h2>", unsafe_allow_html=True)
    
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
    with col1: st.metric("Avg Withdrawals", f"Rs{filtered_df['Total_Withdrawals'].mean():,.0f}")
    with col2: st.metric("Avg Deposits", f"Rs{filtered_df['Total_Deposits'].mean():,.0f}")
    with col3: st.metric("Avg Demand", f"Rs{filtered_df['Cash_Demand_Next_Day'].mean():,.0f}")
    with col4: st.metric("Anomaly Rate", f"{filtered_df['Anomaly'].sum()/len(filtered_df)*100:.1f}%")
    
    st.markdown("### High-Risk ATMs")
    threshold = st.slider("Risk Threshold (%):", 0, 100, 80, step=5)
    atm_anomaly = filtered_df.groupby('ATM_ID')['Anomaly'].agg(['sum', 'count'])
    atm_anomaly['rate'] = (atm_anomaly['sum'] / atm_anomaly['count'] * 100)
    
    threshold_val = np.percentile(atm_anomaly['rate'], threshold)
    high_risk = atm_anomaly[atm_anomaly['rate'] >= threshold_val].head(20)
    st.dataframe(high_risk, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Filtered Data Preview")
    st.dataframe(filtered_df.head(25), use_container_width=True)

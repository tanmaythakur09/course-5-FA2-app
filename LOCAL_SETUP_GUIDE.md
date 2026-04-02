# 🏧 ATM Intelligence Dashboard - Local Setup Guide

## Quick Start (5 Minutes)

### Step 1: Install Python Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy streamlit
```

### Step 2: Generate Sample Data
```bash
python create_sample_data.py
```

### Step 3: Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### Step 4: Open in Browser
Navigate to: **http://localhost:8501**

---

## Project Structure

```
FA-2-ATM-Intelligence/
├── streamlit_app.py              # Main Streamlit dashboard
├── create_sample_data.py         # Generate sample dataset
├── FA2_Complete_Analysis_Script.py  # Full analysis script
├── atm_cash_management_dataset.csv  # Sample data (generated)
└── outputs/                      # Generated visualizations
    ├── EDA_01_Distribution_Analysis.png
    ├── EDA_02_Time_Based_Trends.png
    ├── EDA_03_Holiday_Event_Impact.png
    ├── EDA_04_External_Factors.png
    ├── EDA_05_Correlation_Heatmap.png
    ├── Clustering_01_Optimal_K.png
    ├── Clustering_02_Cluster_Visualization.png
    ├── Anomaly_01_Detection_Visualization.png
    └── Interactive_Summary_Dashboard.png
```

---

## Features

### 🏠 Overview Page
- Dataset statistics and metrics
- Financial summary (Withdrawals, Deposits, Cash Demand)
- Recent transaction preview

### 📈 Stage 3: EDA
- Distribution analysis (histograms)
- Time-based trends (daily & hourly patterns)
- Holiday & event impact analysis

### 🎯 Stage 4: Clustering
- Elbow method visualization
- Silhouette score analysis
- Davies-Bouldin index
- Cluster characteristics with expandable details

### ⚠️ Stage 5: Anomaly Detection
- Z-Score detection
- IQR-based anomalies
- Isolation Forest (ML-based)
- Anomaly timeline visualization
- Holiday vs normal day comparison

### 🔧 Stage 6: Interactive Tools
- Advanced filtering by day, time, location
- Real-time statistics
- High-risk ATM identification
- Customizable risk thresholds
- Filtered data preview

---

## Technology Stack

- **Python 3.8+**
- **Streamlit** - Web dashboard framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib** - Visualizations
- **Seaborn** - Statistical graphics
- **Scikit-learn** - Machine learning (K-Means, Isolation Forest)
- **SciPy** - Statistical functions

---

## Customization

### Change Sample Data Size
Edit `create_sample_data.py`:
```python
n_records = 10000  # Change from 5658
```

### Modify Color Scheme
Edit `streamlit_app.py` CSS section:
```python
--primary: #0f3460;
--accent: #00d4ff;
--danger: #ff006e;
--success: #00ff88;
```

### Adjust Clustering Parameters
Edit `streamlit_app.py`, Stage 4 section:
```python
K_range = range(2, 15)  # Change from range(2, 11)
```

---

## Troubleshooting

### Port Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Data Not Loading
Ensure `atm_cash_management_dataset.csv` is in the same directory as `streamlit_app.py`

### Missing Dependencies
```bash
pip install --upgrade streamlit pandas numpy matplotlib seaborn scikit-learn scipy
```

### Clear Cache
```bash
streamlit cache clear
```

---

## File Descriptions

### streamlit_app.py
Modern, tech-themed interactive dashboard with:
- Dark gradient backgrounds
- Smooth animations and hover effects
- Real-time data filtering
- Advanced analytics

### create_sample_data.py
Generates realistic ATM transaction data with:
- 5,658 records across 50 ATMs
- 2 years of transaction history
- Multiple location types
- Weather and behavioral factors

### FA2_Complete_Analysis_Script.py
Comprehensive analysis pipeline:
- EDA with 5 visualizations
- K-Means clustering
- Anomaly detection (3 methods)
- Interactive analysis functions

---

## Expected Output

When running `streamlit run streamlit_app.py`:

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://your-ip:8501
```

Dashboard loads with:
- 5,658 records | 50 ATMs | 2-year analysis period
- 8 optimal clusters identified
- 281 anomalies detected (5.0% rate)
- 9 high-quality visualizations

---

## Performance Notes

- First load: ~5-10 seconds (data caching)
- Subsequent loads: ~1-2 seconds
- Filtering: Real-time (< 100ms)
- Clustering: Auto-computed on optimal k determination

---

## Support & Reference

- **Streamlit Docs**: https://docs.streamlit.io
- **Scikit-learn**: https://scikit-learn.org
- **Pandas**: https://pandas.pydata.org
- **Matplotlib**: https://matplotlib.org

---

Last Updated: April 2, 2026
Version: 1.0.0 Production-Ready

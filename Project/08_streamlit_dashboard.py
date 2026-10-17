import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FinTech NLP Risk Engine", page_icon="🏦", layout="wide")

# --- DATA LOADING (CACHED FOR SPEED) ---
@st.cache_data
def load_data():
    csv_path = "Final_DeepLearning_Dataset.csv"
    matrix_path = "Compressed_TFIDF_Matrix.npy"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Cleaned_Content'] = df['Cleaned_Content'].fillna("[empty_review]").astype(str)
        # Calculate review lengths for EDA
        df['Word_Count'] = df['Cleaned_Content'].apply(lambda x: len(x.split()))
    else:
        df = pd.DataFrame()
        
    if os.path.exists(matrix_path):
        svd_matrix = np.load(matrix_path)
    else:
        svd_matrix = None
        
    return df, svd_matrix

df, svd_matrix = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏦 FinTech Twin-Model Engine")
st.sidebar.markdown("Explore the mathematical breakdown of E-Satisfaction vs. Risk Perception in Indian Mobile Banking.")
page = st.sidebar.radio("Select Analytics View:", 
                        ["1. Project Overview & Demographics", 
                         "2. Semantic 3D Clustering (SVD)", 
                         "3. Volatility & Risk Analysis"])

st.sidebar.markdown("---")
st.sidebar.info("Developed by Jayesh | PAAML504 Data Science Project")

# --- ERROR HANDLING ---
if df.empty:
    st.error("🚨 Dataset not found! Please run pre_process.py first.")
    st.stop()

# ==========================================
# PAGE 1: DEMOGRAPHICS & EDA
# ==========================================
if page == "1. Project Overview & Demographics":
    st.title("📊 Sector Demographics & Data Shape")
    st.markdown("Visualizing the raw institutional imbalance between Government and Private banking sectors.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Review Volume by Sector")
        # Interactive Pie Chart
        sector_counts = df['Bank_Type'].value_counts().reset_index()
        sector_counts.columns = ['Bank_Type', 'Count']
        fig_pie = px.pie(sector_counts, names='Bank_Type', values='Count', 
                         color='Bank_Type', color_discrete_map={'Government':'#2b5c8f', 'Private':'#dc3545'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Distribution of Review Lengths (Word Count)")
        st.markdown("Notice how extreme outliers (Financial Panic) stretch the tail of the graph.")
        # Taking a sample of 10,000 for faster browser rendering
        fig_hist = px.histogram(df.sample(10000, random_state=42), x="Word_Count", color="Bank_Type", 
                                marginal="box", nbins=50, 
                                color_discrete_map={'Government':'#2b5c8f', 'Private':'#dc3545'},
                                opacity=0.7)
        fig_hist.update_layout(xaxis_range=[0, 80])
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# PAGE 2: 3D SVD CLUSTERING
# ==========================================
elif page == "2. Semantic 3D Clustering (SVD)":
    st.title("🌌 High-Dimensional Text Geometry (Truncated SVD)")
    st.markdown("We compressed 10,000 sparse TF-IDF columns down to core principal components. Below is a 3D projection of how the algorithm autonomously groups review sentiments without human labels.")
    
    if svd_matrix is not None:
        # Sample 1,500 points for a smooth 3D render in the browser
        sample_size = min(1500, len(df))
        np.random.seed(42)
        idx = np.random.choice(len(df), sample_size, replace=False)
        
        plot_df = pd.DataFrame({
            'Component 1 (Variance)': svd_matrix[idx, 0],
            'Component 2 (Variance)': svd_matrix[idx, 1],
            'Component 3 (Variance)': svd_matrix[idx, 2],
            'Bank_Type': df.iloc[idx]['Bank_Type'].values,
            'Snippet': df.iloc[idx]['Cleaned_Content'].str[:60] + "..."
        })
        
        fig_3d = px.scatter_3d(plot_df, x='Component 1 (Variance)', y='Component 2 (Variance)', z='Component 3 (Variance)',
                               color='Bank_Type', hover_name='Snippet',
                               color_discrete_map={'Government':'#2b5c8f', 'Private':'#dc3545'},
                               opacity=0.6, size_max=5)
        
        # Increase the size of the 3D plot
        fig_3d.update_layout(height=700)
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("⚠️ Compressed_TFIDF_Matrix.npy not found. Run 04_nlp_compression.py first.")

# ==========================================
# PAGE 3: VOLATILITY & RISK (CHEBYSHEV INSIGHTS)
# ==========================================
elif page == "3. Volatility & Risk Analysis":
    st.title("📈 Concentration Inequalities & Risk Volatility")
    st.markdown("Using Chebyshev's bounds, we mathematically proved that **Government Banks** exhibit higher volatility in complaint severity (Std Dev: 13.40) compared to Private Banks (12.84).")
    
    # Calculate Mean and Std Dev dynamically
    govt_data = df[df['Bank_Type'] == 'Government']['Word_Count']
    pvt_data = df[df['Bank_Type'] == 'Private']['Word_Count']
    
    colA, colB = st.columns(2)
    colA.metric(label="Government Banks (Volatility / Std Dev)", value=f"{govt_data.std():.2f}", delta="Higher Panic Variance", delta_color="inverse")
    colB.metric(label="Private Banks (Volatility / Std Dev)", value=f"{pvt_data.std():.2f}", delta="Tighter Variance", delta_color="normal")
    
    st.divider()
    
    st.subheader("Top Structural Complaints (PageRank Centrality Recap)")
    st.markdown("""
    Based on our Markov Chain Power Iterations (Phase 7):
    * **Government Sector:** High centrality around infrastructure collapse (`worst`, `working`, `update`).
    * **Private Sector:** High centrality around financial friction (`account`, `open`, `card`).
    """)
    st.info("💡 The Machine Learning Twin-Model Cascade will now utilize these mathematical boundaries to automatically escalate active system failures in real-time.")
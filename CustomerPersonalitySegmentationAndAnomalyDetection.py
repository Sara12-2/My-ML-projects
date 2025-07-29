import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🧠 Customer Segmentation & Anomaly Detection")

# 📥 Load dataset
df = pd.read_csv("marketing_campaign.csv", sep='\t')

# Drop nulls
df = df.dropna()

# Convert date column
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

# Drop non-numeric columns
features = df.drop(columns=['Response', 'Dt_Customer'])

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
#elbow method
inertia=[]
k_range = range(1, 11)  
for k in k_range:
    model=KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)
#plotting
# plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Finding Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within Cluster Sum of Squares)')
plt.grid(True)
# plt.show()
  

# KMeans clustering (k=4 from elbow)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)
df['Cluster'] = kmeans.labels_

# Anomaly Detection
iso = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = iso.fit_predict(scaled_data)
# --- Streamlit Dashboard Section ---

st.sidebar.header("🔍 Filter Options")

# Filter cluster view
clusters = st.sidebar.multiselect("Select Cluster(s)", df['Cluster'].unique(), default=df['Cluster'].unique())
anomaly_view = st.sidebar.selectbox("Anomaly Filter", ["All", "Normal Only", "Anomalies Only"])

# Filter Data
filtered_df = df[df['Cluster'].isin(clusters)]
if anomaly_view == "Normal Only":
    filtered_df = filtered_df[filtered_df['Anomaly'] == 1]
elif anomaly_view == "Anomalies Only":
    filtered_df = filtered_df[filtered_df['Anomaly'] == -1]

# Display Metrics
st.subheader("📊 Customer Cluster Distribution")
st.bar_chart(filtered_df['Cluster'].value_counts())

# Show Anomaly Count
col1, col2 = st.columns(2)
col1.metric("✅ Normal Customers", len(df[df['Anomaly'] == 1]))
col2.metric("🚨 Anomalies Detected", len(df[df['Anomaly'] == -1]))

# Cluster Means Table
st.subheader("📈 Cluster-wise Averages")
st.dataframe(df.groupby('Cluster').mean(numeric_only=True).round(2))

# Optional: Show filtered data
st.subheader("🔎 Filtered Customer Data (Top 50 Rows)")
st.dataframe(filtered_df.head(50))

# Download Button
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_customers.csv',
    mime='text/csv'
)  

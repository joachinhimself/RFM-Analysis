import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit app title
st.title("RFM Analysis and Visualization")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Data Cleaning
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%d/%m/%y')
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], errors='coerce')

    # Calculate Age
    df['Age'] = (df['TransactionDate'].dt.year - df['CustomerDOB'].dt.year).fillna(0).astype(int)

    # Data Processing for RFM
    day = df['TransactionDate'].max()
    recency = df.groupby(['CustomerID']).agg({'TransactionDate': lambda x: (day - x.max()).days + 1}).reset_index().rename(columns={'TransactionDate': 'Recency'})
    frequency = df.groupby(['CustomerID']).size().reset_index(name='Frequency')
    monetary = df.groupby(['CustomerID']).agg({'TransactionAmount (INR)': 'sum'}).reset_index().rename(columns={'TransactionAmount (INR)': 'Monetary'})

    # Create RFM Table
    rfm_table = recency.merge(frequency, on='CustomerID').merge(monetary, on='CustomerID')

    # Assign RFM Scores
    quantile = rfm_table[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.5, 0.75]).to_dict()

    def assign_R_score(x):
        if x <= quantile['Recency'][0.25]:
            return 4
        elif x <= quantile['Recency'][0.5]:
            return 3
        elif x <= quantile['Recency'][0.75]:
            return 2
        else:
            return 1

    def assign_M_score(x):
        if x <= quantile['Monetary'][0.25]:
            return 1
        elif x <= quantile['Monetary'][0.5]:
            return 2
        elif x <= quantile['Monetary'][0.75]:
            return 3
        else:
            return 4

    rfm_table['R_score'] = rfm_table['Recency'].apply(assign_R_score)
    rfm_table['F_score'] = rfm_table['Frequency'].apply(lambda x: 4 if x > 3 else x)
    rfm_table['M_score'] = rfm_table['Monetary'].apply(assign_M_score)
    rfm_table['Summed_RFM_Scores'] = rfm_table[['R_score', 'F_score', 'M_score']].sum(axis=1)

    # Segmentation
    def assign_segments(x):
        if x <= 5:
            return 'Low'
        elif x <= 9:
            return 'Medium'
        else:
            return 'High'

    rfm_table['Segments'] = rfm_table['Summed_RFM_Scores'].apply(assign_segments)

    # Visualization
    st.subheader("RFM Table")
    st.write(rfm_table)

    # Heatmap of Correlation
    st.subheader("Correlation Heatmap of RFM Metrics")
    plt.figure(figsize=(5, 3))
    sns.heatmap(rfm_table[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    # Distribution of Recency
    st.subheader("Distribution of Recency")
    plt.figure(figsize=(10, 6))
    sns.histplot(rfm_table['Recency'], bins=10, kde=True)
    plt.title('Distribution of Recency')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # Count of Customers by Segments
    st.subheader("Count of Customers by Segments")
    plt.figure(figsize=(10, 6))
    sns.countplot(x=rfm_table['Segments'], palette='viridis')
    plt.title('Count of Customers by Segments')
    plt.xlabel('Segments')
    plt.ylabel('Count')
    st.pyplot(plt)

    # K-Means Clustering
    st.subheader("K-Means Clustering")
    RFM_df = rfm_table[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(RFM_df)
    kmeans = KMeans(n_clusters=3, random_state=1)
    cluster_assignment = kmeans.fit_predict(scaled_data)
    rfm_table['Cluster'] = cluster_assignment

    # Scatter Plot of Clusters
    st.subheader("Cluster Visualization")
    plt.figure(figsize=(15, 7))
    sns.scatterplot(data=rfm_table, x='Recency', y='Monetary', hue='Cluster', palette='viridis')
    plt.title('Recency vs Monetary by Cluster')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    st.pyplot(plt)

    # Customer Segmentation based on Clusters
    st.subheader("Customer Segmentation by Clusters")
    cluster_summary = rfm_table.groupby('Cluster').agg({
        'Recency': ['mean', 'std'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std'],
        'Summed_RFM_Scores': ['mean', 'std'],
        'Segments': lambda x: x.value_counts().index[0]  # Most common segment in the cluster
    }).reset_index()
    cluster_summary.columns = ['Cluster', 'Recency Mean', 'Recency Std', 'Frequency Mean', 'Frequency Std',
                               'Monetary Mean', 'Monetary Std', 'Most Common Segment']

    st.write(cluster_summary)

    # Visualization of Cluster Summary
    st.subheader("Cluster Summary Visualization")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cluster_summary, x='Cluster', y='Monetary Mean', palette='viridis')
    plt.title('Average Monetary Value by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Monetary Value')
    st.pyplot(plt)
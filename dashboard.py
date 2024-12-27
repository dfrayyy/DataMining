# streamlit_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Streamlit App Title
st.title("Dashboard: K-Means Clustering & Logistic Regression")
st.write("Upload your CSV file to perform K-Means clustering and Logistic Regression.")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview:")
    st.write(data.head())

    # Select Columns for Analysis
    st.write("### Select Columns for Analysis")
    selected_columns = st.multiselect(
        "Select numeric columns to use for K-Means and Logistic Regression:",
        options=data.columns.tolist(),
        default=data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    )

    if selected_columns:
        # Preprocess Data
        st.write("### Data Preprocessing")
        data = data.dropna(subset=selected_columns)  # Drop rows with missing values in selected columns
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[selected_columns])

        st.write("Data has been normalized for clustering and regression.")

        # --- K-Means Clustering ---
        st.write("## K-Means Clustering")
        n_clusters = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)

        data['Cluster'] = clusters
        silhouette_avg = silhouette_score(data_scaled, clusters)
        st.write(f"### Silhouette Score: {silhouette_avg:.2f}")

        # Visualize Clusters
        st.write("### Cluster Visualization")
        if len(selected_columns) >= 2:
            x_axis = st.selectbox("Select X-axis for visualization:", selected_columns)
            y_axis = st.selectbox("Select Y-axis for visualization:", selected_columns)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=data[x_axis], y=data[y_axis], hue=data['Cluster'], palette='viridis', s=100, ax=ax)
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)
        else:
            st.write("Please select at least two columns for visualization.")

        # --- Logistic Regression ---
        st.write("## Logistic Regression")
        target_column = st.selectbox("Select target column for Logistic Regression:", data.columns)
        
        # Ensure the target column is suitable for classification
        if data[target_column].nunique() <= 1:
            st.error("The target column must have at least two unique classes for classification.")
        else:
            # Encode Target Column if Necessary
            if data[target_column].dtype == 'object':
                le = LabelEncoder()
                data[target_column] = le.fit_transform(data[target_column])

        if target_column:
            # Encode Target Column if Necessary
            if data[target_column].dtype == 'object':
                le = LabelEncoder()
                data[target_column] = le.fit_transform(data[target_column])

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(data[selected_columns], data[target_column], test_size=0.3, random_state=42)

            # Train Logistic Regression
            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)
            y_pred = logreg.predict(X_test)

            # Evaluate Logistic Regression
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### Logistic Regression Accuracy: {accuracy:.2f}")

            # Display Confusion Matrix
            sfig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', ax=ax)
            ax.set_title("Confusion Matrix")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

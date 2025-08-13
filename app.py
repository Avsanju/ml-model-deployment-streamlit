import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

# -----------------------------
# App Title and Description
# -----------------------------
st.set_page_config(page_title="Cricket Match Data Analysis", layout="wide")
st.title("Cricket Match Data Analysis")
st.write("Explore cricket match data, visualize statistics, and make predictions using a trained ML model.")

# -----------------------------
# Sidebar Navigation
# -----------------------------
section = st.sidebar.radio("Select Section", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("data/cleaned_dataset.csv")
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure 'data/cleaned_dataset.csv' exists.")
    st.stop()

# -----------------------------
# Load Trained Model
# -----------------------------
try:
    model = joblib.load("models/best_model.joblib")
except FileNotFoundError:
    st.warning("Trained model not found. Predictions won't work until 'models/best_model.joblib' exists.")
    model = None

# -----------------------------
# Data Exploration Section
# -----------------------------
if section == "Data Exploration":
    st.header("Dataset Overview")
    st.subheader("Shape & Columns")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Filter Data")
    col_to_filter = st.selectbox("Select Column to Filter", df.columns)
    unique_vals = df[col_to_filter].dropna().unique()
    selected_vals = st.multiselect("Select values", unique_vals, default=unique_vals[:5])
    st.write(df[df[col_to_filter].isin(selected_vals)])

# -----------------------------
# Visualizations Section
# -----------------------------
elif section == "Visualizations":
    st.header("Data Visualizations")

    st.subheader("Bar Plot")
    col1 = st.selectbox("Select Column for Bar Plot", df.select_dtypes(include='object').columns)
    bar_data = df[col1].value_counts().reset_index()
    bar_data.columns = [col1, "count"]
    fig1 = px.bar(bar_data, x=col1, y="count", title=f"Bar Plot of {col1}")
    st.plotly_chart(fig1)

    st.subheader("Histogram")
    num_col = st.selectbox("Select Numeric Column for Histogram", df.select_dtypes(include=np.number).columns)
    fig2 = px.histogram(df, x=num_col, nbins=20, title=f"Histogram of {num_col}")
    st.plotly_chart(fig2)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig3, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

# -----------------------------
# Model Prediction Section
# -----------------------------
elif section == "Model Prediction":
    st.header("Predict Match Outcome")

    if model is None:
        st.warning("Model not loaded. Cannot make predictions.")
    else:
        # Replace these with your dataset's feature names
        feature_cols = [col for col in df.columns if col != "target"]
        input_data = {}
        for col in feature_cols:
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(f"Select {col}", df[col].unique())
            else:
                input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.success(f"Predicted outcome: {prediction[0]}")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                st.write("Prediction probabilities:")
                st.write(proba)

# -----------------------------
# Model Performance Section
# -----------------------------
elif section == "Model Performance":
    st.header("Model Performance")

    if model is None:
        st.warning("Model not loaded. Cannot show performance.")
    else:
        st.subheader("Model Evaluation Metrics")
        st.write("Add your evaluation metrics here (accuracy, F1-score, etc.).")

        st.subheader("Confusion Matrix / Performance Charts")
        st.write("Add your confusion matrix or other charts here.")

        st.subheader("Model Comparison")
        st.write("If multiple models are evaluated, show comparison table or chart here.")

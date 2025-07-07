import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Page configuration
st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")
st.title("🔬 Breast Cancer Prediction App")

st.markdown("""
This app uses a logistic regression model to predict whether a tumor is **Malignant (1)** or **Benign (0)** 
based on diagnostic features from breast cancer data.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

df = load_data()

# Prepare model
@st.cache_resource
def train_model(df):
    X = df[df.columns[2:]]
    y = df['diagnosis']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_imputed, y)

    return model, imputer, X.columns.tolist()

model, imputer, feature_names = train_model(df)

# Sidebar input
st.sidebar.header("Enter Patient Data")

def user_input():
    input_dict = {}
    for feature in feature_names:
        val = st.sidebar.number_input(f"{feature}", value=float(df[feature].mean()))
        input_dict[feature] = val
    return pd.DataFrame([input_dict])

user_df = user_input()

# Prediction
if st.sidebar.button("Predict"):
    input_imputed = imputer.transform(user_df)
    prediction = model.predict(input_imputed)[0]
    prediction_proba = model.predict_proba(input_imputed)[0][prediction]

    if prediction == 1:
        st.error(f"🔴 **Malignant Tumor Detected** (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"🟢 **Benign Tumor Detected** (Confidence: {prediction_proba:.2f})")

# Tabs
tab1, tab2 = st.tabs(["🔍 Exploratory Data Analysis", "📊 Data Insights"])

with tab1:
    st.header("Diagnosis Distribution")
    fig, ax = plt.subplots()
    df['diagnosis'].value_counts().plot.pie(labels=['Benign', 'Malignant'], autopct='%1.1f%%', startangle=90, ax=ax)
    st.pyplot(fig)

    st.header("Box Plot: Area Mean by Diagnosis")
    fig, ax = plt.subplots()
    sns.boxplot(x='diagnosis', y='area_mean', data=df, ax=ax)
    st.pyplot(fig)

    st.header("Histogram: Smoothness Mean by Diagnosis")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='smoothness_mean', hue='diagnosis', ax=ax, bins=30)
    st.pyplot(fig)

    st.header("Pairplot: Selected Features")
    selected = df[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean']]
    fig = sns.pairplot(selected, hue='diagnosis', diag_kind='hist')
    st.pyplot(fig)

with tab2:
    st.header("Data Statistics")
    st.write(df.describe())

    st.header("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.header("Violin Plot: Texture Mean by Diagnosis")
    fig, ax = plt.subplots()
    sns.violinplot(x='diagnosis', y='texture_mean', data=df, ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This app is for educational use only and does not provide medical advice. 
Always consult a medical professional for health concerns.
""")

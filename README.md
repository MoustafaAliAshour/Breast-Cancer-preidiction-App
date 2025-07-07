# 🔬 Breast Cancer Diagnosis Prediction App

This Streamlit web application predicts whether a breast tumor is **Malignant (cancerous)** or **Benign (non-cancerous)** using a logistic regression model trained on diagnostic features from a public breast cancer dataset.

---

## 📊 Features

- ✅ Predict cancer diagnosis based on medical features
- ✅ Interactive sidebar to input new patient data
- ✅ Exploratory Data Analysis (EDA) with visualizations
- ✅ Correlation heatmaps, boxplots, histograms, and more
- ✅ Logistic Regression model from scikit-learn

---

## 🚀 Getting Started

### 1. Clone or Download the Project

git clone https://github.com/MoustafaAliAshour/breast-cancer-app.git
cd breast-cancer-app
Or just download the ZIP and extract it.

### 2. Install Dependencies
Make sure Python 3.7+ is installed. Then run:

pip install -r requirements.txt
Or manually:

pip install streamlit pandas numpy scikit-learn matplotlib seaborn

### 3. Add Dataset
Make sure the data.csv file is in the same folder. You can download it from:

🔗 Kaggle - Breast Cancer Wisconsin Dataset

### 4. Run the App
Launch the Streamlit app with:

streamlit run streamlit_app.py
Then open your browser to:

### http://localhost:8501
## 📁 File Structure

📦 breast-cancer-app/
├── data.csv                # Dataset file
├── streamlit_app.py        # Streamlit app source code
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
📈 Model Details
Algorithm: Logistic Regression

Target Variable: diagnosis (1 = Malignant, 0 = Benign)

Preprocessing: Missing values handled using SimpleImputer

Input Features: 30 numeric diagnostic features (e.g., radius, texture, perimeter)

## 🖼 Sample Visualizations
Diagnosis distribution pie chart

Area mean boxplot by diagnosis

Smoothness histogram by class

Pairplots and correlation heatmap

## ⚠️ Disclaimer
This tool is for educational purposes only. It is not a medical device and should not be used to diagnose or treat any medical condition. Always consult a licensed physician for medical advice.

## 🙌 Credits
Data: UCI Machine Learning Repository

Created using: Python, Streamlit, scikit-learn, Matplotlib, Seaborn

## 📬 Contact
For questions or contributions, open an issue or contact: es-moustafa.aly2027@alexu.edu.eg

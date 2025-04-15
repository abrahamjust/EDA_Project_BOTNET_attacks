# EDA_Project_BOTNET_attacks

# 📊 IoT Botnet Attack Detection - EDA & ML Modeling

This project performs **Exploratory Data Analysis (EDA)**, **Time Series Analysis**, and **Classification Modeling** on IoT network traffic data to detect **Mirai** and **Gafgyt (BASHLITE)** botnet attacks. It utilizes the **N-BaIoT** dataset and implements models like **Random Forest**, **XGBoost**, **LightGBM**, **AdaBoost**, and **Stacking**, along with **SHAP** and **LIME** for model explainability.

---

## 📁 Dataset

We used the [N-BaIoT Dataset](https://www.unb.ca/cic/datasets/nbaiot.html), which includes:

- IoT traffic under benign and botnet-infected conditions
- Collected from 9 different IoT devices
- Stream-based statistical features like mean, std, magnitude, weight, etc.
- 115 independent features + 1 class label
- Attack types: **Mirai (UDP/ACK/Scan)** and **Gafgyt (TCP/UDP/HTTP)**

> ⚠️ The dataset is large (multiple GBs). Due to GitHub file size restrictions, CSV files are not included in this repo. You can download them from the link below:

🔗 **[Download the Dataset CSVs from Google Drive](https://drive.google.com/drive/folders/1n-MlmC3Xuq8EqYzU6da6n0lFAW5FVAls?usp=sharing)**

---

## 📌 Key Steps in the Project

### 1. 📈 Exploratory Data Analysis (EDA)
- Distribution of benign vs. attack traffic
- Feature correlation and outlier detection
- Temporal patterns and density plots
- Dimensionality reduction (PCA)

### 2. 🧠 ML Model Training
- Train/test split and preprocessing
- Models: Random Forest, XGBoost, LightGBM, AdaBoost, Stacking
- Hyperparameter tuning
- Metrics: Accuracy, Precision, Recall, F1 Score

### 3. 📉 Time Series & Anomaly Detection
- ARIMA and LSTM for detecting attack trends over time
- Index as a proxy for time-based features

### 4. 🕵️ Model Explainability
- **SHAP**: Feature importance via Shapley values
- **LIME**: Local Interpretable Model-Agnostic Explanations

---

## 📦 Project Structure

```
EDA_PROJECT/
├── data/                      # (Not included in repo, available via Drive)
├── models/                    # .pkl model files (Git LFS used)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_TimeSeries_Analysis.ipynb
│   └── 04_Model_Explainability.ipynb
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/abrahamjust/EDA_Project_BOTNET_attacks.git
   cd EDA_Project_BOTNET_attacks

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Download CSVs from Google Drive and place in data/ folder.

4. Run the notebooks in order to explore, model, and explain.

## 📚 Requirements
Python 3.8+

pandas, numpy, matplotlib, seaborn

scikit-learn, xgboost, lightgbm, shap, lime

statsmodels, keras/tensorflow (for LSTM)

## 🧠 Model Files
All trained model .pkl files are tracked using Git Large File Storage (LFS). Make sure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

## 📬 Contact
For questions or collaborations, reach out to Abraham Justin
abrahamjust@gmail.com
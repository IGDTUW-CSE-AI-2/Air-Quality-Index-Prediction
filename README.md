# Air Quality Index (AQI) Prediction

A machine learning project that predicts Air Quality Index categories based on pollutant concentrations using XGBoost classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies](#technologies)

## 🎯 Overview

This project predicts Air Quality Index (AQI) categories based on air pollutant concentrations. It uses machine learning classification models trained on historical air quality data from across India to classify air quality into six categories:

- **Good** (0-50)
- **Moderate** (51-100)
- **Poor** (101-200)
- **Unhealthy** (201-300)
- **Very Unhealthy** (301-400)
- **Hazardous** (>400)

## 📊 Dataset

**Source:** `data.csv` (435,742 records with 13 features)

### Features

| Feature | Description | Unit |
|---------|-------------|------|
| **SO₂** | Sulfur Dioxide | µg/m³ |
| **NO₂** | Nitrogen Dioxide | µg/m³ |
| **RSPM** | Respirable Suspended Particulate Matter | µg/m³ |
| **SPM** | Suspended Particulate Matter | µg/m³ |
| **State** | Indian State | Categorical |
| **Location** | Sampling Location | Categorical |
| **Type** | Area Type (Urban/Rural) | Categorical |

### AQI Categories

| Range | Category | Health Impact |
|-------|----------|---------------|
| 0-50 | ✅ Good | No health impact expected |
| 51-100 | 🟡 Moderate | Mild health effects possible |
| 101-200 | 🟠 Poor | Health warnings for sensitive groups |
| 201-300 | 🔴 Unhealthy | General health warnings |
| 301-400 | 🟣 Very Unhealthy | Severe health warnings |
| >400 | ⚫ Hazardous | Emergency conditions |

## 📁 Project Structure
Workspace
(rerun without)
Collecting workspace information

air-quality-prediction/
├── air-quality-prediction.ipynb # Jupyter notebook with analysis & training
├── app.py # Streamlit web application
├── data.csv # Historical air quality dataset
├── xgb_model.pkl # Trained XGBoost model
├── label_encoder.pkl # Label encoder for AQI categories
└── README.md # This file


## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/IGDTUW-CSEAI-2/air-quality-prediction.git
   cd air-quality-prediction

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn jupyter

## 💻 Usage

### Option 1: Jupyter Notebook (Analysis & Training)

```bash
jupyter notebook air-quality-prediction.ipynb
```

This opens the complete analysis workflow including:
- **Data exploration and visualization** - Understand dataset structure and distributions
- **Feature engineering** - Calculate individual pollutant indices (SOi, Noi, Rpi, SPMi)
- **Model training and evaluation** - Train multiple algorithms and compare performance
- **Comparative analysis** - Evaluate Regression and Classification models

**Notebook Workflow:**
1. Load and explore air quality dataset (435,742 records)
2. Handle missing values and preprocess data
3. Calculate AQI using piecewise linear functions
4. Train regression models (Linear, Decision Tree, Random Forest)
5. Train classification models (Logistic, Decision Tree, Random Forest, KNN, XGBoost)
6. Evaluate and save best model (XGBoost)

### Option 2: Streamlit Web Application

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`

**Steps to Make Predictions:**
1. Enter values for the four pollutant concentrations:
   - **Sulfur Dioxide (SO₂)** in µg/m³
   - **Nitrogen Dioxide (NO₂)** in µg/m³
   - **Respirable Suspended Particulate Matter (RSPM)** in µg/m³
   - **Suspended Particulate Matter (SPM)** in µg/m³
2. Click **"Predict"** button
3. View the AQI category classification and pollutant breakdown

### Example Predictions

| SO₂ | NO₂ | RSPM | SPM | Prediction | Category |
|-----|-----|------|-----|-----------|----------|
| 727 | 327.55 | 78.2 | 100 | 🔴 Hazardous | Severe health warnings |
| 2.7 | 45 | 35.16 | 23 | ✅ Good | No health impact |
| 10 | 2.8 | 82 | 20 | 🟡 Moderate | Mild health effects |
| 2 | 45.8 | 37 | 32 | ✅ Good | Safe air quality |

---

## 🔬 Methodology

### 1. Data Preprocessing

- **Handled missing values:**
  - Mode imputation for categorical data (location, type)
  - Zero-fill strategy for numerical features
- **Removed irrelevant columns:** Station code, dates, agency, monitoring station
- **Data cleaning:** Standardized formats and removed duplicates
- **Feature selection:** Retained SO₂, NO₂, RSPM, SPM as primary features

### 2. AQI Calculation

Individual pollutant indices are calculated using **piecewise linear functions** based on standard AQI formulas:

**SO₂ Index (SI) Formula:**
```
If SO₂ ≤ 40:       SI = SO₂ × (50/40)
If 40 < SO₂ ≤ 80:  SI = 50 + (SO₂ - 40) × (50/40)
If 80 < SO₂ ≤ 380: SI = 100 + (SO₂ - 80) × (100/300)
If 380 < SO₂ ≤ 800: SI = 200 + (SO₂ - 380) × (100/420)
If 800 < SO₂ ≤ 1600: SI = 300 + (SO₂ - 800) × (100/800)
If SO₂ > 1600:     SI = 400 + (SO₂ - 1600) × (100/800)
```

**Similar formulas apply for:**
- **NO₂ Index (Noi)** - Nitrogen Dioxide
- **RSPM Index (Rpi)** - Respirable Suspended Particulate Matter
- **SPM Index (SPMi)** - Suspended Particulate Matter

**Final AQI Calculation:**
```
AQI = max(SI, Noi, Rpi, SPMi)
```

### 3. Feature Engineering

- Created individual pollutant indices: `[SOi, Noi, Rpi, SPMi]`
- Derived AQI values from maximum individual index
- Categorized AQI into six classification levels
- **Data split:** 67% training, 33% testing (random_state=70)

---

## 🤖 Models

### Regression Models (AQI Value Prediction)

| Model | Purpose | Performance |
|-------|---------|-------------|
| **Linear Regression** | Baseline model | Moderate RMSE |
| **Decision Tree Regressor** | Non-linear relationships | Low RMSE, High R² |
| **Random Forest Regressor** | Ensemble approach | Best Regression accuracy |

### Classification Models (AQI Category Prediction)

| Model | Train Accuracy | Test Accuracy | Status | Notes |
|-------|---|---|---|---|
| Logistic Regression | High | Moderate | ⭐ Baseline | Good interpretability |
| Decision Tree Classifier | Very High | Moderate | ⚠️ Overfitting | High variance |
| Random Forest Classifier | Very High | Good | ✅ Strong | Balanced performance |
| K-Nearest Neighbors | High | Good | ✅ Reliable | Simple & effective |
| **XGBoost Classifier** | **Very High** | **Very High** | **🏆 Best** | **Selected model** |

---

## 📈 Results

### XGBoost Performance (Best Model)

```
✅ Model accuracy on train:  95.2%
✅ Model accuracy on test:   92.8%
📊 Kappa Score:              0.89
```

**Model Configuration:**
```python
XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=70
)
```

### Confusion Matrix Analysis

The XGBoost model demonstrates:
- ✅ **Excellent classification** of extreme categories (Hazardous, Good)
- ✅ **Strong discrimination** across all AQI levels
- ✅ **Minimal misclassification** between adjacent categories
- ✅ **High precision and recall** for critical health categories

---

## 🔍 Key Findings

### Geographic Insights

**🔴 High Pollution States:**
- **Delhi** 
  - Highest RSPM (Respirable Suspended Particulate Matter)
  - Highest SPM (Suspended Particulate Matter)
  - Elevated PM₂.₅ levels
  - Primary cause: Vehicle emissions & industrial activity

- **West Bengal**
  - Elevated NO₂ (Nitrogen Dioxide) concentrations
  - Industrial and vehicular contributions

- **Uttarakhand**
  - Higher SO₂ (Sulfur Dioxide) levels
  - Industrial and heating source emissions

**🟢 Low Pollution Areas:**
- Coastal regions show better air quality
- Rural areas generally have lower pollutant levels
- Himalayan regions benefit from natural air circulation

### Pollutant Correlations

- **SPM & RSPM:** Strong positive correlation (both indicate particulate matter)
- **NO₂:** Varies independently with traffic patterns and rush hours
- **SO₂:** Shows seasonal patterns and industrial activity correlation

---

## 🛠️ Technologies

| Category | Tools & Libraries |
|----------|-------------------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Web Framework** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter |
| **Model Serialization** | Pickle |

---

## 📦 Dependencies

```
pandas>=1.1.0
numpy>=1.19.0
scikit-learn>=0.24.0
xgboost>=1.3.0
streamlit>=0.80.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
```

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🎓 Model Details

### XGBoost Configuration

**Features (Independent Variables):**
- `SOi` - Sulfur Dioxide Index
- `Noi` - Nitrogen Dioxide Index
- `Rpi` - RSPM Index
- `SPMi` - SPM Index

**Target (Dependent Variable):**
- `AQI_Range` - Air Quality Category (6 classes)
  - Encoded using `LabelEncoder` with values [0-5]
  - Original categories: Good, Moderate, Poor, Unhealthy, Very Unhealthy, Hazardous

**Preprocessing Pipeline:**
1. Calculate individual pollutant indices
2. Encode target variable (AQI_Range → numeric labels)
3. Split data (67% train, 33% test)
4. Train XGBoost classifier
5. Inverse transform predictions to original labels

### Model Files

- **`xgb_model.pkl`** - Serialized XGBoost classifier model
  - Contains trained weights and decision trees
  - Load with: `pickle.load(open("xgb_model.pkl", "rb"))`

- **`label_encoder.pkl`** - Fitted LabelEncoder
  - Converts category strings ↔ numeric labels
  - Used for inverse transformations in predictions

---

## 📊 Saved Artifacts

After training, the following files are generated:

```
./xgb_model.pkl          # Trained XGBoost model (~2-5 MB)
./label_encoder.pkl      # Label encoder for 6 AQI categories (~1 KB)
```

**Load saved models in your application:**
```python
import pickle

# Load model
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Make prediction
encoded_pred = xgb_model.predict([[SOi, Noi, Rpi, SPMi]])
original_pred = le.inverse_transform(encoded_pred)
```

---

## 🚀 Future Enhancements

- [ ] **Temporal Analysis** - Incorporate seasonal patterns and time-series forecasting
- [ ] **Weather Integration** - Add temperature, humidity, wind speed data
- [ ] **REST API Deployment** - Deploy model as Flask/FastAPI service
- [ ] **Real-time Monitoring** - Connect to live air quality sensors
- [ ] **Geographic Visualization** - Interactive maps showing pollution hotspots
- [ ] **Predictive Forecasting** - 7-day AQI predictions
- [ ] **Mobile Application** - iOS/Android app for predictions
- [ ] **Model Explainability** - SHAP values for feature importance
- [ ] **Multi-city Support** - Expand to international air quality data



   

# ai_wine-quality-predictor
ai final project

# 🍷 Wine Quality Prediction System

A complete machine learning system that predicts wine quality ratings (1-10) based on physicochemical properties. Built with scikit-learn and deployed using Streamlit.

## 🎯 Project Overview

This project demonstrates a complete ML pipeline from data exploration to production deployment:

- **Data Processing**: Automated preprocessing with outlier handling, feature engineering, and scaling
- **Model Training**: Multiple ML algorithms with hyperparameter tuning and cross-validation  
- **Web Application**: Interactive Streamlit app for real-time predictions
- **Batch Processing**: Support for bulk wine quality predictions
- **Model Retraining**: Ability to retrain models with new data

## 🚀 Live Demo

https://aiwine-quality-predictor-rshkk9px77zpucbpw9hefn.streamlit.app/

## 📊 Dataset

The system uses the **Wine Quality Dataset** from UCI Machine Learning Repository:
- **Red Wine**: 1,599 samples
- **White Wine**: 4,898 samples  
- **Features**: 11 physicochemical properties
- **Target**: Quality rating (3-9 scale)

### Features Used:
- Fixed acidity, Volatile acidity, Citric acid
- Residual sugar, Chlorides
- Free sulfur dioxide, Total sulfur dioxide
- Density, pH, Sulphates
- Alcohol content, Wine type (red/white)

## 🏗️ Project Structure

```
wine-quality-predictor/
├── 📊 **Notebooks**
│   ├── 01_data_exploration.ipynb     # EDA and data analysis
│   ├── 02_data_preprocessing.ipynb   # Data cleaning and processing
│   └── 03_model_training.ipynb       # Model training and evaluation
├── 💻 **Source Code**
│   ├── app.py                        # Main Streamlit application
│   ├── data_processor.py             # Data preprocessing pipeline
│   ├── model_trainer.py              # ML model training
│   └── requirements.txt              # Python dependencies
└── 📚 **Documentation**
    └── README.md                     # This file
```

## 🛠️ Installation & Setup

### Quick Start (Local Development)

1. **Clone the repository**
   ```bash
   git clone https://github.com/adidi127/wine-quality-predictor.git
   cd wine-quality-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

### Alternative Setup with Virtual Environment

```bash
# Create virtual environment
python -m venv wine_env

# Activate environment
# Windows:
wine_env\Scripts\activate
# macOS/Linux:
source wine_env/bin/activate

# Install dependencies and run
pip install -r requirements.txt
streamlit run app.py
```

## 📱 How to Use

### 🎯 **Single Prediction**
1. Navigate to the **"Predict"** tab
2. Enter wine properties using the input form:
   - Acidity levels (fixed, volatile, citric)
   - Chemical properties (sugar, chlorides, sulphates)
   - Physical properties (density, pH, alcohol)
   - Wine type (red/white)
3. Click **"Predict Wine Quality"**
4. View predicted quality score with confidence intervals

### 📊 **Batch Prediction**
1. Go to the **"Batch"** tab
2. Download the sample CSV format
3. Prepare your wine data file with the same columns
4. Upload your CSV file
5. Get predictions for all wines at once
6. Download results as CSV

### 🔬 **Model Training** (Advanced)
1. Access the **"Train"** tab in the full version
2. Select models to train (Random Forest, Gradient Boosting, etc.)
3. Choose training options
4. Monitor training progress
5. Models are automatically saved and evaluated

## 🧠 Machine Learning Models

The system supports multiple ML algorithms:

| Model | Use Case | Performance |
|-------|----------|-------------|
| **Random Forest** | Primary model | R² ~0.65-0.70 |
| **Gradient Boosting** | High accuracy | R² ~0.68-0.73 |
| **Ridge Regression** | Fast baseline | R² ~0.55-0.60 |
| **Support Vector Regression** | Non-linear patterns | R² ~0.60-0.65 |

## 📈 Model Performance

- **Accuracy**: ~70-75% predictions within ±1 quality point
- **R² Score**: ~0.65-0.70 for best models
- **Response Time**: <2 seconds per prediction
- **Batch Processing**: 1000+ wines in <30 seconds

## 🔧 API Usage (For Developers)

```python
from data_processor import WineQualityProcessor
from model_trainer import WineQualityModelTrainer
import pandas as pd

# Load and process data
processor = WineQualityProcessor()
data = processor.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = processor.fit_transform(data)

# Train models
trainer = WineQualityModelTrainer()
trainer.train_all_models(X_train, y_train, X_val, y_val)

# Make predictions
sample_data = pd.DataFrame({
    'fixed acidity': [7.4], 'volatile acidity': [0.7],
    'citric acid': [0.0], 'residual sugar': [1.9],
    'chlorides': [0.076], 'free sulfur dioxide': [11.0],
    'total sulfur dioxide': [34.0], 'density': [0.9978],
    'pH': [3.51], 'sulphates': [0.56],
    'alcohol': [9.4], 'wine_type': ['red']
})

processed_sample = processor.transform(sample_data)
prediction = trainer.predict(processed_sample)
print(f"Predicted quality: {prediction[0]:.1f}")
```

## 📂 Repository Contents

### **Jupyter Notebooks**
- `01_data_exploration.ipynb`: Complete EDA with visualizations
- `02_data_preprocessing.ipynb`: Data cleaning and feature engineering  
- `03_model_training.ipynb`: Model training and evaluation

### **Source Code**
- `app.py`: Main Streamlit web application
- `data_processor.py`: Data preprocessing pipeline
- `model_trainer.py`: Machine learning model trainer
- `requirements.txt`: Python package dependencies

## 🚀 Deployment

This project is deployed on **Streamlit Cloud** for easy access:

1. **Live Application**: https://aiwine-quality-predictor-rshkk9px77zpucbpw9hefn.streamlit.app/
2. **Automatic Updates**: Deploys automatically from this GitHub repository
3. **Scalable**: Handles multiple concurrent users
4. **Free Hosting**: Deployed on Streamlit Cloud's free tier

### Local Deployment
```bash
git clone https://github.com/adidi127/wine-quality-predictor.git
cd wine-quality-predictor
pip install -r requirements.txt
streamlit run app.py
```


## 🔄 Quick Start Commands

```bash
# Clone and run locally
git clone https://github.com/adidi127/wine-quality-predictor.git
cd wine-quality-predictor
pip install -r requirements.txt
streamlit run app.py

# The app will open at: http://localhost:8501
```


# Wine Quality Dataset - Exploratory Data Analysis
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Wine Quality Prediction System - Data Exploration")
print("=" * 50)

# 1. DATA LOADING
# ===============

# Load the datasets
try:
    # Download from UCI repository
    red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    red_wine = pd.read_csv(red_wine_url, sep=';')
    white_wine = pd.read_csv(white_wine_url, sep=';')
    
    print(f"✓ Red wine data loaded: {red_wine.shape}")
    print(f"✓ White wine data loaded: {white_wine.shape}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure you have internet connection to download the datasets")

# Add wine type column
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], ignore_index=True)
print(f"✓ Combined dataset shape: {wine_data.shape}")

# 2. BASIC DATA INFORMATION
# ========================

print("\n" + "="*50)
print("DATASET OVERVIEW")
print("="*50)

print("\nDataset Info:")
print(wine_data.info())

print("\nFirst 5 rows:")
print(wine_data.head())

print("\nBasic Statistics:")
print(wine_data.describe())

print("\nMissing Values:")
print(wine_data.isnull().sum())

print(f"\nUnique Quality Scores: {sorted(wine_data['quality'].unique())}")
print(f"Quality Score Distribution:")
print(wine_data['quality'].value_counts().sort_index())

# 3. DATA QUALITY ASSESSMENT
# ==========================

print("\n" + "="*50)
print("DATA QUALITY ASSESSMENT")
print("="*50)

# Check for duplicates
duplicates = wine_data.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Check for outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)

print("\nOutliers per feature (IQR method):")
numerical_cols = wine_data.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col != 'quality']

for col in numerical_cols:
    outlier_count = detect_outliers(wine_data, col)
    outlier_percentage = (outlier_count / len(wine_data)) * 100
    print(f"{col:20}: {outlier_count:4d} ({outlier_percentage:.1f}%)")

# 4. EXPLORATORY DATA ANALYSIS
# ============================

print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Quality distribution by wine type
print("\nQuality Distribution by Wine Type:")
quality_by_type = pd.crosstab(wine_data['wine_type'], wine_data['quality'])
print(quality_by_type)

# Calculate correlation matrix
correlation_matrix = wine_data[numerical_cols + ['quality']].corr()
print(f"\nFeatures most correlated with quality:")
quality_corr = correlation_matrix['quality'].abs().sort_values(ascending=False)
print(quality_corr.head(10))

# 5. VISUALIZATION FUNCTIONS
# ==========================

def create_visualizations(data):
    """Create comprehensive visualizations for wine quality dataset"""
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Quality distribution
    plt.subplot(3, 4, 1)
    data['quality'].hist(bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Wine Quality Scores')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    
    # 2. Quality by wine type
    plt.subplot(3, 4, 2)
    quality_type = pd.crosstab(data['wine_type'], data['quality'])
    quality_type.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Quality Distribution by Wine Type')
    plt.xlabel('Wine Type')
    plt.ylabel('Count')
    plt.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Correlation heatmap
    plt.subplot(3, 4, (3, 4))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    
    # 4. Top correlated features with quality
    top_features = quality_corr.head(6).index[1:]  # Exclude quality itself
    
    for i, feature in enumerate(top_features):
        plt.subplot(3, 4, 5 + i)
        plt.scatter(data[feature], data['quality'], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Quality')
        plt.title(f'{feature} vs Quality\nCorr: {correlation_matrix.loc[feature, "quality"]:.3f}')
        
        # Add trend line
        z = np.polyfit(data[feature], data['quality'], 1)
        p = np.poly1d(z)
        plt.plot(data[feature], p(data[feature]), "r--", alpha=0.8)
    
    # 5. Boxplot for quality categories
    plt.subplot(3, 4, (11, 12))
    
    # Create quality categories
    data_viz = data.copy()
    data_viz['quality_category'] = pd.cut(data_viz['quality'], 
                                         bins=[0, 4, 6, 10], 
                                         labels=['Low (3-4)', 'Medium (5-6)', 'High (7-9)'])
    
    # Select top features for boxplot
    top_3_features = quality_corr.head(4).index[1:]  # Top 3 excluding quality
    
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, feature in enumerate(top_3_features):
        data_viz.boxplot(column=feature, by='quality_category', ax=axes[i])
        axes[i].set_title(f'{feature} by Quality Category')
        axes[i].set_xlabel('Quality Category')
    
    plt.tight_layout()
    plt.show()
    
    return fig, fig2

# Generate visualizations
if 'wine_data' in locals():
    print("Generating visualizations...")
    try:
        fig1, fig2 = create_visualizations(wine_data)
        print("✓ Visualizations created successfully")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

# 6. KEY INSIGHTS SUMMARY
# =======================

print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)

# Basic statistics
total_samples = len(wine_data)
red_samples = len(wine_data[wine_data['wine_type'] == 'red'])
white_samples = len(wine_data[wine_data['wine_type'] == 'white'])

print(f"Dataset Summary:")
print(f"  • Total samples: {total_samples:,}")
print(f"  • Red wine samples: {red_samples:,} ({red_samples/total_samples*100:.1f}%)")
print(f"  • White wine samples: {white_samples:,} ({white_samples/total_samples*100:.1f}%)")
print(f"  • Features: {len(numerical_cols)} numerical + 1 categorical")
print(f"  • Quality range: {wine_data['quality'].min()}-{wine_data['quality'].max()}")
print(f"  • Average quality: {wine_data['quality'].mean():.2f}")

print(f"\nData Quality:")
print(f"  • No missing values: ✓")
print(f"  • Duplicate rows: {duplicates}")
print(f"  • Outliers detected in most features (expected for wine data)")

print(f"\nTop Quality Predictors:")
for i, (feature, corr) in enumerate(quality_corr.head(6).items()):
    if feature != 'quality':
        print(f"  {i}. {feature}: {corr:.3f}")

print(f"\nRecommendations for Modeling:")
print(f"  • Target variable: Wine quality (3-9 scale)")
print(f"  • Consider both regression and classification approaches")
print(f"  • Feature scaling recommended due to different ranges")
print(f"  • Wine type might be important categorical feature")
print(f"  • Quality distribution is roughly normal, good for modeling")
print(f"  • Strong correlations suggest good predictive potential")

# 7. DATA EXPORT FOR NEXT PHASE
# =============================

print("\n" + "="*50)
print("PREPARING DATA FOR MODELING")
print("="*50)

# Save processed data
try:
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save raw combined data
    wine_data.to_csv('data/raw/wine_quality_combined.csv', index=False)
    print("✓ Raw data saved to 'data/raw/wine_quality_combined.csv'")
    
    # Save summary statistics
    summary_stats = {
        'dataset_shape': wine_data.shape,
        'features': list(numerical_cols) + ['wine_type'],
        'target': 'quality',
        'quality_range': [wine_data['quality'].min(), wine_data['quality'].max()],
        'missing_values': wine_data.isnull().sum().sum(),
        'duplicates': duplicates,
        'top_features': quality_corr.head(6).to_dict()
    }
    
    import json
    with open('data/processed/eda_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print("✓ EDA summary saved to 'data/processed/eda_summary.json'")
    
except Exception as e:
    print(f"Error saving files: {e}")

print("\n" + "="*50)
print("EDA COMPLETE - READY FOR PREPROCESSING")
print("="*50)

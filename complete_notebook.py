# Wine Quality Model Training - Complete Pipeline
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import WineQualityProcessor
from model_trainer import WineQualityModelTrainer

print("Wine Quality Model Training Pipeline")
print("=" * 50)

# 1. LOAD AND PREPROCESS DATA
# ===========================

print("\n1. Loading and preprocessing data...")

# Initialize processor
processor = WineQualityProcessor(
    outlier_method='iqr',
    scaling_method='standard',
    random_state=42
)

# Load and process data
data = processor.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = processor.fit_transform(data)

print(f"✓ Data processed successfully!")
print(f"  - Training set: {X_train.shape}")
print(f"  - Validation set: {X_val.shape}")
print(f"  - Test set: {X_test.shape}")
print(f"  - Features: {len(processor.feature_names)}")

# Display feature information
print(f"\nFeatures used:")
for i, feature in enumerate(processor.feature_names):
    print(f"  {i+1:2d}. {feature}")

# 2. QUICK MODEL COMPARISON
# =========================

print("\n" + "="*50)
print("2. Quick Model Comparison (No Hyperparameter Tuning)")
print("="*50)

# Initialize trainer
trainer = WineQualityModelTrainer(random_state=42)

# Quick training without hyperparameter tuning
quick_models = ['random_forest', 'gradient_boosting', 'ridge', 'lasso']
print(f"Training models: {', '.join(quick_models)}")

start_time = time.time()

for model_name in quick_models:
    print(f"\nTraining {model_name}...")
    results = trainer.train_single_model(
        model_name, X_train, y_train, X_val, y_val, 
        tune_hyperparameters=False
    )
    
    metrics = results['metrics']
    print(f"  - Train R²: {metrics['train_r2']:.4f}")
    print(f"  - Val R²: {metrics['val_r2']:.4f}")
    print(f"  - Val RMSE: {metrics['val_rmse']:.4f}")

quick_time = time.time() - start_time
print(f"\nQuick training completed in {quick_time:.2f} seconds")

# Display quick comparison
quick_comparison = trainer.get_model_comparison()
print("\nQuick Comparison Results:")
print(quick_comparison[['Model', 'Train_R2', 'Val_R2', 'Val_RMSE', 'Training_Time']])

# 3. COMPREHENSIVE MODEL TRAINING WITH HYPERPARAMETER TUNING
# ==========================================================

print("\n" + "="*50)
print("3. Comprehensive Training with Hyperparameter Tuning")
print("="*50)

# Initialize new trainer for comprehensive training
comprehensive_trainer = WineQualityModelTrainer(random_state=42)

# All available models
all_models = ['random_forest', 'gradient_boosting', 'support_vector', 'ridge', 'lasso', 'elastic_net']
print(f"Training models with hyperparameter tuning: {', '.join(all_models)}")

# Train all models with hyperparameter tuning
start_time = time.time()
comprehensive_trainer.train_all_models(
    X_train, y_train, X_val, y_val,
    models_to_train=all_models,
    tune_hyperparameters=True
)

comprehensive_time = time.time() - start_time
print(f"\nComprehensive training completed in {comprehensive_time:.2f} seconds")

# 4. MODEL EVALUATION ON TEST SET
# ===============================

print("\n" + "="*50)
print("4. Test Set Evaluation")
print("="*50)

# Evaluate on test set
test_results = comprehensive_trainer.evaluate_on_test(X_test, y_test)

# Display final comparison
final_comparison = comprehensive_trainer.get_model_comparison()
print("\nFinal Model Comparison (with test results):")
print(final_comparison)

# 5. DETAILED ANALYSIS OF BEST MODEL
# ==================================

print("\n" + "="*50)
print("5. Best Model Analysis")
print("="*50)

best_model_name = comprehensive_trainer.best_model_name
best_model = comprehensive_trainer.best_model

print(f"Best Model: {best_model_name}")
print(f"Best Model Type: {type(best_model).__name__}")

# Get best model results
best_results = comprehensive_trainer.models[best_model_name]

print(f"\nBest Model Performance:")
print(f"  - Training R²: {best_results['metrics']['train_r2']:.4f}")
print(f"  - Validation R²: {best_results['metrics']['val_r2']:.4f}")
print(f"  - Test R²: {best_results['test_metrics']['test_r2']:.4f}")
print(f"  - Test RMSE: {best_results['test_metrics']['test_rmse']:.4f}")
print(f"  - Cross-val Mean: {best_results['cv_mean']:.4f} ± {best_results['cv_std']:.4f}")

# Best hyperparameters
if best_results['best_params']:
    print(f"\nBest Hyperparameters:")
    for param, value in best_results['best_params'].items():
        print(f"  - {param}: {value}")

# 6. FEATURE IMPORTANCE ANALYSIS
# ==============================

print("\n" + "="*50)
print("6. Feature Importance Analysis")
print("="*50)

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    feature_names = processor.feature_names
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row.name+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {best_model_name.replace("_", " ").title()}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

else:
    print("Feature importance not available for this model type")

# 7. PREDICTION ANALYSIS
# ======================

print("\n" + "="*50)
print("7. Prediction Analysis")
print("="*50)

# Make predictions on test set
y_test_pred = best_model.predict(X_test)

# Prediction statistics
print("Test Set Prediction Analysis:")
print(f"  - Actual quality range: {y_test.min():.1f} - {y_test.max():.1f}")
print(f"  - Predicted quality range: {y_test_pred.min():.1f} - {y_test_pred.max():.1f}")
print(f"  - Mean actual quality: {y_test.mean():.2f}")
print(f"  - Mean predicted quality: {y_test_pred.mean():.2f}")

# Prediction accuracy within tolerance
tolerances = [0.5, 1.0, 1.5]
for tolerance in tolerances:
    accurate_predictions = np.abs(y_test - y_test_pred) <= tolerance
    accuracy_pct = accurate_predictions.mean() * 100
    print(f"  - Predictions within ±{tolerance}: {accuracy_pct:.1f}%")

# Visualize predictions vs actual
plt.figure(figsize=(12, 5))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted vs Actual Wine Quality')
plt.grid(True, alpha=0.3)

# Add R² annotation
r2_test = best_results['test_metrics']['test_r2']
plt.text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Residuals plot
plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Quality')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. MODEL COMPARISON VISUALIZATION
# =================================

print("\n" + "="*50)
print("8. Model Performance Visualization")
print("="*50)

# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# R² comparison
ax1 = axes[0, 0]
models = final_comparison['Model']
train_r2 = final_comparison['Train_R2']
val_r2 = final_comparison['Val_R2']
test_r2 = final_comparison.get('Test_R2', [0]*len(models))

x = np.arange(len(models))
width = 0.25

ax1.bar(x - width, train_r2, width, label='Train', alpha=0.8)
ax1.bar(x, val_r2, width, label='Validation', alpha=0.8)
ax1.bar(x + width, test_r2, width, label='Test', alpha=0.8)

ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# RMSE comparison
ax2 = axes[0, 1]
train_rmse = final_comparison['Train_RMSE']
val_rmse = final_comparison['Val_RMSE']
test_rmse = final_comparison.get('Test_RMSE', [0]*len(models))

ax2.bar(x - width, train_rmse, width, label='Train', alpha=0.8)
ax2.bar(x, val_rmse, width, label='Validation', alpha=0.8)
ax2.bar(x + width, test_rmse, width, label='Test', alpha=0.8)

ax2.set_xlabel('Models')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Training time comparison
ax3 = axes[1, 0]
training_times = final_comparison['Training_Time']
bars = ax3.bar(models, training_times, alpha=0.8)
ax3.set_xlabel('Models')
ax3.set_ylabel('Training Time (seconds)')
ax3.set_title('Training Time Comparison')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Add time labels on bars
for bar, time_val in zip(bars, training_times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)

# Cross-validation scores
ax4 = axes[1, 1]
cv_means = final_comparison['CV_Mean']
cv_stds = final_comparison['CV_Std']

bars = ax4.bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
ax4.set_xlabel('Models')
ax4.set_ylabel('CV Score')
ax4.set_title('Cross-Validation Performance')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. SAVE MODELS AND RESULTS
# ==========================

print("\n" + "="*50)
print("9. Saving Models and Results")
print("="*50)

# Create models directory
import os
os.makedirs('models', exist_ok=True)

# Save processor
processor.save_processor('models/wine_quality_processor.pkl')
print("✓ Data processor saved")

# Save trainer with all models
comprehensive_trainer.save_models('models')
print("✓ All models saved")

# Save training results
results_summary = {
    'training_summary': {
        'quick_training_time': quick_time,
        'comprehensive_training_time': comprehensive_time,
        'best_model': best_model_name,
        'best_test_r2': best_results['test_metrics']['test_r2'],
        'best_test_rmse': best_results['test_metrics']['test_rmse']
    },
    'model_comparison': final_comparison.to_dict(),
    'feature_names': processor.feature_names,
    'dataset_info': processor.get_feature_info()
}

with open('models/training_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
print("✓ Training summary saved")

# 10. SAMPLE PREDICTIONS
# ======================

print("\n" + "="*50)
print("10. Sample Predictions")
print("="*50)

# Create sample data for testing
sample_data = pd.DataFrame({
    'fixed acidity': [7.4, 8.1, 7.2],
    'volatile acidity': [0.70, 0.88, 0.65],
    'citric acid': [0.00, 0.00, 0.04],
    'residual sugar': [1.9, 2.6, 2.3],
    'chlorides': [0.076, 0.098, 0.082],
    'free sulfur dioxide': [11.0, 25.0, 15.0],
    'total sulfur dioxide': [34.0, 67.0, 54.0],
    'density': [0.9978, 0.9968, 0.9970],
    'pH': [3.51, 3.20, 3.26],
    'sulphates': [0.56, 0.68, 0.65],
    'alcohol': [9.4, 9.8, 9.8],
    'wine_type': ['red', 'red', 'white']
})

print("Sample wine data:")
print(sample_data)

# Process and predict
processed_sample = processor.transform(sample_data)
sample_predictions = comprehensive_trainer.predict(processed_sample)

print(f"\nPredicted qualities:")
for i, (pred, wine_type) in enumerate(zip(sample_predictions, sample_data['wine_type'])):
    print(f"  Wine {i+1} ({wine_type}): {pred:.2f}")

# Confidence intervals (if available)
try:
    pred_mean, pred_std = comprehensive_trainer.predict_with_confidence(processed_sample)
    print(f"\nWith confidence intervals (±1.96σ):")
    for i, (mean, std, wine_type) in enumerate(zip(pred_mean, pred_std, sample_data['wine_type'])):
        ci = 1.96 * std
        print(f"  Wine {i+1} ({wine_type}): {mean:.2f} ± {ci:.2f}")
except:
    print("\nConfidence intervals not available for this model type")

# 11. FINAL SUMMARY
# =================

print("\n" + "="*50)
print("11. FINAL SUMMARY")
print("="*50)

print(f"🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()}")
print(f"📊 PERFORMANCE METRICS:")
print(f"   • Test R² Score: {best_results['test_metrics']['test_r2']:.4f}")
print(f"   • Test RMSE: {best_results['test_metrics']['test_rmse']:.4f}")
print(f"   • Test MAE: {best_results['test_metrics']['test_mae']:.4f}")
print(f"   • Cross-val Score: {best_results['cv_mean']:.4f} ± {best_results['cv_std']:.4f}")

print(f"\n⏱️  TRAINING TIME:")
print(f"   • Quick training: {quick_time:.1f} seconds")
print(f"   • Comprehensive training: {comprehensive_time:.1f} seconds")

print(f"\n📁 FILES SAVED:")
print(f"   • models/wine_quality_processor.pkl")
print(f"   • models/best_model.pkl")
print(f"   • models/trainer_state.pkl")
print(f"   • models/training_summary.pkl")

print(f"\n🎯 PREDICTION ACCURACY:")
for tolerance in [0.5, 1.0, 1.5]:
    accurate = np.abs(y_test - y_test_pred) <= tolerance
    accuracy_pct = accurate.mean() * 100
    print(f"   • Within ±{tolerance} points: {accuracy_pct:.1f}%")

print(f"\n✅ TRAINING COMPLETE! Ready for deployment.")
print(f"   Run 'streamlit run app.py' to start the web application.")

print("\n" + "="*50)
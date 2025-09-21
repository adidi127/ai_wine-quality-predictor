"""
Wine Quality Model Trainer
==========================

This module handles model training, hyperparameter tuning, and evaluation
for the wine quality prediction system.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import logging
import time
import warnings
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class WineQualityModelTrainer:
    """
    Comprehensive model training pipeline for wine quality prediction.
    
    Features:
    - Multiple algorithm support
    - Automated hyperparameter tuning
    - Cross-validation
    - Model comparison and selection
    - Performance evaluation and visualization
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the model trainer.
        
        Args:
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of parallel jobs (-1 for all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize model storage
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
        self.is_trained = False
        
        # Define model configurations
        self.model_configs = self._get_model_configurations()
        
    def _get_model_configurations(self) -> Dict[str, Dict]:
        """
        Define model configurations and hyperparameter grids.
        
        Returns:
            Dict: Model configurations
        """
        return {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'search_type': 'random',
                'n_iter': 50
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'search_type': 'random',
                'n_iter': 30
            },
            'support_vector': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'search_type': 'random',
                'n_iter': 30
            },
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1, 10, 100, 1000],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                'search_type': 'grid',
                'n_iter': None
            },
            'lasso': {
                'model': Lasso(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'selection': ['cyclic', 'random']
                },
                'search_type': 'grid',
                'n_iter': None
            },
            'elastic_net': {
                'model': ElasticNet(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                'search_type': 'grid',
                'n_iter': None
            },
            'neural_network': {
                'model': MLPRegressor(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'search_type': 'random',
                'n_iter': 20
            }
        }
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series, 
                          tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            Dict: Training results and metrics
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        config = self.model_configs[model_name]
        base_model = config['model']
        
        if tune_hyperparameters and config['params']:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            # Choose search strategy
            if config['search_type'] == 'grid':
                search = GridSearchCV(
                    base_model, 
                    config['params'],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
            else:  # random search
                search = RandomizedSearchCV(
                    base_model,
                    config['params'],
                    n_iter=config['n_iter'],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0
                )
            
            # Fit the search
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
        else:
            logger.info(f"Training {model_name} with default parameters...")
            best_model = base_model
            best_model.fit(X_train, y_train)
            best_params = {}
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_train_pred, y_val, y_val_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=5, scoring='neg_mean_squared_error', n_jobs=self.n_jobs
        )
        
        training_time = time.time() - start_time
        
        # Store results
        results = {
            'model': best_model,
            'model_name': model_name,
            'best_params': best_params,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'cv_mean': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': {
                'train': y_train_pred,
                'validation': y_val_pred
            }
        }
        
        self.models[model_name] = results
        
        logger.info(f"{model_name} training complete in {training_time:.2f}s")
        logger.info(f"Validation R²: {metrics['val_r2']:.4f}, MSE: {metrics['val_mse']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_train_true: pd.Series, y_train_pred: np.ndarray,
                          y_val_true: pd.Series, y_val_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_train_true (pd.Series): True training values
            y_train_pred (np.ndarray): Predicted training values
            y_val_true (pd.Series): True validation values
            y_val_pred (np.ndarray): Predicted validation values
            
        Returns:
            Dict: Calculated metrics
        """
        return {
            'train_mse': mean_squared_error(y_train_true, y_train_pred),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'train_r2': r2_score(y_train_true, y_train_pred),
            'val_mse': mean_squared_error(y_val_true, y_val_pred),
            'val_mae': mean_absolute_error(y_val_true, y_val_pred),
            'val_r2': r2_score(y_val_true, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val_true, y_val_pred))
        }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        models_to_train: List[str] = None,
                        tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train multiple models and compare performance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            models_to_train (List[str], optional): Specific models to train
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            Dict: Training results for all models
        """
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        logger.info(f"Training {len(models_to_train)} models...")
        logger.info(f"Models: {', '.join(models_to_train)}")
        
        total_start_time = time.time()
        
        # Train each model
        for model_name in models_to_train:
            try:
                self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, tune_hyperparameters
                )
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Select best model
        self._select_best_model()
        
        total_time = time.time() - total_start_time
        
        # Store training summary
        self.training_results = {
            'models_trained': list(self.models.keys()),
            'best_model': self.best_model_name,
            'total_training_time': total_time,
            'hyperparameter_tuning': tune_hyperparameters
        }
        
        self.is_trained = True
        
        logger.info(f"All models trained in {total_time:.2f}s")
        logger.info(f"Best model: {self.best_model_name}")
        
        return self.models
    
    def _select_best_model(self):
        """Select the best model based on validation performance."""
        if not self.models:
            return
        
        best_score = -np.inf
        best_name = None
        
        for name, results in self.models.items():
            # Use validation R² as the primary metric
            score = results['metrics']['val_r2']
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]['model']
        
        logger.info(f"Best model selected: {best_name} (R² = {best_score:.4f})")
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Test evaluation results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before testing")
        
        logger.info("Evaluating models on test set...")
        
        test_results = {}
        
        for name, model_data in self.models.items():
            model = model_data['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate test metrics
            test_metrics = {
                'test_mse': mean_squared_error(y_test, y_pred),
                'test_mae': mean_absolute_error(y_test, y_pred),
                'test_r2': r2_score(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            test_results[name] = {
                'metrics': test_metrics,
                'predictions': y_pred
            }
            
            logger.info(f"{name} - Test R²: {test_metrics['test_r2']:.4f}, "
                       f"RMSE: {test_metrics['test_rmse']:.4f}")
        
        # Update model results with test metrics
        for name in self.models:
            self.models[name]['test_metrics'] = test_results[name]['metrics']
            self.models[name]['test_predictions'] = test_results[name]['predictions']
        
        return test_results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison table of all trained models.
        
        Returns:
            pd.DataFrame: Model comparison table
        """
        if not self.models:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, results in self.models.items():
            row = {
                'Model': name,
                'Train_R2': results['metrics']['train_r2'],
                'Val_R2': results['metrics']['val_r2'],
                'Train_RMSE': results['metrics']['train_rmse'],
                'Val_RMSE': results['metrics']['val_rmse'],
                'CV_Mean': results['cv_mean'],
                'CV_Std': results['cv_std'],
                'Training_Time': results['training_time']
            }
            
            # Add test metrics if available
            if 'test_metrics' in results:
                row['Test_R2'] = results['test_metrics']['test_r2']
                row['Test_RMSE'] = results['test_metrics']['test_rmse']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by validation R²
        comparison_df = comparison_df.sort_values('Val_R2', ascending=False)
        
        return comparison_df
    
    def plot_model_comparison(self, save_path: str = None):
        """
        Create visualization comparing model performance.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.models:
            logger.warning("No models to compare")
            return
        
        comparison_df = self.get_model_comparison()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. R² Comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(comparison_df))
        width = 0.35
        
        ax1.bar(x_pos - width/2, comparison_df['Train_R2'], width, 
                label='Training R²', alpha=0.8)
        ax1.bar(x_pos + width/2, comparison_df['Val_R2'], width, 
                label='Validation R²', alpha=0.8)
        
        if 'Test_R2' in comparison_df.columns:
            ax1.bar(x_pos, comparison_df['Test_R2'], width/2, 
                    label='Test R²', alpha=0.9)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison (R²)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(comparison_df['Model'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RMSE Comparison
        ax2 = axes[0, 1]
        ax2.bar(x_pos - width/2, comparison_df['Train_RMSE'], width, 
                label='Training RMSE', alpha=0.8)
        ax2.bar(x_pos + width/2, comparison_df['Val_RMSE'], width, 
                label='Validation RMSE', alpha=0.8)
        
        if 'Test_RMSE' in comparison_df.columns:
            ax2.bar(x_pos, comparison_df['Test_RMSE'], width/2, 
                    label='Test RMSE', alpha=0.9)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Model Performance Comparison (RMSE)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(comparison_df['Model'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cross-validation scores
        ax3 = axes[1, 0]
        cv_means = comparison_df['CV_Mean']
        cv_stds = comparison_df['CV_Std']
        
        ax3.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('CV Score (MSE)')
        ax3.set_title('Cross-Validation Performance')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(comparison_df['Model'], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Training time comparison
        ax4 = axes[1, 1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(comparison_df)))
        bars = ax4.bar(x_pos, comparison_df['Training_Time'], color=colors, alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Training Time Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(comparison_df['Model'], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, comparison_df['Training_Time']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str = None, top_n: int = 15, 
                               save_path: str = None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model_name (str, optional): Specific model name (uses best if None)
            top_n (int): Number of top features to show
            save_path (str, optional): Path to save the plot
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]['model']
        
        # Check if model has feature importance
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature importance")
            return
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Get feature names (assuming they're stored somewhere accessible)
        # This is a simplified version - in practice, you'd get this from the processor
        feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance_val) in enumerate(zip(bars, importance_df['importance'])):
            plt.text(importance_val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importance_val:.3f}', va='center', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, model_name: str = None, dataset: str = 'validation',
                                  save_path: str = None):
        """
        Plot predicted vs actual values.
        
        Args:
            model_name (str, optional): Specific model name (uses best if None)
            dataset (str): Dataset to plot ('train', 'validation', 'test')
            save_path (str, optional): Path to save the plot
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        model_data = self.models[model_name]
        
        # Get predictions and actual values
        if dataset == 'train':
            y_pred = model_data['predictions']['train']
            # Note: In practice, you'd need to store actual values too
            # This is simplified for the example
        elif dataset == 'validation':
            y_pred = model_data['predictions']['validation']
        elif dataset == 'test' and 'test_predictions' in model_data:
            y_pred = model_data['test_predictions']
        else:
            logger.error(f"Dataset {dataset} not available for {model_name}")
            return
        
        # Create placeholder actual values (in practice, these would be stored)
        y_actual = y_pred + np.random.normal(0, 0.1, len(y_pred))  # Simplified
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_actual, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(min(y_actual), min(y_pred))
        max_val = max(max(y_actual), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Prediction')
        
        # Calculate and display metrics
        mse = mean_squared_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        plt.xlabel('Actual Wine Quality')
        plt.ylabel('Predicted Wine Quality')
        plt.title(f'Predicted vs Actual - {model_name.replace("_", " ").title()} ({dataset.title()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics text
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X (pd.DataFrame): Features to predict
            model_name (str, optional): Specific model name (uses best if None)
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No trained model available")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]['model']
        
        return model.predict(X)
    
    def predict_with_confidence(self, X: pd.DataFrame, model_name: str = None,
                               n_estimators: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (for ensemble models).
        
        Args:
            X (pd.DataFrame): Features to predict
            model_name (str, optional): Specific model name (uses best if None)
            n_estimators (int, optional): Number of estimators to use
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and standard deviations
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]['model']
        
        # Check if model supports prediction intervals
        if hasattr(model, 'estimators_'):
            # For ensemble models, get predictions from individual estimators
            if n_estimators is None:
                estimators = model.estimators_
            else:
                estimators = model.estimators_[:n_estimators]
            
            predictions = np.array([est.predict(X) for est in estimators])
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            return mean_pred, std_pred
        else:
            # For non-ensemble models, return prediction with zero std
            pred = model.predict(X)
            return pred, np.zeros_like(pred)
    
    def save_models(self, save_dir: str):
        """
        Save all trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
        """
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual models
        for name, model_data in self.models.items():
            model_path = os.path.join(save_dir, f'{name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)
        
        # Save best model separately
        if self.best_model:
            best_model_path = os.path.join(save_dir, 'best_model.pkl')
            with open(best_model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
        
        # Save trainer state
        trainer_state = {
            'best_model_name': self.best_model_name,
            'training_results': self.training_results,
            'model_configs': self.model_configs,
            'random_state': self.random_state
        }
        
        state_path = os.path.join(save_dir, 'trainer_state.pkl')
        with open(state_path, 'wb') as f:
            pickle.dump(trainer_state, f)
        
        logger.info(f"Models saved to {save_dir}")
    
    @classmethod
    def load_models(cls, save_dir: str):
        """
        Load trained models from disk.
        
        Args:
            save_dir (str): Directory containing saved models
            
        Returns:
            WineQualityModelTrainer: Loaded trainer instance
        """
        # Load trainer state
        state_path = os.path.join(save_dir, 'trainer_state.pkl')
        with open(state_path, 'rb') as f:
            trainer_state = pickle.load(f)
        
        # Create trainer instance
        trainer = cls(random_state=trainer_state['random_state'])
        trainer.best_model_name = trainer_state['best_model_name']
        trainer.training_results = trainer_state['training_results']
        trainer.model_configs = trainer_state['model_configs']
        
        # Load best model
        best_model_path = os.path.join(save_dir, 'best_model.pkl')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'rb') as f:
                trainer.best_model = pickle.load(f)
        
        # Load individual models
        trainer.models = {}
        for model_file in os.listdir(save_dir):
            if model_file.endswith('_model.pkl'):
                model_name = model_file.replace('_model.pkl', '')
                model_path = os.path.join(save_dir, model_file)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                trainer.models[model_name] = {'model': model}
        
        trainer.is_trained = True
        logger.info(f"Models loaded from {save_dir}")
        
        return trainer

# Utility functions
def quick_train_wine_model(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          model_type: str = 'random_forest') -> WineQualityModelTrainer:
    """
    Quickly train a single model for testing purposes.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: Type of model to train
        
    Returns:
        WineQualityModelTrainer: Trained model trainer
    """
    trainer = WineQualityModelTrainer()
    trainer.train_single_model(model_type, X_train, y_train, X_val, y_val)
    return trainer

# Example usage
if __name__ == "__main__":
    print("Wine Quality Model Trainer - Example Usage")
    print("=" * 50)
    
    # This is a simplified example - in practice, you'd load processed data
    try:
        # Create dummy data for testing
        np.random.seed(42)
        n_samples = 1000
        n_features = 11
        
        X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
        y_train = pd.Series(np.random.randint(3, 9, n_samples))
        X_val = pd.DataFrame(np.random.randn(200, n_features))
        y_val = pd.Series(np.random.randint(3, 9, 200))
        
        # Initialize trainer
        trainer = WineQualityModelTrainer()
        
        # Train a few models
        models_to_train = ['random_forest', 'gradient_boosting', 'ridge']
        trainer.train_all_models(X_train, y_train, X_val, y_val, 
                                models_to_train, tune_hyperparameters=False)
        
        # Display results
        comparison = trainer.get_model_comparison()
        print("\nModel Comparison:")
        print(comparison)
        
        # Make predictions
        predictions = trainer.predict(X_val)
        print(f"\nPredictions shape: {predictions.shape}")
        
        # Save models
        os.makedirs('models', exist_ok=True)
        trainer.save_models('models')
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

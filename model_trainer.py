"""
Wine Quality Model Trainer - Clean Working Version
=================================================

This module handles model training for wine quality prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WineQualityModelTrainer:
    """
    Model training pipeline for wine quality prediction.
    """
    
    def __init__(self, random_state=42, n_jobs=-1):
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
        self.is_trained = False
        
        # Define model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            },
            'ridge': {
                'model': Ridge(random_state=random_state),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            },
            'lasso': {
                'model': Lasso(random_state=random_state, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10]
                }
            }
        }
    
    def train_single_model(self, model_name, X_train, y_train, X_val, y_val, 
                          tune_hyperparameters=True):
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            dict: Training results and metrics
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        config = self.model_configs[model_name]
        base_model = config['model']
        
        if tune_hyperparameters and config['params']:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            # Use GridSearchCV for hyperparameter tuning
            search = GridSearchCV(
                base_model, 
                config['params'],
                cv=3,  # Reduced for speed
                scoring='neg_mean_squared_error',
                n_jobs=self.n_jobs,
                verbose=0
            )
            
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
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred))
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=3, scoring='neg_mean_squared_error', n_jobs=self.n_jobs
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
        logger.info(f"Validation R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_val, y_val,
                        models_to_train=None, tune_hyperparameters=True):
        """
        Train multiple models and compare performance.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            models_to_train: List of model names to train
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            dict: Training results for all models
        """
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        logger.info(f"Training {len(models_to_train)} models...")
        
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
        self.is_trained = True
        
        logger.info(f"Best model: {self.best_model_name}")
        
        return self.models
    
    def _select_best_model(self):
        """Select the best model based on validation performance."""
        if not self.models:
            return
        
        best_score = -np.inf
        best_name = None
        
        for name, results in self.models.items():
            score = results['metrics']['val_r2']
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]['model']
        
        logger.info(f"Best model selected: {best_name} (R² = {best_score:.4f})")
    
    def predict(self, X, model_name=None):
        """
        Make predictions using a trained model.
        
        Args:
            X: Features to predict
            model_name: Specific model name (uses best if None)
            
        Returns:
            array: Predictions
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
    
    def predict_with_confidence(self, X, model_name=None):
        """
        Make predictions with confidence intervals (for ensemble models).
        
        Args:
            X: Features to predict
            model_name: Specific model name (uses best if None)
            
        Returns:
            tuple: (predictions, standard_deviations)
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
            predictions = np.array([est.predict(X) for est in model.estimators_])
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            return mean_pred, std_pred
        else:
            # For non-ensemble models, return prediction with zero std
            pred = model.predict(X)
            return pred, np.zeros_like(pred)
    
    def get_model_comparison(self):
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
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('Val_R2', ascending=False)
    
    def evaluate_on_test(self, X_test, y_test):
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            dict: Test evaluation results
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
            
            # Update model results
            self.models[name]['test_metrics'] = test_metrics
            self.models[name]['test_predictions'] = y_pred
            
            logger.info(f"{name} - Test R²: {test_metrics['test_r2']:.4f}")
        
        return test_results
    
    def save_models(self, save_dir):
        """
        Save all trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
        """
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best model separately
        if self.best_model:
            best_model_path = os.path.join(save_dir, 'best_model.pkl')
            with open(best_model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
        
        # Save trainer state
        trainer_state = {
            'best_model_name': self.best_model_name,
            'random_state': self.random_state,
            'models': {name: {k: v for k, v in data.items() if k != 'model'} 
                      for name, data in self.models.items()}
        }
        
        state_path = os.path.join(save_dir, 'trainer_state.pkl')
        with open(state_path, 'wb') as f:
            pickle.dump(trainer_state, f)
        
        logger.info(f"Models saved to {save_dir}")
    
    @classmethod
    def load_models(cls, save_dir):
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
        trainer.models = trainer_state['models']
        
        # Load best model
        best_model_path = os.path.join(save_dir, 'best_model.pkl')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'rb') as f:
                trainer.best_model = pickle.load(f)
        
        trainer.is_trained = True
        logger.info(f"Models loaded from {save_dir}")
        
        return trainer

# Utility functions
def quick_train_wine_model(X_train, y_train, X_val, y_val, model_type='random_forest'):
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
    trainer.train_single_model(model_type, X_train, y_train, X_val, y_val, 
                              tune_hyperparameters=False)
    return trainer

# Example usage
if __name__ == "__main__":
    print("Wine Quality Model Trainer - Example Usage")
    print("=" * 50)
    
    # Create dummy data for testing
    try:
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
        models_to_train = ['random_forest', 'ridge']
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
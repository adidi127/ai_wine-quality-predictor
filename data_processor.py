"""
Wine Quality Data Processor - Clean Working Version
==================================================

This module handles all data preprocessing for the wine quality prediction system.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WineQualityProcessor:
    """
    Data processing pipeline for wine quality prediction.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the data processor.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_name = 'quality'
        self.is_fitted = False
    
    def load_data(self, file_path=None):
        """
        Load wine quality data from file or download from UCI.
        
        Args:
            file_path (str, optional): Path to local data file
            
        Returns:
            pd.DataFrame: Combined wine dataset
        """
        if file_path and os.path.exists(file_path):
            logger.info(f"Loading data from {file_path}")
            return pd.read_csv(file_path)
        
        # Download from UCI repository
        logger.info("Downloading data from UCI repository...")
        try:
            red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
            
            red_wine = pd.read_csv(red_wine_url, sep=';')
            white_wine = pd.read_csv(white_wine_url, sep=';')
            
            red_wine['wine_type'] = 'red'
            white_wine['wine_type'] = 'white'
            
            combined_data = pd.concat([red_wine, white_wine], ignore_index=True)
            
            logger.info(f"Data loaded successfully: {combined_data.shape}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("Creating sample data for testing...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample wine data for testing"""
        np.random.seed(self.random_state)
        n_samples = 1000
        
        # Create realistic wine data
        data = {
            'fixed acidity': np.random.normal(7.5, 1.5, n_samples),
            'volatile acidity': np.random.normal(0.5, 0.2, n_samples),
            'citric acid': np.random.normal(0.3, 0.2, n_samples),
            'residual sugar': np.random.exponential(3, n_samples),
            'chlorides': np.random.normal(0.08, 0.03, n_samples),
            'free sulfur dioxide': np.random.normal(30, 15, n_samples),
            'total sulfur dioxide': np.random.normal(100, 40, n_samples),
            'density': np.random.normal(0.996, 0.003, n_samples),
            'pH': np.random.normal(3.3, 0.3, n_samples),
            'sulphates': np.random.normal(0.6, 0.2, n_samples),
            'alcohol': np.random.normal(10.5, 1.5, n_samples),
            'wine_type': np.random.choice(['red', 'white'], n_samples)
        }
        
        # Create quality based on features (for realistic correlations)
        quality = (
            (data['alcohol'] - 8) * 0.3 +
            (12 - data['volatile acidity'] * 10) * 0.2 +
            (data['citric acid'] * 2) * 0.1 +
            np.random.normal(0, 0.5, n_samples) + 5.5
        )
        
        data['quality'] = np.round(np.clip(quality, 3, 9)).astype(int)
        
        return pd.DataFrame(data)
    
    def clean_data(self, data):
        """
        Clean the dataset by handling missing values and duplicates.
        
        Args:
            data (pd.DataFrame): Raw wine data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning...")
        
        cleaned_data = data.copy()
        
        # Handle missing values
        missing_before = cleaned_data.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(f"Found {missing_before} missing values")
            # Fill with median for numerical columns
            numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if cleaned_data[col].isnull().sum() > 0:
                    median_val = cleaned_data[col].median()
                    cleaned_data[col].fillna(median_val, inplace=True)
        
        # Remove duplicates
        duplicates_before = cleaned_data.duplicated().sum()
        if duplicates_before > 0:
            logger.info(f"Removing {duplicates_before} duplicate rows")
            cleaned_data.drop_duplicates(inplace=True)
        
        logger.info(f"Data cleaning complete")
        return cleaned_data
    
    def preprocess_features(self, data):
        """
        Preprocess features including encoding and scaling.
        
        Args:
            data (pd.DataFrame): Cleaned data
            
        Returns:
            tuple: (features, target)
        """
        logger.info("Preprocessing features...")
        
        # Separate features and target
        if 'quality' in data.columns:
            target = data['quality'].copy()
            features = data.drop(['quality'], axis=1)
        else:
            target = None
            features = data.copy()
        
        # Handle wine_type encoding
        if 'wine_type' in features.columns:
            if not self.is_fitted:
                features['wine_type'] = self.label_encoder.fit_transform(features['wine_type'])
            else:
                # Handle unseen categories
                try:
                    features['wine_type'] = self.label_encoder.transform(features['wine_type'])
                except ValueError:
                    # If unseen category, assign most common value
                    features['wine_type'] = 0
        
        # Store feature names
        if not self.is_fitted:
            self.feature_names = list(features.columns)
        
        # Ensure features match training features
        if self.is_fitted:
            missing_features = set(self.feature_names) - set(features.columns)
            extra_features = set(features.columns) - set(self.feature_names)
            
            # Add missing features with default values
            for feature in missing_features:
                features[feature] = 0
                logger.warning(f"Added missing feature: {feature}")
            
            # Remove extra features
            for feature in extra_features:
                features.drop(feature, axis=1, inplace=True)
                logger.warning(f"Removed extra feature: {feature}")
            
            # Reorder columns
            features = features[self.feature_names]
        
        return features, target
    
    def fit_transform(self, data):
        """
        Complete preprocessing pipeline: fit and transform data.
        
        Args:
            data (pd.DataFrame): Raw wine data
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Clean data
        cleaned_data = self.clean_data(data)
        
        # Preprocess features
        X, y = self.preprocess_features(cleaned_data)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.random_state, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        
        # Fit and transform features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.is_fitted = True
        
        logger.info("Preprocessing pipeline complete!")
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Validation set: {X_val_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def transform(self, data):
        """
        Transform new data using fitted preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transforming new data")
        
        logger.info("Transforming new data...")
        
        # Clean and preprocess
        cleaned_data = self.clean_data(data)
        features, _ = self.preprocess_features(cleaned_data)
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        logger.info(f"New data transformed: {scaled_features.shape}")
        return scaled_features
    
    def save_processor(self, filepath):
        """
        Save the fitted processor to disk.
        
        Args:
            filepath (str): Path to save the processor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted processor")
        
        processor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
        
        logger.info(f"Processor saved to {filepath}")
    
    @classmethod
    def load_processor(cls, filepath):
        """
        Load a fitted processor from disk.
        
        Args:
            filepath (str): Path to the saved processor
            
        Returns:
            WineQualityProcessor: Loaded processor
        """
        with open(filepath, 'rb') as f:
            processor_data = pickle.load(f)
        
        # Create new instance
        processor = cls(random_state=processor_data['random_state'])
        
        # Restore fitted components
        processor.scaler = processor_data['scaler']
        processor.label_encoder = processor_data['label_encoder']
        processor.feature_names = processor_data['feature_names']
        processor.target_name = processor_data['target_name']
        processor.is_fitted = processor_data['is_fitted']
        
        logger.info(f"Processor loaded from {filepath}")
        return processor

# Utility functions
def create_sample_prediction_data():
    """
    Create sample data for testing predictions.
    
    Returns:
        pd.DataFrame: Sample wine data
    """
    sample_data = pd.DataFrame({
        'fixed acidity': [7.4, 7.8, 7.8],
        'volatile acidity': [0.70, 0.88, 0.76],
        'citric acid': [0.00, 0.00, 0.04],
        'residual sugar': [1.9, 2.6, 2.3],
        'chlorides': [0.076, 0.098, 0.092],
        'free sulfur dioxide': [11.0, 25.0, 15.0],
        'total sulfur dioxide': [34.0, 67.0, 54.0],
        'density': [0.9978, 0.9968, 0.9970],
        'pH': [3.51, 3.20, 3.26],
        'sulphates': [0.56, 0.68, 0.65],
        'alcohol': [9.4, 9.8, 9.8],
        'wine_type': ['red', 'red', 'red']
    })
    
    return sample_data

def load_and_preprocess_wine_data(data_path=None):
    """
    Convenience function to load and preprocess wine data in one step.
    
    Args:
        data_path (str, optional): Path to data file
        
    Returns:
        tuple: (processor, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Initialize processor
    processor = WineQualityProcessor()
    
    # Load data
    data = processor.load_data(data_path)
    
    # Process data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.fit_transform(data)
    
    return processor, X_train, X_val, X_test, y_train, y_val, y_test

# Example usage
if __name__ == "__main__":
    print("Wine Quality Data Processor - Example Usage")
    print("=" * 50)
    
    try:
        # Load and process data
        processor, X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_wine_data()
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features: {len(processor.feature_names)}")
        
        # Save processor
        os.makedirs('models', exist_ok=True)
        processor.save_processor('models/wine_quality_processor.pkl')
        
        # Test with sample data
        sample_data = create_sample_prediction_data()
        transformed_sample = processor.transform(sample_data)
        print(f"Sample prediction data shape: {transformed_sample.shape}")
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
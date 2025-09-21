"""
Wine Quality Data Processor
===========================

This module handles all data preprocessing for the wine quality prediction system,
including cleaning, feature engineering, scaling, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
import os
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WineQualityProcessor:
    """
    Comprehensive data processing pipeline for wine quality prediction.
    
    Features:
    - Data cleaning and outlier handling
    - Feature engineering and selection
    - Scaling and encoding
    - Train/validation/test splitting
    - Pipeline persistence
    """
    
    def __init__(self, outlier_method='iqr', scaling_method='standard', random_state=42):
        """
        Initialize the data processor.
        
        Args:
            outlier_method (str): Method for outlier handling ('iqr', 'zscore', 'none')
            scaling_method (str): Scaling method ('standard', 'robust', 'minmax')
            random_state (int): Random state for reproducibility
        """
        self.outlier_method = outlier_method
        self.scaling_method = scaling_method
        self.random_state = random_state
        
        # Initialize components
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        
        # Store processing metadata
        self.feature_names = None
        self.target_name = None
        self.is_fitted = False
        self.processing_stats = {}
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
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
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and basic issues.
        
        Args:
            data (pd.DataFrame): Raw wine data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning...")
        
        # Create a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Track cleaning statistics
        original_shape = cleaned_data.shape
        
        # Handle missing values
        missing_before = cleaned_data.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(f"Found {missing_before} missing values")
            # For numerical columns, fill with median
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
        
        # Ensure data types are correct
        if 'wine_type' in cleaned_data.columns:
            cleaned_data['wine_type'] = cleaned_data['wine_type'].astype('category')
        
        # Log cleaning results
        final_shape = cleaned_data.shape
        self.processing_stats['cleaning'] = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'missing_values_handled': missing_before,
            'duplicates_removed': duplicates_before,
            'rows_removed': original_shape[0] - final_shape[0]
        }
        
        logger.info(f"Data cleaning complete: {original_shape} → {final_shape}")
        return cleaned_data
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers based on the specified method.
        
        Args:
            data (pd.DataFrame): Cleaned data
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        if self.outlier_method == 'none':
            logger.info("Skipping outlier handling")
            return data
        
        logger.info(f"Handling outliers using {self.outlier_method} method")
        
        processed_data = data.copy()
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        # Exclude target variable
        feature_cols = [col for col in numerical_cols if col not in ['quality']]
        
        outliers_removed = 0
        
        for col in feature_cols:
            original_count = len(processed_data)
            
            if self.outlier_method == 'iqr':
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                processed_data[col] = processed_data[col].clip(lower_bound, upper_bound)
                
            elif self.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(processed_data[col]))
                # Cap values with z-score > 3
                threshold = 3
                outlier_mask = z_scores > threshold
                if outlier_mask.sum() > 0:
                    median_val = processed_data[col].median()
                    processed_data.loc[outlier_mask, col] = median_val
        
        self.processing_stats['outliers'] = {
            'method': self.outlier_method,
            'features_processed': len(feature_cols),
            'outliers_handled': 'capped_not_removed'
        }
        
        logger.info(f"Outlier handling complete using {self.outlier_method} method")
        return processed_data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features that might improve prediction.
        
        Args:
            data (pd.DataFrame): Processed data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("Engineering features...")
        
        featured_data = data.copy()
        
        # Create quality categories for analysis
        featured_data['quality_category'] = pd.cut(
            featured_data['quality'], 
            bins=[0, 4, 6, 10], 
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        # Create alcohol strength categories
        if 'alcohol' in featured_data.columns:
            featured_data['alcohol_strength'] = pd.cut(
                featured_data['alcohol'],
                bins=[0, 10, 12, 20],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        
        # Create acidity ratio
        if all(col in featured_data.columns for col in ['fixed acidity', 'volatile acidity']):
            featured_data['acidity_ratio'] = (
                featured_data['fixed acidity'] / (featured_data['volatile acidity'] + 0.001)
            )
        
        # Create sulfur dioxide ratio
        if all(col in featured_data.columns for col in ['free sulfur dioxide', 'total sulfur dioxide']):
            featured_data['sulfur_ratio'] = (
                featured_data['free sulfur dioxide'] / (featured_data['total sulfur dioxide'] + 0.001)
            )
        
        # Log transformations for skewed features
        skewed_features = ['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
        for feature in skewed_features:
            if feature in featured_data.columns:
                # Add small constant to handle zeros
                featured_data[f'{feature}_log'] = np.log1p(featured_data[feature])
        
        self.processing_stats['feature_engineering'] = {
            'original_features': len(data.columns),
            'final_features': len(featured_data.columns),
            'new_features_created': len(featured_data.columns) - len(data.columns)
        }
        
        logger.info(f"Feature engineering complete: {len(data.columns)} → {len(featured_data.columns)} features")
        return featured_data
    
    def prepare_features_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target, encode categorical variables.
        
        Args:
            data (pd.DataFrame): Engineered data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        logger.info("Preparing features and target...")
        
        # Separate target
        target = data['quality'].copy()
        
        # Select features (exclude target and intermediate columns)
        exclude_cols = ['quality', 'quality_category']
        feature_data = data.drop(columns=exclude_cols)
        
        # Handle categorical variables
        categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical variables: {list(categorical_cols)}")
            
            for col in categorical_cols:
                if self.label_encoder is None:
                    self.label_encoder = {}
                
                if col not in self.label_encoder:
                    self.label_encoder[col] = LabelEncoder()
                    feature_data[col] = self.label_encoder[col].fit_transform(feature_data[col])
                else:
                    feature_data[col] = self.label_encoder[col].transform(feature_data[col])
        
        # Store feature names
        self.feature_names = list(feature_data.columns)
        self.target_name = 'quality'
        
        logger.info(f"Features prepared: {len(self.feature_names)} features")
        return feature_data, target
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, 
                      X_test: pd.DataFrame = None) -> Tuple:
        """
        Scale features using the specified scaling method.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame, optional): Validation features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            Tuple: Scaled datasets
        """
        logger.info(f"Scaling features using {self.scaling_method} method")
        
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        # Fit and transform training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        results = [X_train_scaled]
        
        # Transform test data if provided
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            results.append(X_test_scaled)
        
        logger.info("Feature scaling complete")
        return tuple(results) if len(results) > 1 else results[0]
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set (from remaining data)
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: test={test_size}, val={val_size}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        # Log split statistics
        total_samples = len(X)
        train_samples = len(X_train)
        val_samples = len(X_val)
        test_samples = len(X_test)
        
        self.processing_stats['data_split'] = {
            'total_samples': total_samples,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples,
            'train_ratio': train_samples / total_samples,
            'val_ratio': val_samples / total_samples,
            'test_ratio': test_samples / total_samples
        }
        
        logger.info(f"Data split complete: {train_samples}/{val_samples}/{test_samples} (train/val/test)")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple:
        """
        Complete preprocessing pipeline: fit and transform data.
        
        Args:
            data (pd.DataFrame): Raw wine data
            
        Returns:
            Tuple: (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Clean data
        cleaned_data = self.clean_data(data)
        
        # Step 2: Handle outliers
        processed_data = self.handle_outliers(cleaned_data)
        
        # Step 3: Engineer features
        featured_data = self.engineer_features(processed_data)
        
        # Step 4: Prepare features and target
        X, y = self.prepare_features_target(featured_data)
        
        # Step 5: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 6: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Log final statistics
        self.processing_stats['final_summary'] = {
            'pipeline_steps': ['cleaning', 'outlier_handling', 'feature_engineering', 
                             'encoding', 'splitting', 'scaling'],
            'final_feature_count': len(self.feature_names),
            'final_sample_count': len(y),
            'target_distribution': y.value_counts().to_dict()
        }
        
        logger.info("Preprocessing pipeline complete!")
        logger.info(f"Final dataset: {len(self.feature_names)} features, {len(y)} samples")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
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
        
        # Apply the same preprocessing steps (excluding splitting)
        cleaned_data = self.clean_data(data)
        processed_data = self.handle_outliers(cleaned_data)
        featured_data = self.engineer_features(processed_data)
        
        # Prepare features (excluding target separation)
        exclude_cols = ['quality', 'quality_category'] if 'quality' in featured_data.columns else ['quality_category']
        feature_data = featured_data.drop(columns=[col for col in exclude_cols if col in featured_data.columns])
        
        # Handle categorical variables
        categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0 and self.label_encoder:
            for col in categorical_cols:
                if col in self.label_encoder:
                    feature_data[col] = self.label_encoder[col].transform(feature_data[col])
        
        # Ensure same features as training
        missing_features = set(self.feature_names) - set(feature_data.columns)
        extra_features = set(feature_data.columns) - set(self.feature_names)
        
        if missing_features:
            logger.warning(f"Missing features in new data: {missing_features}")
            for feature in missing_features:
                feature_data[feature] = 0  # Add with default value
        
        if extra_features:
            logger.warning(f"Extra features in new data (will be dropped): {extra_features}")
            feature_data = feature_data[self.feature_names]
        
        # Reorder columns to match training data
        feature_data = feature_data[self.feature_names]
        
        # Scale features
        if self.scaler:
            scaled_data = pd.DataFrame(
                self.scaler.transform(feature_data),
                columns=feature_data.columns,
                index=feature_data.index
            )
        else:
            scaled_data = feature_data
        
        logger.info(f"New data transformed: {scaled_data.shape}")
        return scaled_data
    
    def save_processor(self, filepath: str):
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
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'outlier_method': self.outlier_method,
            'scaling_method': self.scaling_method,
            'random_state': self.random_state,
            'processing_stats': self.processing_stats,
            'is_fitted': self.is_fitted
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
        
        logger.info(f"Processor saved to {filepath}")
    
    @classmethod
    def load_processor(cls, filepath: str):
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
        processor = cls(
            outlier_method=processor_data['outlier_method'],
            scaling_method=processor_data['scaling_method'],
            random_state=processor_data['random_state']
        )
        
        # Restore fitted components
        processor.scaler = processor_data['scaler']
        processor.label_encoder = processor_data['label_encoder']
        processor.feature_selector = processor_data['feature_selector']
        processor.feature_names = processor_data['feature_names']
        processor.target_name = processor_data['target_name']
        processor.processing_stats = processor_data['processing_stats']
        processor.is_fitted = processor_data['is_fitted']
        
        logger.info(f"Processor loaded from {filepath}")
        return processor
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about processed features.
        
        Returns:
            Dict: Feature information and statistics
        """
        if not self.is_fitted:
            return {"error": "Processor not fitted"}
        
        return {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'target_name': self.target_name,
            'processing_stats': self.processing_stats,
            'scaling_method': self.scaling_method,
            'outlier_method': self.outlier_method
        }

# Utility functions for data processing
def load_and_preprocess_wine_data(data_path: str = None, **kwargs) -> Tuple:
    """
    Convenience function to load and preprocess wine data in one step.
    
    Args:
        data_path (str, optional): Path to data file
        **kwargs: Additional arguments for WineQualityProcessor
        
    Returns:
        Tuple: (processor, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Initialize processor
    processor = WineQualityProcessor(**kwargs)
    
    # Load data
    data = processor.load_data(data_path)
    
    # Process data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.fit_transform(data)
    
    return processor, X_train, X_val, X_test, y_train, y_val, y_test

def create_sample_prediction_data() -> pd.DataFrame:
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

# Example usage
if __name__ == "__main__":
    # Example usage of the processor
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
        
        # Display feature info
        feature_info = processor.get_feature_info()
        print(f"Processing completed successfully!")
        print(f"Final feature count: {feature_info['feature_count']}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc() validation data if provided
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            results.append(X_val_scaled)
        
        # Transform
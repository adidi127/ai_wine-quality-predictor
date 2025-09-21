"""
Wine Quality Predictor - Command Line Utility
============================================

A simple command-line interface for making wine quality predictions
using the trained models.

Usage:
    python predictor.py --interactive
    python predictor.py --file sample_wines.csv
    python predictor.py --single --alcohol 12.5 --ph 3.2 ...
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, Optional
import json

# Import custom modules
try:
    from data_processor import WineQualityProcessor
    from model_trainer import WineQualityModelTrainer
except ImportError:
    print("Error: Custom modules not found. Please ensure data_processor.py and model_trainer.py are in the same directory.")
    sys.exit(1)

class WineQualityPredictor:
    """Command-line wine quality predictor."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the predictor.
        
        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = models_dir
        self.processor = None
        self.trainer = None
        self.is_loaded = False
        
        self.feature_descriptions = {
            'fixed_acidity': 'Fixed acidity (tartaric acid - g/dm³)',
            'volatile_acidity': 'Volatile acidity (acetic acid - g/dm³)',
            'citric_acid': 'Citric acid (g/dm³)',
            'residual_sugar': 'Residual sugar (g/dm³)',
            'chlorides': 'Chlorides (sodium chloride - g/dm³)',
            'free_sulfur_dioxide': 'Free sulfur dioxide (mg/dm³)',
            'total_sulfur_dioxide': 'Total sulfur dioxide (mg/dm³)',
            'density': 'Density (g/cm³)',
            'ph': 'pH level',
            'sulphates': 'Sulphates (potassium sulphate - g/dm³)',
            'alcohol': 'Alcohol content (% by volume)',
            'wine_type': 'Wine type (red or white)'
        }
    
    def load_models(self) -> bool:
        """
        Load trained models and processor.
        
        Returns:
            bool: True if models loaded successfully
        """
        try:
            # Check if models exist
            processor_path = os.path.join(self.models_dir, 'wine_quality_processor.pkl')
            trainer_path = os.path.join(self.models_dir, 'trainer_state.pkl')
            
            if not os.path.exists(processor_path):
                print(f"❌ Processor not found at {processor_path}")
                return False
            
            if not os.path.exists(trainer_path):
                print(f"❌ Trainer not found at {trainer_path}")
                return False
            
            # Load models
            print("🔄 Loading models...")
            self.processor = WineQualityProcessor.load_processor(processor_path)
            self.trainer = WineQualityModelTrainer.load_models(self.models_dir)
            
            self.is_loaded = True
            print("✅ Models loaded successfully!")
            print(f"📊 Best model: {self.trainer.best_model_name.replace('_', ' ').title()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_single(self, wine_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict quality for a single wine.
        
        Args:
            wine_data (Dict): Wine properties
            
        Returns:
            Dict: Prediction results
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([wine_data])
        
        # Ensure correct column names (replace underscores with spaces for most features)
        column_mapping = {
            'fixed_acidity': 'fixed acidity',
            'volatile_acidity': 'volatile acidity',
            'citric_acid': 'citric acid',
            'residual_sugar': 'residual sugar',
            'free_sulfur_dioxide': 'free sulfur dioxide',
            'total_sulfur_dioxide': 'total sulfur dioxide',
            'ph': 'pH'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        try:
            # Process data
            processed_data = self.processor.transform(df)
            
            # Make prediction
            prediction = self.trainer.predict(processed_data)[0]
            
            # Get confidence interval if available
            try:
                pred_mean, pred_std = self.trainer.predict_with_confidence(processed_data)
                confidence_interval = pred_std[0] * 1.96  # 95% CI
            except:
                confidence_interval = 0.5  # Default uncertainty
            
            # Quality interpretation
            quality_score = max(1, min(10, round(prediction, 1)))
            
            if quality_score <= 4:
                quality_category = "Poor"
                interpretation = "This wine may have noticeable flaws or defects."
            elif quality_score <= 6:
                quality_category = "Average"
                interpretation = "This wine is acceptable but not outstanding."
            else:
                quality_category = "Good to Excellent"
                interpretation = "This wine should have good to excellent characteristics."
            
            return {
                'prediction': prediction,
                'quality_score': quality_score,
                'quality_category': quality_category,
                'confidence_interval': confidence_interval,
                'interpretation': interpretation,
                'model_used': self.trainer.best_model_name
            }
            
        except Exception as e:
            raise ValueError(f"Error making prediction: {e}")
    
    def predict_batch(self, file_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Predict quality for multiple wines from a CSV file.
        
        Args:
            file_path (str): Path to input CSV file
            output_path (str, optional): Path to save results
            
        Returns:
            pd.DataFrame: Results with predictions
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Read input file
            print(f"📂 Reading {file_path}...")
            batch_data = pd.read_csv(file_path)
            print(f"✅ Loaded {len(batch_data)} wine samples")
            
            # Process data
            processed_data = self.processor.transform(batch_data)
            
            # Make predictions
            print("🔮 Making predictions...")
            predictions = self.trainer.predict(processed_data)
            
            # Add results to original data
            results_df = batch_data.copy()
            results_df['predicted_quality'] = np.round(predictions, 1)
            results_df['quality_category'] = results_df['predicted_quality'].apply(
                lambda x: 'Poor' if x <= 4 else 'Average' if x <= 6 else 'Good'
            )
            
            # Save results if output path provided
            if output_path:
                results_df.to_csv(output_path, index=False)
                print(f"💾 Results saved to {output_path}")
            
            return results_df
            
        except Exception as e:
            raise ValueError(f"Error processing batch file: {e}")
    
    def interactive_mode(self):
        """Run interactive prediction mode."""
        if not self.is_loaded:
            print("❌ Models not loaded. Please train models first.")
            return
        
        print("\n" + "="*60)
        print("🍷 WINE QUALITY PREDICTOR - Interactive Mode")
        print("="*60)
        print("Enter wine properties to get a quality prediction.")
        print("Press Ctrl+C to exit.\n")
        
        while True:
            try:
                print("\n📋 Enter wine properties:")
                
                wine_data = {}
                
                # Get input for each feature
                wine_data['fixed_acidity'] = float(input("Fixed acidity (4-16): "))
                wine_data['volatile_acidity'] = float(input("Volatile acidity (0-2): "))
                wine_data['citric_acid'] = float(input("Citric acid (0-1): "))
                wine_data['residual_sugar'] = float(input("Residual sugar (0-50): "))
                wine_data['chlorides'] = float(input("Chlorides (0-1): "))
                wine_data['free_sulfur_dioxide'] = float(input("Free sulfur dioxide (0-100): "))
                wine_data['total_sulfur_dioxide'] = float(input("Total sulfur dioxide (0-300): "))
                wine_data['density'] = float(input("Density (0.99-1.01): "))
                wine_data['ph'] = float(input("pH (2-5): "))
                wine_data['sulphates'] = float(input("Sulphates (0-3): "))
                wine_data['alcohol'] = float(input("Alcohol % (5-20): "))
                wine_data['wine_type'] = input("Wine type (red/white): ").lower().strip()
                
                # Validate wine type
                if wine_data['wine_type'] not in ['red', 'white']:
                    print("❌ Invalid wine type. Please enter 'red' or 'white'.")
                    continue
                
                # Make prediction
                print("\n🔮 Making prediction...")
                result = self.predict_single(wine_data)
                
                # Display results
                print("\n" + "="*40)
                print("🎯 PREDICTION RESULTS")
                print("="*40)
                print(f"🍷 Predicted Quality: {result['quality_score']}/10")
                print(f"📊 Category: {result['quality_category']}")
                print(f"🎯 Confidence: ±{result['confidence_interval']:.2f}")
                print(f"🤖 Model: {result['model_used'].replace('_', ' ').title()}")
                print(f"\n💡 Interpretation: {result['interpretation']}")
                print("="*40)
                
                # Continue or exit
                continue_choice = input("\nMake another prediction? (y/n): ").lower().strip()
                if continue_choice != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except ValueError as e:
                print(f"❌ Error: {e}")
                print("Please check your input values and try again.")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    def get_sample_data(self) -> Dict[str, Any]:
        """Get sample wine data for testing."""
        return {
            'fixed_acidity': 7.4,
            'volatile_acidity': 0.7,
            'citric_acid': 0.0,
            'residual_sugar': 1.9,
            'chlorides': 0.076,
            'free_sulfur_dioxide': 11.0,
            'total_sulfur_dioxide': 34.0,
            'density': 0.9978,
            'ph': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4,
            'wine_type': 'red'
        }
    
    def create_sample_file(self, file_path: str = 'sample_wines.csv'):
        """Create a sample CSV file for batch prediction testing."""
        sample_data = [
            self.get_sample_data(),
            {
                'fixed_acidity': 8.1, 'volatile_acidity': 0.88, 'citric_acid': 0.0,
                'residual_sugar': 2.6, 'chlorides': 0.098, 'free_sulfur_dioxide': 25.0,
                'total_sulfur_dioxide': 67.0, 'density': 0.9968, 'ph': 3.20,
                'sulphates': 0.68, 'alcohol': 9.8, 'wine_type': 'red'
            },
            {
                'fixed_acidity': 7.2, 'volatile_acidity': 0.23, 'citric_acid': 0.32,
                'residual_sugar': 8.5, 'chlorides': 0.058, 'free_sulfur_dioxide': 47.0,
                'total_sulfur_dioxide': 186.0, 'density': 0.9956, 'ph': 3.19,
                'sulphates': 0.40, 'alcohol': 9.9, 'wine_type': 'white'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df.to_csv(file_path, index=False)
        print(f"📄 Sample file created: {file_path}")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Wine Quality Predictor - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predictor.py --interactive
  python predictor.py --file wines.csv --output results.csv
  python predictor.py --single --alcohol 12.5 --ph 3.2 --wine_type red
  python predictor.py --sample-file
        """
    )
    
    parser.add_argument('--models-dir', default='models', 
                       help='Directory containing trained models')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--file', help='CSV file for batch prediction')
    parser.add_argument('--output', help='Output file for batch results')
    parser.add_argument('--single', action='store_true',
                       help='Make single prediction from command line args')
    parser.add_argument('--sample-file', action='store_true',
                       help='Create sample CSV file')
    
    # Single prediction arguments
    parser.add_argument('--fixed-acidity', type=float, default=7.4)
    parser.add_argument('--volatile-acidity', type=float, default=0.7)
    parser.add_argument('--citric-acid', type=float, default=0.0)
    parser.add_argument('--residual-sugar', type=float, default=1.9)
    parser.add_argument('--chlorides', type=float, default=0.076)
    parser.add_argument('--free-sulfur-dioxide', type=float, default=11.0)
    parser.add_argument('--total-sulfur-dioxide', type=float, default=34.0)
    parser.add_argument('--density', type=float, default=0.9978)
    parser.add_argument('--ph', type=float, default=3.51)
    parser.add_argument('--sulphates', type=float, default=0.56)
    parser.add_argument('--alcohol', type=float, default=9.4)
    parser.add_argument('--wine-type', choices=['red', 'white'], default='red')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = WineQualityPredictor(args.models_dir)
    
    # Create sample file
    if args.sample_file:
        predictor.create_sample_file()
        return
    
    # Load models
    if not predictor.load_models():
        print("\n❌ Failed to load models. Please ensure:")
        print("   1. Models have been trained (run the training notebook)")
        print("   2. Model files exist in the models/ directory")
        print("   3. File paths are correct")
        return
    
    try:
        # Interactive mode
        if args.interactive:
            predictor.interactive_mode()
            
        # Batch prediction
        elif args.file:
            if not os.path.exists(args.file):
                print(f"❌ File not found: {args.file}")
                return
            
            results = predictor.predict_batch(args.file, args.output)
            
            print(f"\n📊 Batch Prediction Results:")
            print(f"   Total wines: {len(results)}")
            print(f"   Average quality: {results['predicted_quality'].mean():.2f}")
            print(f"   Quality range: {results['predicted_quality'].min():.1f} - {results['predicted_quality'].max():.1f}")
            
            # Show distribution
            quality_dist = results['quality_category'].value_counts()
            print(f"\n   Quality distribution:")
            for category, count in quality_dist.items():
                percentage = (count / len(results)) * 100
                print(f"     {category}: {count} ({percentage:.1f}%)")
            
        # Single prediction
        elif args.single:
            wine_data = {
                'fixed_acidity': args.fixed_acidity,
                'volatile_acidity': args.volatile_acidity,
                'citric_acid': args.citric_acid,
                'residual_sugar': args.residual_sugar,
                'chlorides': args.chlorides,
                'free_sulfur_dioxide': args.free_sulfur_dioxide,
                'total_sulfur_dioxide': args.total_sulfur_dioxide,
                'density': args.density,
                'ph': args.ph,
                'sulphates': args.sulphates,
                'alcohol': args.alcohol,
                'wine_type': args.wine_type
            }
            
            result = predictor.predict_single(wine_data)
            
            print(f"\n🍷 Wine Quality Prediction")
            print("="*40)
            print(f"Input: {args.wine_type.title()} wine")
            print(f"Alcohol: {args.alcohol}%")
            print(f"pH: {args.ph}")
            print(f"Fixed acidity: {args.fixed_acidity}")
            print("-"*40)
            print(f"Predicted Quality: {result['quality_score']}/10")
            print(f"Category: {result['quality_category']}")
            print(f"Confidence: ±{result['confidence_interval']:.2f}")
            print(f"Model: {result['model_used'].replace('_', ' ').title()}")
            print(f"\nInterpretation: {result['interpretation']}")
            print("="*40)
            
        else:
            # Show help if no action specified
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

def validate_models_exist(models_dir: str = 'models') -> bool:
    """
    Check if required model files exist.
    
    Args:
        models_dir (str): Models directory path
        
    Returns:
        bool: True if all required files exist
    """
    required_files = [
        'wine_quality_processor.pkl',
        'trainer_state.pkl',
        'best_model.pkl'
    ]
    
    for file_name in required_files:
        file_path = os.path.join(models_dir, file_name)
        if not os.path.exists(file_path):
            print(f"❌ Missing required file: {file_path}")
            return False
    
    return True

def setup_models():
    """Setup function to train models if they don't exist."""
    if not validate_models_exist():
        print("\n🔧 Models not found. Setting up models...")
        print("This will take a few minutes...\n")
        
        try:
            # Import and run training
            from data_processor import WineQualityProcessor
            from model_trainer import WineQualityModelTrainer
            import os
            
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Initialize and train
            processor = WineQualityProcessor()
            data = processor.load_data()
            X_train, X_val, X_test, y_train, y_val, y_test = processor.fit_transform(data)
            
            trainer = WineQualityModelTrainer()
            trainer.train_all_models(X_train, y_train, X_val, y_val, 
                                   models_to_train=['random_forest', 'ridge'], 
                                   tune_hyperparameters=False)
            
            # Save models
            processor.save_processor('models/wine_quality_processor.pkl')
            trainer.save_models('models')
            
            print("✅ Models trained and saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up models: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🍷 Wine Quality Predictor v1.0")
    print("="*40)
    
    # Check if models exist, train if necessary
    if not validate_models_exist():
        setup_choice = input("Models not found. Train new models? (y/n): ").lower().strip()
        if setup_choice == 'y':
            if not setup_models():
                print("❌ Setup failed. Exiting.")
                sys.exit(1)
        else:
            print("❌ Models required to run predictor. Exiting.")
            sys.exit(1)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)

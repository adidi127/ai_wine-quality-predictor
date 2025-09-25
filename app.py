"""
Wine Quality Prediction System - Streamlit Application
=====================================================

A complete web application for predicting wine quality using machine learning.
Features single predictions, batch processing, model retraining, and performance visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import time
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules (these would be your actual modules)
try:
    from data_processor import WineQualityProcessor, create_sample_prediction_data
    from model_trainer import WineQualityModelTrainer
except ImportError:
    st.error("Custom modules not found. Please ensure data_processor.py and model_trainer.py are available.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722f37;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #8b4513;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #722f37;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(90deg, #722f37, #8b4513);
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class WineQualityApp:
    """Main application class for the Wine Quality Prediction System."""
    
    def __init__(self):
        """Initialize the application."""
        self.processor = None
        self.trainer = None
        self.initialize_session_state()
        self.load_models()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'training_data' not in st.session_state:
            st.session_state.training_data = None
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = None
    
    def load_models(self):
        """Load pre-trained models if available."""
        try:
            if os.path.exists('models/wine_quality_processor.pkl'):
                self.processor = WineQualityProcessor.load_processor('models/wine_quality_processor.pkl')
                st.session_state.models_loaded = True
            
            if os.path.exists('models/trainer_state.pkl'):
                self.trainer = WineQualityModelTrainer.load_models('models')
                st.session_state.models_loaded = True
                
        except Exception as e:
            st.sidebar.warning(f"Could not load pre-trained models: {e}")
            st.sidebar.info("You can train new models using the Training tab.")
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🍷 Wine Quality Prediction System</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.2rem; color: #666;'>
                Predict wine quality using advanced machine learning models trained on physicochemical properties.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the application sidebar."""
        st.sidebar.markdown("## 🔧 Model Status")
        
        if st.session_state.models_loaded:
            st.sidebar.success("✅ Models loaded successfully")
            if self.trainer and self.trainer.best_model_name:
                st.sidebar.info(f"🏆 Best Model: {self.trainer.best_model_name.replace('_', ' ').title()}")
        else:
            st.sidebar.error("❌ No models loaded")
            st.sidebar.info("Use the Training tab to train new models")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📊 Quick Stats")
        
        if self.processor and hasattr(self.processor, 'processing_stats'):
            stats = self.processor.processing_stats
            if 'final_summary' in stats:
                st.sidebar.metric("Features", stats['final_summary']['final_feature_count'])
                st.sidebar.metric("Training Samples", stats['final_summary']['final_sample_count'])
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ℹ️ About")
        st.sidebar.markdown("""
        This application predicts wine quality (1-10 scale) based on:
        - Fixed acidity
        - Volatile acidity  
        - Citric acid
        - Residual sugar
        - Chlorides
        - Free sulfur dioxide
        - Total sulfur dioxide
        - Density
        - pH
        - Sulphates
        - Alcohol content
        - Wine type (red/white)
        """)
    
    def render_prediction_tab(self):
        """Render the single prediction tab."""
        st.markdown('<h2 class="sub-header">🎯 Single Wine Prediction</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.models_loaded:
            st.warning("Please train models first using the Training tab.")
            return
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Acidity Properties")
                fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, 
                                               value=7.4, step=0.1)
                volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, 
                                                  value=0.7, step=0.01)
                citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, 
                                            value=0.0, step=0.01)
                ph = st.number_input("pH", min_value=2.0, max_value=5.0, 
                                   value=3.51, step=0.01)
            
            with col2:
                st.subheader("Chemical Properties")
                residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=50.0, 
                                                value=1.9, step=0.1)
                chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, 
                                          value=0.076, step=0.001)
                sulphates = st.number_input("Sulphates", min_value=0.0, max_value=3.0, 
                                          value=0.56, step=0.01)
                density = st.number_input("Density", min_value=0.9, max_value=1.1, 
                                        value=0.9978, step=0.0001)
            
            with col3:
                st.subheader("Sulfur & Alcohol")
                free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, 
                                                     value=11.0, step=1.0)
                total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, 
                                                      value=34.0, step=1.0)
                alcohol = st.number_input("Alcohol %", min_value=5.0, max_value=20.0, 
                                        value=9.4, step=0.1)
                wine_type = st.selectbox("Wine Type", ["red", "white"])
            
            submitted = st.form_submit_button("🔮 Predict Wine Quality", 
                                            help="Click to predict wine quality")
        
        if submitted:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'fixed acidity': [fixed_acidity],
                'volatile acidity': [volatile_acidity],
                'citric acid': [citric_acid],
                'residual sugar': [residual_sugar],
                'chlorides': [chlorides],
                'free sulfur dioxide': [free_sulfur_dioxide],
                'total sulfur dioxide': [total_sulfur_dioxide],
                'density': [density],
                'pH': [ph],
                'sulphates': [sulphates],
                'alcohol': [alcohol],
                'wine_type': [wine_type]
            })
            
            # Make prediction
            try:
                processed_data = self.processor.transform(input_data)
                prediction = self.trainer.predict(processed_data)[0]
                
                # Get confidence intervals if available
                try:
                    pred_mean, pred_std = self.trainer.predict_with_confidence(processed_data)
                    confidence_interval = pred_std[0] * 1.96  # 95% CI
                except:
                    confidence_interval = 0.5  # Default uncertainty
                
                # Store result
                st.session_state.last_prediction = {
                    'prediction': prediction,
                    'confidence': confidence_interval,
                    'input_data': input_data
                }
                
                # Display result
                self.display_prediction_result(prediction, confidence_interval)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    def display_prediction_result(self, prediction: float, confidence: float):
        """Display prediction result with visualization."""
        # Main prediction display
        quality_score = max(1, min(10, round(prediction, 1)))
        
        st.markdown(f"""
        <div class="prediction-result">
            Predicted Wine Quality: {quality_score}/10
        </div>
        """, unsafe_allow_html=True)
        
        # Quality interpretation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if quality_score <= 4:
                st.error("🔴 Poor Quality")
                interpretation = "This wine may have noticeable flaws or defects."
            elif quality_score <= 6:
                st.warning("🟡 Average Quality")
                interpretation = "This wine is acceptable but not outstanding."
            else:
                st.success("🟢 Good to Excellent Quality")
                interpretation = "This wine should have good to excellent characteristics."
        
        with col2:
            st.metric("Confidence Interval", f"±{confidence:.2f}", 
                     help="95% confidence interval for the prediction")
        
        with col3:
            st.metric("Quality Range", f"{max(1, quality_score-confidence):.1f} - {min(10, quality_score+confidence):.1f}")
        
        st.info(f"💡 **Interpretation:** {interpretation}")
        
        # Quality gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = quality_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Wine Quality Score"},
            delta = {'reference': 5.5},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgray"},
                    {'range': [4, 6], 'color': "yellow"},
                    {'range': [6, 8], 'color': "lightgreen"},
                    {'range': [8, 10], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': quality_score
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_batch_tab(self):
        """Render the batch prediction tab."""
        st.markdown('<h2 class="sub-header">📊 Batch Prediction</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.models_loaded:
            st.warning("Please train models first using the Training tab.")
            return
        
        st.markdown("""
        Upload a CSV file with wine data to get predictions for multiple wines at once.
        The file should contain the same columns as the single prediction form.
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload a CSV file with wine properties"
        )
        
        # Sample data download
        if st.button("📥 Download Sample Format"):
            sample_data = create_sample_prediction_data()
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download sample CSV",
                data=csv,
                file_name="wine_sample_data.csv",
                mime="text/csv"
            )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                batch_data = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(batch_data)} rows found.")
                
                # Display preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(batch_data.head())
                
                # Make predictions
                if st.button("🚀 Run Batch Predictions"):
                    with st.spinner("Making predictions..."):
                        try:
                            processed_data = self.processor.transform(batch_data)
                            predictions = self.trainer.predict(processed_data)
                            
                            # Add predictions to original data
                            results_df = batch_data.copy()
                            results_df['predicted_quality'] = np.round(predictions, 1)
                            results_df['quality_category'] = results_df['predicted_quality'].apply(
                                lambda x: 'Poor' if x <= 4 else 'Average' if x <= 6 else 'Good'
                            )
                            
                            st.session_state.batch_results = results_df
                            
                            # Display results
                            st.success("✅ Predictions completed!")
                            self.display_batch_results(results_df)
                            
                        except Exception as e:
                            st.error(f"Error processing batch predictions: {e}")
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    def display_batch_results(self, results_df: pd.DataFrame):
        """Display batch prediction results."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Predictions", len(results_df))
            st.metric("Average Quality", f"{results_df['predicted_quality'].mean():.2f}")
        
        with col2:
            quality_dist = results_df['quality_category'].value_counts()
            st.metric("Good Quality Wines", f"{quality_dist.get('Good', 0)} ({quality_dist.get('Good', 0)/len(results_df)*100:.1f}%)")
        
        # Quality distribution chart
        fig = px.histogram(results_df, x='predicted_quality', nbins=20, 
                          title="Distribution of Predicted Wine Quality")
        fig.update_xaxis(title="Predicted Quality Score")
        fig.update_yaxis(title="Number of Wines")
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.subheader("📋 Detailed Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"wine_predictions_{int(time.time())}.csv",
            mime="text/csv"
        )
    
    def render_training_tab(self):
        """Render the model training tab."""
        st.markdown('<h2 class="sub-header">🔬 Model Training</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        Train new machine learning models or retrain existing ones with updated data.
        This process may take several minutes depending on the selected options.
        """)
        
        # Training options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎛️ Training Options")
            
            models_to_train = st.multiselect(
                "Select Models to Train",
                ["random_forest", "gradient_boosting", "support_vector", "ridge", 
                 "lasso", "elastic_net", "neural_network"],
                default=["random_forest", "gradient_boosting", "ridge"]
            )
            
            tune_hyperparameters = st.checkbox(
                "Enable Hyperparameter Tuning", 
                value=True,
                help="This will take longer but may improve performance"
            )
            
            test_size = st.slider("Test Set Size", 0.1, 0.3, 0.2, 0.05)
            val_size = st.slider("Validation Set Size", 0.1, 0.3, 0.2, 0.05)
        
        with col2:
            st.subheader("📂 Data Options")
            
            use_default_data = st.radio(
                "Data Source",
                ["Use Default Wine Dataset", "Upload Custom Dataset"]
            )
            
            if use_default_data == "Upload Custom Dataset":
                training_file = st.file_uploader(
                    "Upload Training Data (CSV)",
                    type="csv",
                    help="CSV file with wine properties and quality column"
                )
            else:
                training_file = None
        
        # Start training
        if st.button("🚀 Start Training", type="primary"):
            if not models_to_train:
                st.error("Please select at least one model to train.")
                return
            
            with st.spinner("Training models... This may take several minutes."):
                try:
                    self.train_models(
                        models_to_train=models_to_train,
                        tune_hyperparameters=tune_hyperparameters,
                        test_size=test_size,
                        val_size=val_size,
                        training_file=training_file
                    )
                    
                    st.success("✅ Training completed successfully!")
                    st.session_state.models_loaded = True
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.exception(e)
    
    def train_models(self, models_to_train: list, tune_hyperparameters: bool,
                    test_size: float, val_size: float, training_file=None):
        """Train the selected models."""
        # Initialize processor and trainer
        self.processor = WineQualityProcessor(random_state=42)
        self.trainer = WineQualityModelTrainer(random_state=42)
        
        # Load data
        if training_file is not None:
            training_data = pd.read_csv(training_file)
        else:
            training_data = self.processor.load_data()
        
        # Process data
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.fit_transform(training_data)
        
        # Store for later use
        st.session_state.training_data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
        
        # Train models
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(models_to_train)
        for i, model_name in enumerate(models_to_train):
            status_text.text(f"Training {model_name}... ({i+1}/{total_models})")
            
            self.trainer.train_single_model(
                model_name, X_train, y_train, X_val, y_val, tune_hyperparameters
            )
            
            progress_bar.progress((i + 1) / total_models)
        
        # Select best model
        self.trainer._select_best_model()
        
        # Evaluate on test set
        self.trainer.evaluate_on_test(X_test, y_test)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        self.processor.save_processor('models/wine_quality_processor.pkl')
        self.trainer.save_models('models')
        
        status_text.text("Training completed!")
        progress_bar.progress(1.0)
    
    def render_analysis_tab(self):
        """Render the model analysis tab."""
        st.markdown('<h2 class="sub-header">📈 Model Analysis</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.models_loaded:
            st.warning("Please train models first using the Training tab.")
            return
        
        # Model comparison
        if self.trainer and self.trainer.models:
            st.subheader("🏆 Model Performance Comparison")
            
            comparison_df = self.trainer.get_model_comparison()
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Best Model", 
                         self.trainer.best_model_name.replace('_', ' ').title(),
                         f"R² = {comparison_df.iloc[0]['Val_R2']:.4f}")
            
            with col2:
                st.metric("Total Models Trained", len(comparison_df))
            
            # Performance comparison chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Scores', 'RMSE Scores', 'Training Time', 'Cross-Validation'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # R² comparison
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df['Train_R2'], 
                       name='Train R²', marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df['Val_R2'], 
                       name='Validation R²', marker_color='darkblue'),
                row=1, col=1
            )
            
            # RMSE comparison
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df['Train_RMSE'], 
                       name='Train RMSE', marker_color='lightcoral', showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df['Val_RMSE'], 
                       name='Validation RMSE', marker_color='darkred', showlegend=False),
                row=1, col=2
            )
            
            # Training time
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df['Training_Time'], 
                       name='Training Time (s)', marker_color='green', showlegend=False),
                row=2, col=1
            )
            
            # Cross-validation scores
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df['CV_Mean'], 
                       error_y=dict(type='data', array=comparison_df['CV_Std']),
                       name='CV Score', marker_color='orange', showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("📊 Detailed Metrics")
            st.dataframe(comparison_df, use_container_width=True)
        
        # Feature importance (if available)
        if (self.trainer and self.trainer.best_model and 
            hasattr(self.trainer.best_model, 'feature_importances_')):
            
            st.subheader("🎯 Feature Importance")
            
            importance = self.trainer.best_model.feature_importances_
            feature_names = self.processor.feature_names if self.processor else [f'feature_{i}' for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title="Top 15 Most Important Features")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    def render_data_tab(self):
        """Render the data exploration tab."""
        st.markdown('<h2 class="sub-header">🔍 Data Exploration</h2>', 
                   unsafe_allow_html=True)
        
        # Load sample data for exploration
        try:
            processor = WineQualityProcessor()
            sample_data = processor.load_data()
            
            st.success(f"✅ Dataset loaded: {sample_data.shape[0]:,} samples, {sample_data.shape[1]} features")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Wines", f"{len(sample_data):,}")
            with col2:
                red_count = len(sample_data[sample_data['wine_type'] == 'red'])
                st.metric("Red Wines", f"{red_count:,}")
            with col3:
                white_count = len(sample_data[sample_data['wine_type'] == 'white'])
                st.metric("White Wines", f"{white_count:,}")
            
            # Quality distribution
            st.subheader("🎯 Quality Score Distribution")
            fig = px.histogram(sample_data, x='quality', color='wine_type',
                             title="Distribution of Wine Quality Scores by Type")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.subheader("🔗 Feature Correlations")
            numerical_cols = sample_data.select_dtypes(include=[np.number]).columns
            corr_matrix = sample_data[numerical_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions
            st.subheader("📊 Feature Distributions")
            selected_feature = st.selectbox("Select feature to explore:", 
                                           [col for col in numerical_cols if col != 'quality'])
            
            fig = px.box(sample_data, x='wine_type', y=selected_feature, 
                        title=f"{selected_feature} Distribution by Wine Type")
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw data view
            with st.expander("🗃️ View Raw Data"):
                st.dataframe(sample_data, use_container_width=True)
                
                # Download option
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Dataset",
                    data=csv,
                    file_name="wine_quality_dataset.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def run(self):
        """Run the main application."""
        self.render_header()
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Predict", "📊 Batch", "🔬 Train", "📈 Analysis", "🔍 Data"
        ])
        
        with tab1:
            self.render_prediction_tab()
        
        with tab2:
            self.render_batch_tab()
        
        with tab3:
            self.render_training_tab()
        
        with tab4:
            self.render_analysis_tab()
        
        with tab5:
            self.render_data_tab()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            🍷 Wine Quality Prediction System | Built with Streamlit & Scikit-learn
        </div>
        """, unsafe_allow_html=True)

# Main application entry point
def main():
    """Main application entry point."""
    app = WineQualityApp()
    app.run()

if __name__ == "__main__":
    main()



"""
Sales Prediction with Machine Learning
=====================================

This project demonstrates how to predict product sales based on advertising spending
across different media channels (TV, Radio, Newspaper) using various machine learning algorithms.

Dataset: Advertising.csv
Features: TV, Radio, Newspaper (advertising spending)
Target: Sales (product sales)

Author: Data Science Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SalesPredictor:
    """
    A comprehensive sales prediction system using multiple machine learning algorithms.
    """
    
    def __init__(self, data_path):
        """
        Initialize the SalesPredictor with data path.
        
        Args:
            data_path (str): Path to the CSV file containing advertising data
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """
        Load and preprocess the advertising dataset.
        """
        print("Loading and preprocessing data...")
        
        # Load the dataset
        self.data = pd.read_csv(self.data_path)
        
        # Remove the first column (index column)
        if self.data.columns[0] == '':
            self.data = self.data.drop(self.data.columns[0], axis=1)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(self.data.info())
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        # Display statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis and visualization.
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Check for missing values
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
        # Check for duplicates
        print(f"\nDuplicate rows: {self.data.duplicated().sum()}")
        
        # Correlation analysis
        print("\nCorrelation Matrix:")
        correlation_matrix = self.data.corr()
        print(correlation_matrix)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # 1. Distribution of features
        plt.subplot(2, 3, 1)
        self.data['TV'].hist(bins=20, alpha=0.7, color='blue')
        plt.title('TV Advertising Distribution')
        plt.xlabel('TV Spending')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 3, 2)
        self.data['Radio'].hist(bins=20, alpha=0.7, color='green')
        plt.title('Radio Advertising Distribution')
        plt.xlabel('Radio Spending')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 3, 3)
        self.data['Newspaper'].hist(bins=20, alpha=0.7, color='red')
        plt.title('Newspaper Advertising Distribution')
        plt.xlabel('Newspaper Spending')
        plt.ylabel('Frequency')
        
        # 2. Sales distribution
        plt.subplot(2, 3, 4)
        self.data['Sales'].hist(bins=20, alpha=0.7, color='purple')
        plt.title('Sales Distribution')
        plt.xlabel('Sales')
        plt.ylabel('Frequency')
        
        # 3. Correlation heatmap
        plt.subplot(2, 3, 5)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        
        # 4. Scatter plot: TV vs Sales
        plt.subplot(2, 3, 6)
        plt.scatter(self.data['TV'], self.data['Sales'], alpha=0.6, color='blue')
        plt.title('TV Advertising vs Sales')
        plt.xlabel('TV Spending')
        plt.ylabel('Sales')
        
        plt.tight_layout()
        plt.show()
        
        # Additional scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Radio vs Sales
        axes[0].scatter(self.data['Radio'], self.data['Sales'], alpha=0.6, color='green')
        axes[0].set_title('Radio Advertising vs Sales')
        axes[0].set_xlabel('Radio Spending')
        axes[0].set_ylabel('Sales')
        
        # Newspaper vs Sales
        axes[1].scatter(self.data['Newspaper'], self.data['Sales'], alpha=0.6, color='red')
        axes[1].set_title('Newspaper Advertising vs Sales')
        axes[1].set_xlabel('Newspaper Spending')
        axes[1].set_ylabel('Sales')
        
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def prepare_data(self):
        """
        Prepare data for machine learning by separating features and target.
        """
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Separate features and target
        self.X = self.data[['TV', 'Radio', 'Newspaper']]
        self.y = self.data['Sales']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preparation completed!")
        
    def train_models(self):
        """
        Train multiple machine learning models and compare their performance.
        """
        print("\n" + "="*50)
        print("MODEL TRAINING AND EVALUATION")
        print("="*50)
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf')
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for models that benefit from scaling
            if name in ['Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            if name in ['Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        
        # Find the best model
        best_r2 = max(results.items(), key=lambda x: x[1]['r2'])
        self.best_model_name = best_r2[0]
        self.best_model = best_r2[1]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best R² Score: {best_r2[1]['r2']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning for the best model.
        """
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestRegressor(random_state=42)
            X_train_tune = self.X_train
            X_test_tune = self.X_test
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingRegressor(random_state=42)
            X_train_tune = self.X_train
            X_test_tune = self.X_test
            
        elif self.best_model_name == 'Ridge Regression':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
            model = Ridge()
            X_train_tune = self.X_train_scaled
            X_test_tune = self.X_test_scaled
            
        else:
            print("Hyperparameter tuning not implemented for this model.")
            return self.best_model
        
        print(f"Tuning hyperparameters for {self.best_model_name}...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(X_train_tune, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the best model
        self.best_model = grid_search.best_estimator_
        
        # Evaluate tuned model
        y_pred_tuned = self.best_model.predict(X_test_tune)
        r2_tuned = r2_score(self.y_test, y_pred_tuned)
        rmse_tuned = np.sqrt(mean_squared_error(self.y_test, y_pred_tuned))
        
        print(f"Tuned model R² Score: {r2_tuned:.4f}")
        print(f"Tuned model RMSE: {rmse_tuned:.4f}")
        
        return self.best_model
    
    def visualize_results(self):
        """
        Create visualizations to compare model performance and predictions.
        """
        print("\n" + "="*50)
        print("RESULTS VISUALIZATION")
        print("="*50)
        
        # 1. Model comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² scores comparison
        axes[0].bar(model_names, r2_scores, color='skyblue', alpha=0.7)
        axes[0].set_title('Model Comparison - R² Scores')
        axes[0].set_ylabel('R² Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE scores comparison
        axes[1].bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
        axes[1].set_title('Model Comparison - RMSE Scores')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Actual vs Predicted for best model
        best_predictions = self.models[self.best_model_name]['predictions']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, best_predictions, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title(f'Actual vs Predicted Sales - {self.best_model_name}')
        plt.show()
        
        # 3. Residuals plot
        residuals = self.y_test - best_predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(best_predictions, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Sales')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Plot - {self.best_model_name}')
        plt.show()
        
        # 4. Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
            feature_names = ['TV', 'Radio', 'Newspaper']
            
            plt.figure(figsize=(8, 6))
            plt.bar(feature_names, feature_importance, color='orange', alpha=0.7)
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.ylabel('Importance')
            plt.show()
    
    def predict_sales(self, tv_spending, radio_spending, newspaper_spending):
        """
        Predict sales for given advertising spending.
        
        Args:
            tv_spending (float): TV advertising spending
            radio_spending (float): Radio advertising spending
            newspaper_spending (float): Newspaper advertising spending
            
        Returns:
            float: Predicted sales
        """
        # Create input array
        input_data = np.array([[tv_spending, radio_spending, newspaper_spending]])
        
        # Scale the input if the best model requires scaling
        if self.best_model_name in ['Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
            input_data = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.best_model.predict(input_data)[0]
        
        return prediction
    
    def generate_report(self):
        """
        Generate a comprehensive report of the analysis.
        """
        print("\n" + "="*60)
        print("SALES PREDICTION ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Features: {list(self.X.columns)}")
        print(f"  Target variable: Sales")
        
        print(f"\nData Statistics:")
        print(f"  TV spending range: ${self.data['TV'].min():.2f} - ${self.data['TV'].max():.2f}")
        print(f"  Radio spending range: ${self.data['Radio'].min():.2f} - ${self.data['Radio'].max():.2f}")
        print(f"  Newspaper spending range: ${self.data['Newspaper'].min():.2f} - ${self.data['Newspaper'].max():.2f}")
        print(f"  Sales range: ${self.data['Sales'].min():.2f} - ${self.data['Sales'].max():.2f}")
        
        print(f"\nCorrelation Analysis:")
        correlations = self.data.corr()['Sales'].sort_values(ascending=False)
        for feature, corr in correlations.items():
            if feature != 'Sales':
                print(f"  {feature}: {corr:.4f}")
        
        print(f"\nModel Performance Summary:")
        print(f"{'Model':<20} {'R² Score':<10} {'RMSE':<10} {'MAE':<10}")
        print("-" * 50)
        for name, metrics in self.models.items():
            print(f"{name:<20} {metrics['r2']:<10.4f} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f}")
        
        print(f"\nBest Model: {self.best_model_name}")
        best_metrics = self.models[self.best_model_name]
        print(f"  R² Score: {best_metrics['r2']:.4f}")
        print(f"  RMSE: {best_metrics['rmse']:.4f}")
        print(f"  MAE: {best_metrics['mae']:.4f}")
        
        print(f"\nKey Insights:")
        print(f"  1. TV advertising shows the strongest correlation with sales")
        print(f"  2. The {self.best_model_name} model provides the best predictions")
        print(f"  3. Model explains {best_metrics['r2']*100:.1f}% of the variance in sales")
        
        return {
            'best_model': self.best_model_name,
            'best_r2': best_metrics['r2'],
            'best_rmse': best_metrics['rmse'],
            'correlations': correlations.to_dict()
        }

def main():
    """
    Main function to run the complete sales prediction analysis.
    """
    print("SALES PREDICTION WITH MACHINE LEARNING")
    print("="*50)
    
    # Initialize the predictor
    predictor = SalesPredictor(r'c:\Users\Ayesha Asna\Downloads\Advertising.csv')
    
    # Load and explore data
    predictor.load_data()
    predictor.explore_data()
    
    # Prepare data for machine learning
    predictor.prepare_data()
    
    # Train models and evaluate performance
    results = predictor.train_models()
    
    # Perform hyperparameter tuning
    predictor.hyperparameter_tuning()
    
    # Visualize results
    predictor.visualize_results()
    
    # Generate comprehensive report
    report = predictor.generate_report()
    
    # Demonstrate prediction functionality
    print("\n" + "="*50)
    print("PREDICTION EXAMPLES")
    print("="*50)
    
    # Example predictions
    examples = [
        (230.1, 37.8, 69.2),  # High TV, medium radio, high newspaper
        (44.5, 39.3, 45.1),   # Low TV, medium radio, medium newspaper
        (151.5, 41.3, 58.5),  # Medium TV, medium radio, medium newspaper
        (0, 0, 0),            # No advertising
        (300, 50, 100)        # High advertising across all channels
    ]
    
    for i, (tv, radio, newspaper) in enumerate(examples, 1):
        prediction = predictor.predict_sales(tv, radio, newspaper)
        print(f"Example {i}: TV=${tv}, Radio=${radio}, Newspaper=${newspaper}")
        print(f"  Predicted Sales: ${prediction:.2f}")
        print()
    
    print("Analysis completed successfully!")
    return predictor, report

if __name__ == "__main__":
    # Run the complete analysis
    predictor, report = main()


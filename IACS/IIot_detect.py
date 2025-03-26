# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:05:48 2025

@author: Sanjay
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import ipaddress
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from datetime import datetime, timedelta

# Load the dataset (assuming you have a CSV file)
# Replace 'iot_dataset.csv' with your actual file path
def load_and_preprocess_data(file_path):
    print("Loading dataset...")
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Could not read the file with any of the supported encodings")
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

# Feature engineering based on domain knowledge
def feature_engineering(df):
    print("\nPerforming feature engineering...")
    
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # 1. Check if source and destination are in same subnet
    def same_subnet(row):
        try:
            src = ipaddress.ip_address(row['SrcAddr'])
            dst = ipaddress.ip_address(row['DstAddr'])
            # Check if they share the first 24 bits (for IPv4) or 64 bits (for IPv6)
            if isinstance(src, ipaddress.IPv4Address):
                src_net = ipaddress.IPv4Network(f"{src}/24", strict=False)
                return 1 if dst in src_net else 0
            else:
                src_net = ipaddress.IPv6Network(f"{src}/64", strict=False)
                return 1 if dst in src_net else 0
        except:
            return -1  # Error handling
    
    # Apply the function to create a new feature
    df_processed['SameSubnet'] = df.apply(same_subnet, axis=1)
    
    # 2. Create feature for DPort != 80
    df_processed['DPortNot80'] = (df_processed['Dport'] != 80).astype(int)
    
    # 3. Create features for DAppBytes != 0 and dTtl != 0
    df_processed['DAppBytesNonZero'] = (df_processed['DAppBytes'] != 0).astype(int)
    df_processed['dTtlNonZero'] = (df_processed['dTtl'] != 0).astype(int)
    
    # 4. Create a combined feature representing all normal conditions
    df_processed['AllNormalConditions'] = ((df_processed['SameSubnet'] == 1) & 
                                          (df_processed['DPortNot80'] == 1) & 
                                          (df_processed['DAppBytesNonZero'] == 1) & 
                                          (df_processed['dTtlNonZero'] == 1)).astype(int)
    
    # 5. Calculate ratios that might be useful
    df_processed['PktRatio'] = df_processed['SrcPkts'] / df_processed['TotPkts'].replace(0, 1)
    df_processed['ByteRatio'] = df_processed['SrcBytes'] / df_processed['TotBytes'].replace(0, 1)
    
    # 6. Calculate statistical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[f'{col}_log'] = np.log1p(df_processed[col].replace(0, 1e-5))
    
    print(f"Engineered dataset shape: {df_processed.shape}")
    return df_processed

# Feature selection
def select_features(X, y):
    print("\nPerforming feature selection...")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])
    
    # Create a RandomForest feature selector
    rf_selector = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    rf_selector.fit(X, y)
    
    # Get feature importances
    feature_importances = rf_selector.named_steps['classifier'].feature_importances_
    
    # Get preprocessing feature names
    ohe_feature_names = []
    if len(categorical_cols) > 0:
        ohe = rf_selector.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    
    all_feature_names = list(numerical_cols) + list(ohe_feature_names)
    
    # Create a DataFrame with feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names[:len(feature_importances)],  # Ensure lengths match
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 important features:")
    print(feature_importance_df.head(20))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    
    # Select the top features
    # Here we select features with importance > mean importance
    threshold = np.mean(feature_importances)
    selector = SelectFromModel(rf_selector.named_steps['classifier'], threshold=threshold)
    selector.fit(preprocessor.transform(X), y)
    
    # Create the final pipeline with selected features
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return final_pipeline, feature_importance_df

# Train and evaluate model
def train_and_evaluate(X, y, pipeline):
    print("\nTraining and evaluating model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return pipeline

# New function to predict on new data and export to Excel
def predict_and_export(model, dataset_path):
    print(f"\nPredicting traffic types for dataset: {dataset_path}")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        new_data = None
        
        for encoding in encodings:
            try:
                new_data = pd.read_csv(dataset_path, encoding=encoding)
                print(f"Successfully loaded file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if new_data is None:
            raise ValueError("Could not read the file with any of the supported encodings")
            
        print(f"Loaded new dataset with shape: {new_data.shape}")
        
        # Apply the same feature engineering
        processed_data = feature_engineering(new_data)
        
        # If the target column exists in the new data, save it for comparison
        target_column_exists = "Traffic" in processed_data.columns
        if target_column_exists:
            actual_traffic = processed_data["Traffic"].copy()
            X_new = processed_data.drop(["Traffic"], axis=1, errors='ignore')
        else:
            X_new = processed_data
        
        # Make predictions
        predictions = model.predict(X_new)
        
        # Create a new DataFrame with original data and predictions
        result_df = new_data.copy()
        result_df['Predicted_Traffic'] = predictions
        
        # If target column exists, add a comparison column
        if target_column_exists:
            result_df['Actual_Traffic'] = actual_traffic
            result_df['Prediction_Correct'] = (result_df['Predicted_Traffic'] == result_df['Actual_Traffic'])
            
            # Calculate accuracy
            accuracy = (result_df['Prediction_Correct'].sum() / len(result_df)) * 100
            print(f"Prediction accuracy on new data: {accuracy:.2f}%")
        
        # Generate output filename based on input filename
        input_filename = os.path.basename(dataset_path)
        output_filename = f"prediction_results_{input_filename.split('.')[0]}.xlsx"
        
        # Export to Excel
        result_df.to_excel(output_filename, index=False)
        print(f"Predictions exported to: {output_filename}")
        
        # Create a summary sheet with prediction distribution
        traffic_counts = result_df['Predicted_Traffic'].value_counts()
        
        # Create a pie chart of predictions
        plt.figure(figsize=(10, 8))
        plt.pie(traffic_counts, labels=traffic_counts.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Distribution of Predicted Traffic Types')
        plt.savefig('prediction_distribution.png')
        
        return result_df
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

# Function to get file path using CLI
def get_file_path(prompt="Enter the path to your CSV file: "):
    while True:
        file_path = input(prompt).strip()
        if file_path.lower() == 'exit':
            return None
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            return file_path
        print("Invalid file path or not a CSV file. Please try again or type 'exit' to quit.")

def create_test_dataset(original_data, num_samples=1000):
    """Create test dataset by sampling random logs from the original dataset"""
    print("Creating test dataset from original data...")
    
    # Create a copy of the original data
    df = original_data.copy()
    
    # Remove target columns if they exist
    target_columns = ['Traffic', 'target']
    df = df.drop(columns=[col for col in target_columns if col in df.columns], errors='ignore')
    
    # Sample random rows
    if len(df) > num_samples:
        test_data = df.sample(n=num_samples, random_state=42)
    else:
        print(f"Warning: Original dataset has fewer samples ({len(df)}) than requested ({num_samples}). Using all samples.")
        test_data = df
    
    print(f"Created test dataset with {len(test_data)} samples")
    return test_data

# Main function to run the entire process
def main(file_path=None, target_column="Traffic"):
    # If no file path provided, ask user to input
    if not file_path:
        print("Please provide the path to your training dataset file...")
        file_path = get_file_path()
        if not file_path:
            print("No file path provided. Exiting.")
            return None, None
    
    # Load and explore data
    df = load_and_preprocess_data(file_path)
    
    # Engineer features
    df_processed = feature_engineering(df)
    
    # Define features and target
    X = df_processed.drop([target_column], axis=1, errors='ignore')
    y = df_processed[target_column]
    
    # Feature selection and model training
    final_pipeline, feature_importance_df = select_features(X, y)
    
    # Train and evaluate model
    trained_model = train_and_evaluate(X, y, final_pipeline)
    
    # Save model and important features
    feature_importance_df.to_csv('feature_importances.csv', index=False)
    
    # Create test dataset from original data
    print("\nCreating test dataset from original data...")
    test_data = create_test_dataset(df, num_samples=1000)
    test_data.to_csv('test_data.csv', index=False)
    print("Test data saved to test_data.csv")
    
    # Run predictions on the test dataset
    predict_and_export(trained_model, 'test_data.csv')
    
    print("\nProcess completed successfully!")
    return trained_model, feature_importance_df

# Example usage
if __name__ == "__main__":
    file_path = None
    target_column = "Traffic"  
    
    trained_model, feature_importances = main(file_path, target_column)
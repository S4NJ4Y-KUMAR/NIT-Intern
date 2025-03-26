import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_scada_network_data(dataset_path):
    """
    Preprocess SCADA network traffic data with specific handling for each feature type
    """
    print(f"Loading dataset from {dataset_path}...")
    data = pd.read_csv(dataset_path)
    
    print(f"Dataset shape: {data.shape}")
    
    # Create a copy to avoid modifying the original
    processed_data = data.copy()
    
    # 1. Process timestamp columns (StartTime, LastTime)
    time_columns = ['StartTime', 'LastTime']
    for col in time_columns:
        if col in processed_data.columns:
            # Try to convert to datetime
            try:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                # Extract time features
                processed_data[f'{col}_hour'] = processed_data[col].dt.hour
                processed_data[f'{col}_minute'] = processed_data[col].dt.minute
                processed_data[f'{col}_second'] = processed_data[col].dt.second
                processed_data[f'{col}_dayofweek'] = processed_data[col].dt.dayofweek
                # Drop original column
                processed_data.drop(col, axis=1, inplace=True)
            except Exception as e:
                print(f"Error processing timestamp column {col}: {e}")
                processed_data.drop(col, axis=1, inplace=True)
    
    # 2. Process IP address columns (SrcAddr, DstAddr)
    ip_columns = ['SrcAddr', 'DstAddr']
    for col in ip_columns:
        if col in processed_data.columns:
            # Extract features from IP addresses
            try:
                # Check if it contains valid IP addresses
                if processed_data[col].astype(str).str.contains(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}').any():
                    # Extract first octet as network class indicator
                    processed_data[f'{col}_first_octet'] = processed_data[col].astype(str).apply(
                        lambda x: int(x.split('.')[0]) if '.' in str(x) and x.split('.')[0].isdigit() else 0
                    )
                    # Is private IP (simplified check for common private ranges)
                    processed_data[f'{col}_is_private'] = processed_data[col].astype(str).apply(
                        lambda x: 1 if x.startswith(('10.', '192.168.', '172.16.', '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.')) else 0
                    )
                # Drop original column
                processed_data.drop(col, axis=1, inplace=True)
            except Exception as e:
                print(f"Error processing IP column {col}: {e}")
                processed_data.drop(col, axis=1, inplace=True)
    
    # 3. Handle protocol column (Proto) - categorical
    if 'Proto' in processed_data.columns:
        try:
            # One-hot encode protocol
            dummies = pd.get_dummies(processed_data['Proto'], prefix='Proto', drop_first=True)
            processed_data = pd.concat([processed_data, dummies], axis=1)
            processed_data.drop('Proto', axis=1, inplace=True)
        except Exception as e:
            print(f"Error processing Proto column: {e}")
            processed_data.drop('Proto', axis=1, inplace=True)
    
    # 4. Process port numbers (Sport, Dport) - keep as numeric
    # These are already numeric, but ensure they're properly typed
    for col in ['Sport', 'Dport']:
        if col in processed_data.columns:
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                # Create feature for well-known ports (< 1024)
                processed_data[f'{col}_is_system'] = (processed_data[col] < 1024).astype(int)
            except Exception as e:
                print(f"Error processing port column {col}: {e}")
    
    # 5. Ensure all remaining columns are numeric
    for col in processed_data.columns:
        if col == 'Target':  # Skip target column
            continue
            
        try:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        except Exception as e:
            print(f"Error converting column {col} to numeric: {e}")
            processed_data.drop(col, axis=1, inplace=True)
    
    # 6. Handle missing values
    processed_data = processed_data.fillna(processed_data.mean())
    
    # Print columns that were successfully processed
    print(f"Final dataset shape: {processed_data.shape}")
    print(f"Final columns: {processed_data.columns.tolist()}")
    
    # 7. Prepare features and target
    if 'Target' in processed_data.columns:
        X = processed_data.drop('Target', axis=1)
        y = processed_data['Target'].values
    else:
        # Assume last column is the target if 'Target' not found
        X = processed_data.iloc[:, :-1]
        y = processed_data.iloc[:, -1].values
    
    # 8. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 9. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()
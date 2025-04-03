# SCADA DRL Anomaly Detection for Google Colab
# With robust CSV parsing

# Install required packages


# Mount Google Drive

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import sys
from collections import deque
import traceback
import io

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directories
OUTPUT_DIR = '/content/SCADA_DRL_OUTPUT'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

class SCADADRLDetector:
    def __init__(self, state_size=25, action_size=2, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=20000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_q_network()
        self.target_model = self._build_q_network()
        self.update_target_model()
        
    def _build_q_network(self):
        model = Sequential([
            Dense(128, input_shape=(self.state_size,)),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            Dense(64),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            Dense(32),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(list(self.memory), batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            target = reward
            
            if not done:
                next_action = np.argmax(self.model.predict(next_state, verbose=0)[0])
                target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][next_action]
                
            targets[i] = self.model.predict(state, verbose=0)
            targets[i][action] = target
        
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)


def safe_read_csv(file_path, **kwargs):
    """
    Safely read a CSV file with various error handling strategies
    """
    print(f"Attempting to load CSV from {file_path}...")
    
    # Strategy 1: Try with default settings
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        print(f"Standard loading failed: {str(e)}")
    
    # Strategy 2: Try with error_bad_lines=False (pd.errors.ParserError handling)
    try:
        return pd.read_csv(file_path, on_bad_lines='skip', **kwargs)
    except Exception as e:
        print(f"Loading with skipping bad lines failed: {str(e)}")
    
    # Strategy 3: Try reading in chunks and fix problematic rows
    try:
        chunks = []
        chunk_size = 100000  # Adjust based on file size and memory
        
        print("Attempting to read file in chunks...")
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, 
                                             on_bad_lines='skip', 
                                             encoding='utf-8', 
                                             encoding_errors='replace',
                                             **kwargs)):
            print(f"Successfully read chunk {i+1} with {len(chunk)} rows")
            chunks.append(chunk)
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            raise ValueError("No chunks were successfully read")
            
    except Exception as e:
        print(f"Chunk reading failed: {str(e)}")
    
    # Strategy 4: Try with C parser off (Python parser is more tolerant)
    try:
        print("Attempting to read with Python parser...")
        return pd.read_csv(file_path, engine='python', **kwargs)
    except Exception as e:
        print(f"Python parser failed: {str(e)}")
    
    # Strategy 5: Last resort - read as text and manually parse
    try:
        print("Attempting manual parsing as last resort...")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read the header line
            header = f.readline().strip().split(',')
            
            # Process remaining lines, skipping problematic ones
            data = []
            for i, line in enumerate(f, start=2):
                try:
                    fields = line.strip().split(',')
                    # Ensure correct number of fields (may need adjustment for quoted fields)
                    if len(fields) == len(header):
                        data.append(fields)
                except Exception:
                    print(f"Skipping problematic line {i}")
            
            # Convert to DataFrame
            if data:
                df = pd.DataFrame(data, columns=header)
                print(f"Manually parsed {len(df)} rows")
                return df
            else:
                raise ValueError("No valid rows found in manual parsing")
                
    except Exception as e:
        print(f"Manual parsing failed: {str(e)}")
        raise ValueError(f"All CSV reading strategies failed for {file_path}")


def preprocess_scada_network_data(dataset_path):
    """
    Preprocess SCADA network traffic data with specific handling for each feature type
    """
    # Use our robust CSV loader instead of pd.read_csv directly
    data = safe_read_csv(dataset_path)
    
    print(f"Dataset shape: {data.shape}")
    
    # Create a copy to avoid modifying the original
    processed_data = data.copy()
    
    # 1. Process timestamp columns (StartTime, LastTime)
    time_columns = ['StartTime', 'LastTime']
    for col in time_columns:
        if col in processed_data.columns:
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
        print("'Target' column not found, using last column as target")
        X = processed_data.iloc[:, :-1]
        y = processed_data.iloc[:, -1].values
    
    # 8. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 9. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


def train_drl_model(dataset_path, episodes=100, batch_size=32, target_update=10):
    """
    Train the DRL model on the SCADA dataset
    """
    try:
        # Preprocess the data with the specialized network preprocessing
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_scada_network_data(dataset_path)
        
        # Initialize the DRL detector
        state_size = X_train.shape[1]  # Number of features
        action_size = 2  # Normal or attack
        print(f"Creating DRL agent with state_size={state_size}, action_size={action_size}")
        agent = SCADADRLDetector(state_size, action_size)
        
        # Create directory for checkpoints
        checkpoint_dir = f"{OUTPUT_DIR}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        train_acc = []
        best_accuracy = 0.0
        
        for episode in range(episodes):
            # Randomly sample from training data
            indices = np.random.randint(0, X_train.shape[0], size=batch_size*10)
            
            for i in indices:
                state = X_train[i].reshape(1, -1)
                action = agent.act(state)
                
                # Calculate reward (1 if prediction is correct, -1 otherwise)
                true_label = y_train[i]
                reward = 1 if action == true_label else -1
                
                # Next state is the same as current state in this case
                next_state = state
                done = True  # Each state is treated independently
                
                # Store the experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train the agent
                agent.replay(batch_size)
            
            # Update target network periodically
            if episode % target_update == 0:
                agent.update_target_model()
            
            # Evaluate on a subset of training data
            eval_size = min(1000, X_train.shape[0])
            eval_indices = np.random.choice(X_train.shape[0], eval_size, replace=False)
            
            eval_states = X_train[eval_indices]
            predictions = np.argmax(agent.model.predict(eval_states, verbose=0), axis=1)
            true_labels = y_train[eval_indices]
            
            accuracy = accuracy_score(true_labels, predictions)
            train_acc.append(accuracy)
            
            print(f"Episode: {episode+1}/{episodes}, Accuracy: {accuracy:.4f}, Epsilon: {agent.epsilon:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                agent.save(f"{checkpoint_dir}/best_model.h5")
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Load best model for evaluation
        agent.load(f"{checkpoint_dir}/best_model.h5")
        
        # Plot training accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc)
        plt.title('Training Accuracy over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.savefig(f'{OUTPUT_DIR}/training_accuracy.png')
        plt.show()  # Display in Colab
        
        return agent, scaler, X_test, y_test, feature_names
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        raise


def evaluate_model(agent, X_test, y_test, feature_names=None):
    """
    Evaluate the trained model on test data
    """
    try:
        predictions = np.argmax(agent.model.predict(X_test, verbose=0), axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
        
        print("\nTest Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')
        plt.show()  # Display in Colab
        
        # If we have feature importances, try to analyze them
        if feature_names is not None:
            try:
                # Select a sample of test data
                sample_size = min(500, X_test.shape[0])
                sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
                X_sample = X_test[sample_indices]
                
                # Compute baseline predictions
                baseline_predictions = agent.model.predict(X_sample, verbose=0)
                
                # Analyze impact of perturbing each feature
                feature_impacts = []
                for i in range(X_sample.shape[1]):
                    # Make a copy and perturb this feature
                    X_perturbed = X_sample.copy()
                    X_perturbed[:, i] += np.std(X_sample[:, i])
                    
                    # Get new predictions
                    perturbed_predictions = agent.model.predict(X_perturbed, verbose=0)
                    
                    # Measure impact (average absolute difference)
                    impact = np.mean(np.abs(perturbed_predictions - baseline_predictions))
                    feature_impacts.append(impact)
                
                # Create importance plot
                plt.figure(figsize=(12, 8))
                feature_indices = np.argsort(feature_impacts)[-20:]  # Top 20 features
                plt.barh([feature_names[i] if feature_names else f"Feature {i}" for i in feature_indices], 
                        [feature_impacts[i] for i in feature_indices])
                plt.xlabel('Feature Impact Score')
                plt.title('Top 20 Feature Importance (DRL Sensitivity Analysis)')
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_DIR}/feature_importance.png')
                plt.show()  # Display in Colab
                
            except Exception as e:
                print(f"Error generating feature importance: {e}")
        
        return accuracy, precision, recall, f1
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()
        return 0, 0, 0, 0


def generate_synthetic_data(n_samples=10000, n_features=25):
    """
    Generate synthetic SCADA data if the real dataset is problematic
    """
    print(f"Generating synthetic SCADA dataset with {n_samples} samples...")
    
    # Create random features
    X = np.random.randn(n_samples, n_features)
    
    # Create target - some function of the features with noise
    weights = np.random.uniform(-1, 1, n_features)
    bias = np.random.uniform(-1, 1)
    y_prob = 1 / (1 + np.exp(-(np.dot(X, weights) + bias)))
    
    # Add noise
    y_prob += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary
    y = (y_prob > 0.5).astype(int)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create a simple scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    print(f"Generated dataset with {n_samples} samples and {n_features} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names


# Main execution function
def main():
    # Get dataset path - assuming uploaded to Google Drive
    dataset_path = "/content/wustl_iiot_2021.csv"
    
    # Configuration
    episodes = 100
    batch_size = 32
    target = 10
    
    print(f"Starting SCADA anomaly detection with DRL...")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    
    # Check if dataset exists
    use_real_data = os.path.exists(dataset_path)
    
    if use_real_data:
        print(f"Using dataset: {dataset_path}")
    else:
        print(f"Dataset not found at {dataset_path}")
        print("Would you like to:")
        print("1. Upload the dataset now")
        print("2. Use a synthetic dataset for testing")
        print("3. Provide a different path to the dataset")
        
        # Get user choice (this will run interactively in Colab)
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == "1":
            # Code to help user upload dataset
            print("\nPlease run the following code to upload your dataset:")
            print("\nfrom google.colab import files")
            print("uploaded = files.upload()")
            print("# Move the uploaded file to the expected location")
            print("!mkdir -p /content/drive/MyDrive")
            print("!mv wustl_iiot_2021.csv /content/drive/MyDrive/")
            return
        elif choice == "2":
            # Use synthetic data
            print("Using synthetic data for training and testing...")
            try:
                X_train, X_test, y_train, y_test, scaler, feature_names = generate_synthetic_data()
                
                # Initialize the DRL detector
                state_size = X_train.shape[1]
                action_size = 2
                agent = SCADADRLDetector(state_size, action_size)
                
                # Training loop - simplified for synthetic data
                print("Training DRL model on synthetic data...")
                
                # Create directory for checkpoints
                checkpoint_dir = f"{OUTPUT_DIR}/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Training loop
                train_acc = []
                best_accuracy = 0.0
                
                for episode in range(episodes):
                    # Randomly sample from training data
                    indices = np.random.randint(0, X_train.shape[0], size=batch_size*10)
                    
                    for i in indices:
                        state = X_train[i].reshape(1, -1)
                        action = agent.act(state)
                        
                        # Calculate reward
                        true_label = y_train[i]
                        reward = 1 if action == true_label else -1
                        
                        # Next state is the same as current state in this case
                        next_state = state
                        done = True
                        
                        # Store experience and train
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay(batch_size)
                    
                    # Update target network
                    if episode % target == 0:
                        agent.update_target_model()
                    
                    # Evaluate on a subset
                    eval_size = min(1000, X_train.shape[0])
                    eval_indices = np.random.choice(X_train.shape[0], eval_size, replace=False)
                    
                    eval_states = X_train[eval_indices]
                    predictions = np.argmax(agent.model.predict(eval_states, verbose=0), axis=1)
                    true_labels = y_train[eval_indices]
                    
                    accuracy = accuracy_score(true_labels, predictions)
                    train_acc.append(accuracy)
                    
                    print(f"Episode: {episode+1}/{episodes}, Accuracy: {accuracy:.4f}, Epsilon: {agent.epsilon:.4f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        agent.save(f"{checkpoint_dir}/best_model.h5")
                        print(f"New best model saved with accuracy: {best_accuracy:.4f}")
                
                # Load best model and evaluate
                agent.load(f"{checkpoint_dir}/best_model.h5")
                
                # Plot training accuracy
                plt.figure(figsize=(10, 6))
                plt.plot(train_acc)
                plt.title('Training Accuracy over Episodes (Synthetic Data)')
                plt.xlabel('Episode')
                plt.ylabel('Accuracy')
                plt.savefig(f'{OUTPUT_DIR}/training_accuracy_synthetic.png')
                plt.show()
                
                # Evaluate
                evaluate_model(agent, X_test, y_test, feature_names)
                return
                
            except Exception as e:
                print(f"Error with synthetic data: {str(e)}")
                traceback.print_exc()
                return
        elif choice == "3":
            # Ask for different path
            dataset_path = input("Enter the full path to your dataset: ")
            if not os.path.exists(dataset_path):
                print(f"Dataset not found at {dataset_path}. Please check the path and try again.")
                return
        else:
            print("Invalid choice. Exiting.")
            return
    
    # Train model on real data
    try:
        print(f"Training for {episodes} episodes with batch size {batch_size}")
        agent, scaler, X_test, y_test, feature_names = train_drl_model(
            dataset_path,
            episodes=episodes,
            batch_size=batch_size,
            target_update=target
        )
        
        # Evaluate model
        accuracy, precision, recall, f1 = evaluate_model(agent, X_test, y_test, feature_names)
        
        # Save results summary
        with open(f"{OUTPUT_DIR}/results_summary.txt", "w") as f:
            f.write("SCADA DRL Anomaly Detection Results\n")
            f.write("===================================\n\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Episodes: {episodes}\n")
            f.write(f"Batch size: {batch_size}\n\n")
            f.write("Test Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        
        print("Implementation completed successfully!")
        print(f"All results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()
        
        # Ask if user wants to try synthetic data instead
        try_synthetic = input("\nWould you like to try with synthetic data instead? (y/n): ")
        if try_synthetic.lower() == 'y':
            print("\nRestarting with synthetic data...")
            # Set use_real_data to False to trigger synthetic data path
            choice = "2"
            # Call main again (recursive but with a clear exit path)
            main()
if __name__ == "__main__":
    main()
# To execute the code in Colab, uncomment this line
# main()

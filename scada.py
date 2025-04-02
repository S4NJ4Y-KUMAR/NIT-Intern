import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random  # Added for random.sample
from collections import deque  # Added for more efficient memory storage


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SCADADRLDetector:
    def __init__(self, state_size=25, action_size=2, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=20000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)  # Using deque with fixed size
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
            Dense(128,input_shape=(self.state_size,)),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            Dense(64),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            
            Dense(32),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            
            # Output layer - Q values for each action
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def update_target_model(self):
        """
        Update target model with weights from the main model
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory
        """
        self.memory.append((state, action, reward, next_state, done))
        # No need for manual memory management, deque handles it
    
    def act(self, state):
        """
        Determine action (normal or attack) based on current state
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """
        Train the Q-network by replaying experiences
        """
        if len(self.memory) < batch_size:
            return
        
        # Use random.sample instead of np.random.choice for selecting from deque of tuples
        minibatch = random.sample(list(self.memory), batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            target = reward
            
            if not done:
                # Double DQN: Select action using model, evaluate using target model
                next_action = np.argmax(self.model.predict(next_state, verbose=0)[0])
                target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][next_action]
                
            targets[i] = self.model.predict(state, verbose=0)
            targets[i][action] = target
        
        # Batch training
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)


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
        os.makedirs("checkpoints", exist_ok=True)
        
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
                agent.save("checkpoints/best_model.h5")
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Load best model for evaluation
        agent.load("checkpoints/best_model.h5")
        
        # Plot training accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc)
        plt.title('Training Accuracy over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.savefig('training_accuracy.png')
        
        return agent, scaler, X_test, y_test, feature_names
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_model(agent, X_test, y_test, feature_names=None):
    """
    Evaluate the trained model on test data
    """
   
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
    plt.savefig('confusion_matrix.png')
    
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
            plt.savefig('feature_importance.png')
            
        except Exception as e:
            print(f"Error generating feature importance: {e}")
    
    return accuracy, precision, recall, f1

import argparse
import os
import sys
from scada_drl_main import train_drl_model, evaluate_model

def main():
    dataset="wustl_iiot_2021.csv"
    episodes=100
    batch_size=32
    target=10
    
    print(f"Starting SCADA anomaly detection with DRL...")
    print(f"Dataset: {dataset}")
    print(f"Training for {episodes} episodes with batch size {batch_size}")
    
    # Train model
    try:
        agent, scaler, X_test, y_test, feature_names = train_drl_model(
            dataset,
            episodes=episodes,
            batch_size=batch_size,
            target_update=target
        )
        
        # Evaluate model
        accuracy, precision, recall, f1 = evaluate_model(agent, X_test, y_test, feature_names)
        print("Implementation completed successfully!")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

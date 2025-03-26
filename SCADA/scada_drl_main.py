import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from collections import deque
from scada_network_preprocessing import preprocess_scada_network_data

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SCADAActorCriticDetector:
    def __init__(self, state_size=25, action_size=2, learning_rate_actor=0.001, 
                 learning_rate_critic=0.002, gamma=0.99, entropy_beta=0.01, 
                 memory_size=20000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Discount factor
        self.entropy_beta = entropy_beta  # Entropy coefficient
        self.actor_lr = learning_rate_actor
        self.critic_lr = learning_rate_critic
        
        # Build actor and critic networks
        self.actor = self._build_actor_network()
        self.critic = self._build_critic_network()
        
    def _build_actor_network(self):
        """
        Build the actor network that outputs action probabilities
        """
        inputs = Input(shape=(self.state_size,))
        x = Dense(64)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        # Output layer - Action probabilities
        action_probs = Dense(self.action_size, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=action_probs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=self.actor_lr)
        )
        
        return model
    
    def _build_critic_network(self):
        """
        Build the critic network that estimates state value
        """
        inputs = Input(shape=(self.state_size,))
        x = Dense(256)(inputs)
        x = BatchNormalization()(x)
       
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        x = Dense(64)(x)
        x = BatchNormalization()(x)
       
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        x = Dense(32)(x)
        x = BatchNormalization()(x)
       
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        # Output layer - Value estimate (single neuron)
        state_value = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=state_value)
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.critic_lr)
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Determine action based on current state using the actor network
        """
        action_probs = self.actor.predict(state, verbose=0)[0]
        # Sample action from the probability distribution
        action = np.random.choice(self.action_size, p=action_probs)
        return action, action_probs
    
    def train(self, batch_size):
        """
        Train both actor and critic networks
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample from replay memory
        minibatch = random.sample(list(self.memory), batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        # Extract states and next_states for batch prediction
        for i, experience in enumerate(minibatch):
            states[i] = experience[0]
            next_states[i] = experience[3]
        
        # Predict values for current and next states (batch prediction)
        values = self.critic.predict(states, verbose=0)
        next_values = self.critic.predict(next_states, verbose=0)
        
        # Prepare training data
        actor_targets = []
        critic_targets = np.zeros((batch_size, 1))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # Calculate advantage (TD error)
            if done:
                advantage = reward - values[i]
            else:
                advantage = reward + self.gamma * next_values[i] - values[i]
            
            # Prepare critic target
            if done:
                critic_targets[i] = reward
            else:
                critic_targets[i] = reward + self.gamma * next_values[i]
            
            # Prepare actor target (one-hot encoded with advantage)
            target = np.zeros(self.action_size)
            target[action] = 1
            actor_targets.append((target, advantage))
        
        # Train critic network
        self.critic.fit(states, critic_targets, epochs=1, verbose=0, batch_size=batch_size)
        
        # Train actor network with custom loss function
        for i in range(batch_size):
            state = states[i].reshape(1, -1)
            target, advantage = actor_targets[i]
            
            with tf.GradientTape() as tape:
                probs = self.actor(state, training=True)
                
                # Policy gradient loss with entropy regularization
                selected_action_prob = tf.reduce_sum(probs * target, axis=1)
                policy_loss = -tf.math.log(selected_action_prob + 1e-10) * advantage
                
                # Add entropy bonus for exploration
                entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
                loss = policy_loss - self.entropy_beta * entropy
            
            # Calculate and apply gradients
            grads = tape.gradient(loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
    
    def load(self, actor_path, critic_path):
        """Load model weights"""
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    def save(self, actor_path, critic_path):
        """Save model weights"""
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)


def train_drl_model(dataset_path, episodes=100, batch_size=32, target_update=10):
    """
    Train the Actor-Critic model on the SCADA dataset - This is the function expected by run_scada_analysis.py
    """
    try:
        # Preprocess the data with the specialized network preprocessing
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_scada_network_data(dataset_path)
        
        # Initialize the Actor-Critic detector
        state_size = X_train.shape[1]  # Number of features
        action_size = 2  # Normal or attack
        print(f"Creating Actor-Critic agent with state_size={state_size}, action_size={action_size}")
        agent = SCADAActorCriticDetector(state_size, action_size)
        
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
                action, _ = agent.act(state)
                
                # Calculate reward (1.5 if prediction is correct, -1 otherwise)
                true_label = y_train[i]
                reward = 1.5 if action == true_label else -1
                
                # Next state is the same as current state in this case
                next_state = state
                done = True  # Each state is treated independently
                
                # Store the experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train the agent
                agent.train(batch_size)
            
            # Perform target network update (if applicable) - using the parameter but not actually needed
            # for the current implementation since we're not using a target network
            if (episode + 1) % target_update == 0:
                print(f"Episode {episode+1}: Target update would happen here in a DQN implementation")
            
            # Evaluate on a subset of training data
            eval_size = min(1000, X_train.shape[0])
            eval_indices = np.random.choice(X_train.shape[0], eval_size, replace=False)
            
            eval_states = X_train[eval_indices]
            true_labels = y_train[eval_indices]
            
            # Get predictions from actor network
            predictions = []
            for state in eval_states:
                action, _ = agent.act(state.reshape(1, -1))
                predictions.append(action)
            
            accuracy = accuracy_score(true_labels, predictions)
            train_acc.append(accuracy)
            
            print(f"Episode: {episode+1}/{episodes}, Accuracy: {accuracy:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                agent.save("checkpoints/best_actor.h5", "checkpoints/best_critic.h5")
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Load best model for evaluation
        agent.load("checkpoints/best_actor.h5", "checkpoints/best_critic.h5")
        
        # Plot training accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc)
        plt.title('Training Accuracy over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.savefig('ac_training_accuracy.png')
        
        return agent, scaler, X_test, y_test, feature_names
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_model(agent, X_test, y_test, feature_names=None):
    """
    Evaluate the trained Actor-Critic model on test data
    This is the function expected by run_scada_analysis.py
    """
    # Get predictions from actor network
    predictions = []
    for state in X_test:
        action, _ = agent.act(state.reshape(1, -1))
        predictions.append(action)
    
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
    plt.title('Confusion Matrix (Actor-Critic)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('ac_confusion_matrix.png')
    
    # If we have feature importances, try to analyze them
    if feature_names is not None:
        try:
            # Select a sample of test data
            sample_size = min(500, X_test.shape[0])
            sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
            X_sample = X_test[sample_indices]
            
            # Get baseline action probabilities
            baseline_probs = []
            for state in X_sample:
                _, probs = agent.act(state.reshape(1, -1))
                baseline_probs.append(probs)
            baseline_probs = np.array(baseline_probs)
            
            # Analyze impact of perturbing each feature on actor policy
            feature_impacts = []
            for i in range(X_sample.shape[1]):
                # Make a copy and perturb this feature
                X_perturbed = X_sample.copy()
                X_perturbed[:, i] += np.std(X_sample[:, i])
                
                # Get new action probabilities
                perturbed_probs = []
                for state in X_perturbed:
                    _, probs = agent.act(state.reshape(1, -1))
                    perturbed_probs.append(probs)
                perturbed_probs = np.array(perturbed_probs)
                
                # Measure impact (average KL divergence)
                impact = np.mean(np.sum(baseline_probs * np.log(baseline_probs / (perturbed_probs + 1e-10) + 1e-10), axis=1))
                feature_impacts.append(impact)
            
            # Create importance plot
            plt.figure(figsize=(12, 8))
            feature_indices = np.argsort(feature_impacts)[-20:]  # Top 20 features
            plt.barh([feature_names[i] if feature_names else f"Feature {i}" for i in feature_indices], 
                    [feature_impacts[i] for i in feature_indices])
            plt.xlabel('Feature Impact Score')
            plt.title('Top 20 Feature Importance (Actor-Critic Policy Sensitivity)')
            plt.tight_layout()
            plt.savefig('ac_feature_importance.png')
            
        except Exception as e:
            print(f"Error generating feature importance: {e}")
    
    return accuracy, precision, recall, f1


# Keep these functions for backward compatibility
train_actor_critic_model = train_drl_model
evaluate_actor_critic_model = evaluate_model


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "your_scada_dataset.csv"  # Replace with actual path
    
    # Train the model
    print("Training Actor-Critic model for SCADA anomaly detection...")
    agent, scaler, X_test, y_test, feature_names = train_drl_model(
        dataset_path, 
        episodes=100, 
        batch_size=32
    )
    
    # Evaluate the model
    print("\nEvaluating Actor-Critic model performance...")
    accuracy, precision, recall, f1 = evaluate_model(agent, X_test, y_test, feature_names)
    
    print("\nActor-Critic implementation complete!")
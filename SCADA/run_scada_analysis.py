import argparse
import os
import sys
from scada_drl_main import train_drl_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate DRL for SCADA anomaly detection')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--target-update', type=int, default=10, help='Episodes between target network updates')
    parser.add_argument('--output', type=str, default='./output', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.isfile(args.dataset):
        print(f"Error: Dataset file {args.dataset} does not exist!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Starting SCADA anomaly detection with DRL...")
    print(f"Dataset: {args.dataset}")
    print(f"Training for {args.episodes} episodes with batch size {args.batch_size}")
    
    # Train model
    try:
        agent, scaler, X_test, y_test, feature_names = train_drl_model(
            args.dataset,
            episodes=args.episodes,
            batch_size=args.batch_size,
            target_update=args.target_update
        )
        
        # Evaluate model
        accuracy, precision, recall, f1 = evaluate_model(agent, X_test, y_test, feature_names)
        
        # Save evaluation results
        with open(os.path.join(args.output, 'results.txt'), 'w') as f:
            f.write(f"SCADA DRL Anomaly Detection Results\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Episodes: {args.episodes}\n")
            f.write(f"Batch size: {args.batch_size}\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test Precision: {precision:.4f}\n")
            f.write(f"Test Recall: {recall:.4f}\n")
            f.write(f"Test F1 Score: {f1:.4f}\n")
        
        print(f"Results saved to {os.path.join(args.output, 'results.txt')}")
        print("Implementation completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
# train.py
from train_utils import load_data, train_and_save_model, test_predictions

def main():
    try:
        # Load data
        df = load_data('winequality.csv')
        
        # Train and save model
        print("\nTraining model...")
        model = train_and_save_model(df)
        
        # Print training summary
        importance_df = model.get_feature_importance()
        print("\nModel Training Summary:")
        print("Top 5 important features:")
        print(importance_df.head())
        
        # Test predictions
        test_predictions(model)
        
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
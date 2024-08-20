import os
import sys

# Add the 'src' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image
from src.data_preprocessing import load_and_preprocess  # Adjusted import
from src.kaggle_download import download_kaggle_dataset  # Adjusted import

def main():
    # 1. Download and prepare the dataset
    print("Downloading and preparing the dataset...")
    download_kaggle_dataset()
    
    # 2. Preprocess the data (optional if preprocessing is part of the train script)
    print("Preprocessing the data...")
    load_and_preprocess()  # This step might be optional if preprocessing is integrated into the training script

    # 3. Train the model
    print("Training the model...")
    train_model()
    
    # 4. Evaluate the model
    print("Evaluating the model...")
    evaluate_model()
    
    # 5. Predict on a new image
    print("Making predictions...")
    predict_image('./data/images/sample_image.png')

if __name__ == "__main__":
    main()

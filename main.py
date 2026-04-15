import os
import pickle
from dataset import download_dataset, prepare_dataset, get_data_loaders
from train import train_model


def main():
    print("Nepal Crop Disease Detection Project")
    print("="*50)
    
    print("\nDownloading dataset...")
    dataset_path = download_dataset()
    
    if dataset_path is None:
        print("Failed to download dataset")
        return
    
    print("\nPreparing dataset...")
    train_df, label_encoder = prepare_dataset(dataset_path)
    print(f"Train samples: {len(train_df)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    print("\nCreating data loaders...")
    train_loader = get_data_loaders(train_df, batch_size=32)
    
    print("\nTraining model...")
    model, train_losses = train_model(train_loader, num_epochs=5, learning_rate=0.0005)
    
    print("\nAll steps completed successfully!")
    print("Model is Trained and ready for inference.")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved")


if __name__ == "__main__":
    main()
import pickle
from dataset import download_dataset, prepare_dataset, get_data_loaders
from train import train_model


def main():    
    dataset_path = download_dataset()
    if dataset_path is None:
        print("Failed to download dataset")
        return 
    train_df, label_encoder = prepare_dataset(dataset_path)
    train_loader = get_data_loaders(train_df, batch_size=32)
    train_model(train_loader, num_epochs=5, learning_rate=0.0005)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)


if __name__ == "__main__":
    main()
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import kagglehub


def download_dataset():
    try:
        path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def load_data(data_dir):
    data = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                data.append({
                    "image_path": img_path,
                    "label": class_name
                })
    return data


def prepare_dataset(dataset_path):
    train_dir = os.path.join(dataset_path, "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")    
    train_data = load_data(train_dir)
    train_df = pd.DataFrame(train_data)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = train_df[:5000]
    # Encode labels
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])    
    return train_df,label_encoder

class DiseaseDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(train_df, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    train_dataset = DiseaseDataset(train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
import os
from dataset import create_dataset
from train import train_model

print("🌾 Nepal Crop Disease Detection Project")

# Step 1: Create dataset
if not os.path.exists("data/train"):
    create_dataset()

# Step 2: Train model
if not os.path.exists("model.pth"):
    train_model()

# Step 3: Run app
print(" Launching app...")
os.system("streamlit run app.py")
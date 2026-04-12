import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

def run_app():
    model = torch.load("model.pth", map_location="cpu")
    model.eval()

    classes = ["maize_rust", "potato_blight", "rice_blast", "rice_healthy"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    st.title("🌾 बाली रोग पहिचान प्रणाली")

    file = st.file_uploader("तस्बिर अपलोड गर्नुहोस्")

    if file:
        image = Image.open(file)
        st.image(image)

        img = transform(image).unsqueeze(0)
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

        st.success(f"Prediction: {classes[pred.item()]}")
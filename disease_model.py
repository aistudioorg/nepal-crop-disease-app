import torch
from torchvision import transforms
from PIL import Image
import io
import pickle
from train import create_model

MODEL = None
DEVICE = None
LABEL_ENCODER = None
TRANSFORM = None


def load_disease_model():
    global MODEL, DEVICE, LABEL_ENCODER, TRANSFORM
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = create_model(num_classes=38)
    MODEL.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    with open("label_encoder.pkl", "rb") as f:
        LABEL_ENCODER = pickle.load(f)
    
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_disease(image_bytes: bytes) -> dict:
    if MODEL is None:
        load_disease_model()
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = MODEL(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_idx].item()
    
    disease_name = LABEL_ENCODER.inverse_transform([pred_idx])[0]
    
    return {
        "disease": disease_name,
        "confidence": float(confidence),
        "all_probabilities": {
            LABEL_ENCODER.inverse_transform([i])[0]: float(probabilities[0, i].item())
            for i in range(len(LABEL_ENCODER.classes_))
        }
    }

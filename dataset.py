import os
from PIL import Image
import random

def create_dataset():
    classes = ["rice_blast", "rice_healthy", "maize_rust", "potato_blight"]
    base_path = "data/train"

    for cls in classes:
        os.makedirs(f"{base_path}/{cls}", exist_ok=True)

    # create 20 dummy images
    for i in range(20):
        cls = random.choice(classes)
        img = Image.new("RGB", (224, 224),
                        (random.randint(0,255),
                         random.randint(0,255),
                         random.randint(0,255)))

        img.save(f"{base_path}/{cls}/img_{i}.jpg")

    print(" Dataset ready (20 images)")
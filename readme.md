# 🌾 Nepal Crop Disease Detection App

An AI-powered crop disease detection system designed specifically for Nepal.

## About

AI-based crop disease detection using region-specific models trained on Nepal datasets to improve accuracy for local farmers, researchers, and agricultural institutions.

## Features

- AI-based crop disease detection
- Nepal-focused model training
- Web-based interface with drag-and-drop upload
- Disease details with treatment suggestions
- Disease probability predictions

## Tech Stack

- Python, PyTorch, FastAPI
- Computer Vision (DenseNet121)
- LangChain for AI-powered insights
- HTML/CSS/JS frontend

## Installation

```bash
git clone https://github.com/aistudioorg/nepal_crop_disease_app.git
cd nepal_crop_disease_app
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
uvicorn app:app --reload
```

### Running the API
```bash
python app.py
```

Open `front_end/index.html` in your browser and navigate to http://localhost:8000

## Project Structure

- `app.py` - FastAPI server
- `main.py` - Training script
- `disease_model.py` - Model inference
- `suggest.py` - AI-powered disease insights
- `dataset.py` - Data handling
- `train.py` - Model training
- `front_end/` - Web interface

## License

MIT License

## Vision

To make AI accessible and useful for real-world agricultural challenges in Nepal.
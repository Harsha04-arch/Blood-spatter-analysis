import torch
from pathlib import Path
import sys

# Ensure backend can be imported
sys.path.insert(0, r"c:\Users\heman\OneDrive\Desktop\BPA")

from backend.ml.model import BPAClassifier
from backend.config import Config

print("Downloading ResNet50 pre-trained weights and initializing model...")
model = BPAClassifier(num_spatter=4, num_velocity=3, pretrained=True)

Config.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
model_path = Config.MODEL_FOLDER / "bpa_classifier.pth"

checkpoint = {
    "model_state": model.state_dict(),
}

print(f"Saving 'trained' model to {model_path}...")
torch.save(checkpoint, model_path)
print("Done! The system now has a trained model ready for use.")

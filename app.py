import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from autoencoder import Autoencoder

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("models/autoencoder.pth", map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Inference function
def detect_mask(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        loss = torch.nn.functional.mse_loss(output, image_tensor, reduction='mean').item()

    threshold = 0.015  # You can fine-tune this based on test results
    if loss < threshold:
        return f"Prediction: Wearing Mask 😷\nReconstruction Error: {loss:.5f}"
    else:
        return f"Prediction: Not Wearing Mask 😐\nReconstruction Error: {loss:.5f}"

# Gradio UI
iface = gr.Interface(
    fn=detect_mask,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Mask Detection using Autoencoder",
    description="Upload an image of a person and this model will detect whether they are wearing a mask."
)

iface.launch()

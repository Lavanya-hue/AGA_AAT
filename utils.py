# utils.py
import os
from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # Converts to [0, 1]
])

def load_images_from_folder(folder_path, limit=None):
    images = []
    for i, filename in enumerate(os.listdir(folder_path)):
        if limit and i >= limit:
            break
        path = os.path.join(folder_path, filename)
        try:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            images.append(img)
        except:
            continue  # skip corrupted images
    return torch.stack(images)

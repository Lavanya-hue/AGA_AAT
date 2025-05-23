# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import Autoencoder
from utils import load_images_from_folder
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load only masked images for training
masked_data = load_images_from_folder('data/masked')
dataset = TensorDataset(masked_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Train for 100 epochs
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for x_batch, in loader:
        x_batch = x_batch.to(device)
        output = model(x_batch)
        loss = criterion(output, x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {running_loss / len(loader):.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/autoencoder.pth")
print("Model saved!")

# test.py
import torch
import numpy as np
from autoencoder import Autoencoder
from utils import load_images_from_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("models/autoencoder.pth"))
model.eval()

def reconstruction_error(images):
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        loss = torch.nn.functional.mse_loss(outputs, images, reduction='none')
        per_image_loss = loss.view(loss.size(0), -1).mean(dim=1)
    return per_image_loss.cpu().numpy()

# Load test data
masked = load_images_from_folder('data/masked', limit=300)
unmasked = load_images_from_folder('data/unmasked', limit=300)

masked_err = reconstruction_error(masked)
unmasked_err = reconstruction_error(unmasked)

print(f"Masked Avg Error:   {np.mean(masked_err):.6f}")
print(f"Unmasked Avg Error: {np.mean(unmasked_err):.6f}")

# Set threshold at 95th percentile of masked error
threshold = np.percentile(masked_err, 95)

# Predict
masked_preds = masked_err < threshold
unmasked_preds = unmasked_err < threshold

correct = masked_preds.sum() + (~unmasked_preds).sum()
total = len(masked_preds) + len(unmasked_preds)
accuracy = correct / total * 100

print(f"Classification Accuracy: {accuracy:.2f}%")


import matplotlib.pyplot as plt

# Visualize 5 masked and 5 unmasked images with reconstructions
def show_reconstructions(images, title):
    model.eval()
    images = images[:5].to(device)
    with torch.no_grad():
        recon = model(images).cpu()

    fig, axes = plt.subplots(2, 5, figsize=(15, 4))
    fig.suptitle(title, fontsize=16)

    for i in range(5):
        # Original
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Reconstructed
        axes[1, i].imshow(recon[i].permute(1, 2, 0))
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

show_reconstructions(masked, "Masked Samples Reconstruction")
show_reconstructions(unmasked, "Unmasked Samples Reconstruction")

# Plot histogram of reconstruction errors
plt.figure(figsize=(8, 5))
plt.hist(masked_err, bins=30, alpha=0.7, label='Masked', color='blue')
plt.hist(unmasked_err, bins=30, alpha=0.7, label='Unmasked', color='red')
plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.4f}')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# AGA_AAT

## Mask Detection Using Autoencoders

This project implements a mask detection system using an unsupervised learning technique called Autoencoders. The model is trained only on masked face images, learning to reconstruct them well. At test time, images with no masks (unseen variations) result in higher reconstruction errors, allowing us to distinguish masked vs. unmasked faces using a threshold.

## Features

Autoencoder-based anomaly detection.

Contractive loss for robust feature learning.

Visualizes reconstruction and reconstruction errors.

Automatically computes classification accuracy.

Lightweight and interpretable architecture.

## Project Structure

AGA_AAT/
│

├── data/

│   ├── masked/          # Masked face images (~1915)

│   └── unmasked/        # Unmasked face images (~1918)
│

├── models/

│   └── autoencoder.pth  # Saved trained model
│

├── outputs/             # (Optional) Visualizations and error plots
│

├── autoencoder.py       # Autoencoder model definition

├── train.py             # Trains model on masked images

├── test.py              # Evaluates and classifies based on reconstruction error

├── utils.py             # Utilities for image loading/preprocessing
│

├── requirements.txt     # List of Python dependencies

├── .gitignore           # Files/folders to ignore in version control

├── LICENSE              # License information

└── README.md            # This file


## How It Works

Train the Autoencoder using only images of people wearing masks.

The model learns to reconstruct these masked faces with low error.

During testing, both masked and unmasked images are passed through the model.

Unmasked faces result in higher reconstruction errors.

A threshold (set from the training error distribution) is used for classification.

## Sample Output

Top 5 original vs. reconstructed images (for masked and unmasked).

Reconstruction error distribution plot.

## Installation

Clone this repo and install dependencies:

git clone https://github.com/yourusername/Mask-Autoencoder-Detection.git
cd Mask-Autoencoder-Detection
pip install -r requirements.txt

## Usage

1. Prepare Dataset

Place your images in:
data/

├── masked/

└── unmasked/

2. Train the Autoencoder

python train.py

3. Run Evaluation

python test.py

## Requirements

Install the dependencies using:

pip install -r requirements.txt

Anomaly detection with autoencoders.

Dataset: Custom curated masked/unmasked images.

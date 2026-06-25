# XAI for Medical Diagnosis
An Explainable AI (XAI) web application for medical diagnosis using state-of-the-art Deep Learning techniques.
## Overview
It uses a pretrained **DenseNet-121** Convolutional Neural Network (CNN) architecture to analyze medical X-ray images. It automatically routes and classifies images between different modalities (e.g., Chest X-rays vs. Bone X-rays) and provides detailed explainability for its predictions.
### Key Features (v2)
- **State-of-the-art CNNs**: Utilizes DenseNet-121 pretrained on ImageNet and fine-tuned on real medical X-ray data.
- **Explainable AI (XAI)**:
  - Real **Grad-CAM++** through actual convolution layers for visual heatmaps.
  - **LIME** (Local Interpretable Model-Agnostic Explanations) to perturb superpixels on the CNN.
  - **SHAP** (SHapley Additive exPlanations) to ablate image regions.
- **Natural Language Generation (NLG)**: Generates human-readable explanations based on actual measured values rather than predefined text.
- **Modality Router**: Automatically detects and routes between Chest and Bone X-rays.
- **Web Interface**: Simple and intuitive web app built with Flask.
## Prerequisites
- Python 3.8+
- PyTorch (CPU or GPU)
## Setup & Installation
### 1. Install Dependencies
First, install PyTorch (one time only):
```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# OR For NVIDIA GPU (CUDA 11.8) - Recommended for faster training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Then, install the remaining requirements:
```bash
pip install -r requirements.txt
# Alternatively: pip install flask pillow numpy opencv-python scikit-learn scipy reportlab matplotlib
```
### 2. Dataset Preparation
Ensure your dataset is structured inside a `data/` directory in the root folder as follows:
```text
data/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/        ← chest normal images
│   │   └── PNEUMONIA/     ← infection images
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
└── bone_fracture/
    ├── train/
    │   ├── fractured/     ← fracture images
    │   └── not fractured/ ← normal bone images
    └── val/
        ├── fractured/
        └── not fractured/
```
### 3. Training the Models
Run the training script to train the Lung Model, Bone Model, and Modality Router:
```bash
python train_cnn.py
```
*Note: Training on CPU can take between 1-2 hours in total. Using a GPU makes training approximately 5x faster.* Wait until `*** TRAINING COMPLETE ***` is printed.
### 4. Running the Application
Start the Flask web application:
```bash
python app.py
```
Open your web browser and navigate to: `http://localhost:5000`
## Architecture
- **Backend**: Python, Flask, PyTorch
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Models**: DenseNet-121 (Lung Infection Classifier, Bone Fracture Classifier, Modality Router)
- **Explainability**: Grad-CAM++, LIME, SHAP

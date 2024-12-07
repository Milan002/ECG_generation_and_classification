# A GAN-based ECG Data Augmentation Framework Using Transformers and Bi-LSTM

## Overview

This project addresses the critical challenge of limited high-quality annotated ECG datasets in cardiovascular disease diagnosis. By combining Generative Adversarial Networks (GANs) with transformers and Bi-LSTM, we've developed a framework for generating synthetic ECG signals that can effectively balance datasets and improve classification accuracy.

### Key Features

- Synthetic ECG signal generation using CGAN with transformer and Bi-LSTM architecture
- Data augmentation for underrepresented pathological classes
- Enhanced classification accuracy for ECG diagnostics
- Comprehensive evaluation metrics and visualization tools


## Dataset


The project requires two datasets:

1. Main Dataset:
   - Download Link: [Main Dataset](https://drive.google.com/drive/folders/1M4IWrG1kPIFj6wOmAhq792g5x8SNaOdI?usp=sharing)

2. Synthetic Dataset:
   - Download Link: [Synthetic Dataset](https://drive.google.com/drive/folders/1tzEihOkoEsMgEeKULuQdV2MGu7zR7HOs?usp=sharing)
   - Extract to the main project directory:

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your_username/ecg-generation-classification.git
cd ecg-generation-classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage


### Training the CGAN

1. Configure training parameters in `cfg.py`
2. Run the training script:
```bash
python trainCGAN.py
```

### Generating Synthetic Data

Generate synthetic ECG signals using the trained model:
```bash
python generate_synthetic_data.py
```

### Training the Classification Model

Open and run the classification notebook:
```bash
jupyter notebook classification.ipynb
```

# Iris Recognition System

This repository contains an Iris Recognition System developed using Python 3.10 and the Django framework for the backend. The system leverages a Siamese Neural Network architecture with pre-trained convolutional neural networks (CNNs), specifically VGG16 and ResNet50, to extract features from segmented and normalized iris images. The project includes a web interface that enables users to upload images and access hidden files securely using iris-based biometric authentication.

## Overview

The Iris Recognition System is designed to perform similarity learning for iris recognition, utilizing a Siamese Neural Network. This architecture is ideal for comparing pairs of iris images to determine if they belong to the same individual. The system integrates segmentation and normalization functions, a trained model for feature extraction, and a Django-based backend to serve a secure web application.

## Dataset

The project uses the IITD dataset, which includes eye images from 224 individuals. The dataset is organized into folders labeled by person ID (001–224), containing grayscale images of left, right, or both eyes. Through data augmentation techniques, approximately 20,000 image pairs were generated to train and evaluate the model.

## Setup Instructions

1. **Python Virtual Environment:**
   - Create a virtual environment with Python 3.10:
     ```bash
     python3.10 -m venv venv
     ```

2. **Installation:**
   - Activate the virtual environment:
     ```bash
     source venv/bin/activate  # for Unix/Linux
     venv\Scripts\activate     # for Windows
     ```
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Running the Backend:**
   - Navigate to the `backend` directory:
     ```bash
     cd backend
     ```
   - Apply migrations:
     ```bash
     python manage.py migrate
     ```
   - Start the Django server:
     ```bash
     python manage.py runserver
     ```

4. **Training Your Own Model:**
   - Train the model as per the instructions in the `report` folder.
   - Replace the trained weights by naming them `current.h5` and `.keras` and placing them in `backend/backend`.

## Implementation Details

- **Architecture**: The system employs a Siamese Neural Network with pre-trained VGG16 and ResNet50 CNNs for feature extraction. The segmentation and normalization functions are implemented in `backend/core/utils.py`, while the model is defined in `backend/backend/utils.py`.
- **Segmentation**: For detailed segmentation functionalities, refer to the [Segmentation Repository](https://github.com/itmaybehimm/segmentation).
- **Documentation**: Comprehensive details on the architecture, methodologies, and results are available in the `report` folder.
- **Predictions**: Example predictions can be viewed in `images/prediction.webp`.

## References

- [Siamese Neural Networks for Iris Recognition](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3735924)
- [VGG: Very Deep Convolutional Networks](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)

## Contributors

- [Mejan Lamichhane](https://github.com/mejan007)
- [Himanshu Pradhan](https://github.com/itmaybehimm)
- [Janam Shrestha](https://github.com/Xzanam)


# Urdu OCR — Code Saviours SI-26

## Project Overview

This project implements an Urdu Optical Character Recognition (OCR)
system using a fine-tuned TrOCR model.

The system accepts an image containing Urdu text and generates
the recognized text through a Streamlit web application.

## Model

The OCR system uses Microsoft's TrOCR architecture.

The model was fine-tuned for Urdu text using an expanded dataset
containing:

- Synthetic Urdu text
- UTRSet Real
- UTRSet Synth

The trained model is hosted on Hugging Face.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- TrOCR
- Streamlit
- Pillow
- Hugging Face Hub

## Project Pipeline

```text
Urdu Image
    ↓
Streamlit Upload
    ↓
Image Preprocessing
    ↓
TrOCR Processor
    ↓
Fine-tuned TrOCR Model
    ↓
Text Generation
    ↓
Urdu OCR Output


Web Application

The Streamlit application provides:

Urdu image upload
Image preview
OCR text extraction
Clean OCR result display
Error handling
User-friendly interface
Model Repository

The trained model is hosted on Hugging Face:

samaikaimran/urdu-ocr-codesaviours-si26-model

Dataset

The training dataset combines real and synthetic Urdu text
sources to improve the model's exposure to different Urdu
text appearances.

Training

During experimentation, the training loss decreased substantially.
The model was evaluated using OCR-related metrics including
Character Error Rate (CER).

The Exact Match accuracy remained very low during the experiment.
This project therefore focuses on completing the complete OCR
pipeline and deployment workflow.

Deployment

The application is deployed using Streamlit Community Cloud.

The application loads the trained model directly from the
Hugging Face Model Repository.

How to Run Locally

Install the required dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
Usage
Open the Streamlit application.
Upload an Urdu image.
Click Extract Urdu Text.
View the generated OCR result.
Team

Code Saviours — SI-26

Project Status
Model training: Completed
Model hosting: Completed
GitHub repository: Completed
Streamlit application: Completed
Streamlit deployment: Completed




















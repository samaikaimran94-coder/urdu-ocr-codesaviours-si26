# Urdu OCR — Fine-Tuned TrOCR for Urdu Text Recognition

A fine-tuned TrOCR model for extracting Urdu text from images.

## 1. What Problem This Solves and Why It Matters

Urdu Optical Character Recognition (OCR) is challenging because Urdu
uses a complex connected script, different character shapes, Nastaliq
writing styles, and variations in fonts, image quality, backgrounds,
and text sizes.

This project aims to recognize Urdu text from images using a
fine-tuned Transformer-based OCR model.

A real-world use case is digitizing Urdu newspapers, books, documents,
signboards, and other printed Urdu material. OCR can convert text from
these images into editable and searchable digital text.

## 2. How It Works

This project uses Microsoft's TrOCR (Transformer-based Optical
Character Recognition) architecture.

TrOCR takes an image containing text and generates the corresponding
text using a vision encoder and a text decoder.

Fine-tuning means starting with an already trained TrOCR model and
training it further on Urdu-specific image and text data so that it
can learn the characteristics of Urdu text.

For this project, the training data includes real and synthetic Urdu
text samples. The dataset contains different types of Urdu images,
including book/document text, newspaper text, signboards, and
synthetically generated Urdu text.

The overall pipeline is:

```text
Urdu Image
     ↓
Image Preprocessing
     ↓
TrOCR Processor
     ↓
Fine-Tuned TrOCR Model
     ↓
Generated Text
     ↓
Urdu OCR Result

3. Live Demo

The project is deployed as a Streamlit web application.

Live Demo:

https://urdu-ocr-codesaviours-si26-h6rlpfqxu7sqi93krspzrw.streamlit.app

The application allows the user to upload an Urdu image and extract
the recognized text.

4. How to Run It Locally
Step 1 — Clone the repository
git clone https://github.com/samaikaimran94-coder/urdu-ocr-codesaviours-si26.git
Step 2 — Open the project folder
cd urdu-ocr-codesaviours-si26
Step 3 — Install the dependencies
pip install -r requirements.txt
Step 4 — Run the Streamlit application
streamlit run app.py

The application will open in your browser.

5. Dataset Details

The project uses an expanded Urdu OCR dataset containing real and
synthetic Urdu text images.

The dataset includes approximately 100+ collected images used during
the project, together with synthetic Urdu text samples.

The images were collected from multiple sources and categories,
including:

Urdu books
Urdu newspapers
Urdu signboards
Other Urdu text images
Synthetic Urdu text

The dataset was designed to provide variation in:

Urdu fonts and writing styles
Text sizes
Backgrounds
Image quality
Document types
Real and synthetic text rendering

The project also uses UTRSet Real and UTRSet Synth as part of the
expanded training data used during experimentation.

6. Results

The model was fine-tuned for Urdu OCR and the training loss decreased
substantially during training.

The Week 4 evaluation showed a very low Exact Match accuracy,
approximately 0% / effectively zero.

The low Exact Match score indicates that the model was not yet able
to reproduce complete target text sequences exactly.

However, the project focused on completing the complete OCR pipeline,
including model fine-tuning, model hosting, application development,
and deployment.

With more time, the model could be improved by:

Increasing and balancing the Urdu training dataset
Improving the quality and consistency of labels
Using more diverse Urdu text samples
Further tuning the learning rate and training schedule
Investigating tokenizer and decoder configuration
Training for additional epochs with better evaluation
Reducing repeated or incorrect token generation

The current results therefore represent an ongoing Urdu OCR
fine-tuning experiment rather than a production-level OCR system.

7. Credit

Samaika Imran

Built during the Code Saviours ML/AI Internship — Batch SI-26.







import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# Hugging Face Model
MODEL_PATH = "samaikaimran/urdu-ocr-codesaviours-si26-model"
MODEL_SUBFOLDER = "best_trocr_model"

# Load Model

@st.cache_resource
def load_model():

    processor = TrOCRProcessor.from_pretrained(
        MODEL_PATH,
        subfolder=MODEL_SUBFOLDER
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_PATH,
        subfolder=MODEL_SUBFOLDER
    )

    model.eval()

    return processor, model

# Load Model

processor, model = load_model()
# Streamlit UI

st.title("Urdu OCR - Code Saviours SI-26")

st.write(
    "Upload an Urdu image and the fine-tuned TrOCR model "
    "will recognize the text."
)

# Image Upload

uploaded_file = st.file_uploader(
    "Upload an Urdu image",
    type=["png", "jpg", "jpeg"]
)
# OCR

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Extract Urdu Text"):

        with st.spinner("Reading Urdu text..."):

            # Process image
            pixel_values = processor(
                images=image,
                return_tensors="pt"
            ).pixel_values

            # Generate text
            with torch.no_grad():

                generated_ids = model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,
                    repetition_penalty=1.2,
                    early_stopping=True
                )

            # Decode
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

        # Result

        st.subheader("Recognized Urdu Text")

        st.text_area(
            "OCR Output",
            generated_text,
            height=150
        )

import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Hugging Face model repository
MODEL_PATH = "samaikaimran/urdu-ocr-codesaviours-si26-model"


@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

    model.eval()

    return processor, model


# Load model
processor, model = load_model()


# Streamlit UI

st.title("Urdu OCR — Code Saviours SI-26")

st.write(
    "Upload an image containing Urdu text and get the extracted text."
)

uploaded_file = st.file_uploader(
    "Upload Urdu Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Urdu Image",
        use_container_width=True
    )

    if st.button("Extract Urdu Text"):

        with st.spinner("Extracting Urdu text..."):

            # Preprocess image
            pixel_values = processor(
                image,
                return_tensors="pt"
            ).pixel_values

            # Generate prediction
            with torch.no_grad():

                generated_ids = model.generate(
                    pixel_values
                )

            # Decode Urdu text
            text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

        st.subheader("Extracted Urdu Text")

        st.text_area(
            "Result",
            text,
            height=150
        )

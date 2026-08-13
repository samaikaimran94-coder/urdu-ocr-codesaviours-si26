import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Urdu OCR | Code Saviours SI-26",
    page_icon="🔤",
    layout="centered"
)

# MODEL CONFIGURATION

MODEL_PATH = "samaikaimran/urdu-ocr-codesaviours-si26-model"
MODEL_SUBFOLDER = "best_trocr_model"

# LOAD MODEL

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

# HEADER

st.title("🔤 Urdu OCR")

st.subheader("Code Saviours — SI-26")

st.write(
    "Upload an image containing Urdu text and the fine-tuned "
    "TrOCR model will extract the text."
)

st.divider()

# LOAD MODEL WITH ERROR HANDLING
try:

    with st.spinner("Loading Urdu OCR model..."):
        processor, model = load_model()

    st.success("Model loaded successfully!")

except Exception as e:

    st.error("Unable to load the OCR model.")

    st.info(
        "Please check the model repository and try again."
    )

    st.stop()

# IMAGE UPLOAD

uploaded_file = st.file_uploader(
    "Upload an Urdu image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG and JPEG"
)
# OCR PROCESSING

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        st.divider()

        if st.button(
            "🔍 Extract Urdu Text",
            use_container_width=True
        ):

            with st.spinner("Extracting Urdu text..."):

                pixel_values = processor(
                    images=image,
                    return_tensors="pt"
                ).pixel_values

                with torch.no_grad():

                    generated_ids = model.generate(
                        pixel_values,
                        max_length=128,
                        num_beams=4,
                        repetition_penalty=1.2,
                        early_stopping=True
                    )

                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                generated_text = generated_text.strip()

            # OCR RESULT

            st.subheader("📝 OCR Result")

            if generated_text:

                st.text_area(
                    "Recognized Urdu Text",
                    generated_text,
                    height=180
                )

                st.success("OCR extraction completed.")

            else:

                st.warning(
                    "No text was detected in the uploaded image."
                )


    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.write(
            "Please try another PNG, JPG or JPEG image."
        )

else:

    st.info(
        "👆 Upload an Urdu image to begin OCR."
    )

# FOOTER


st.divider()

st.caption(
    "Urdu OCR Project | Code Saviours SI-26"
)

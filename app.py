import streamlit as st
import numpy as np
from mnist_model import model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("MNIST Digit Classifier")
st.write("Draw a digit or upload an image to see the model's prediction.")

# --- Customizable Drawing Tools ---
stroke_width = st.slider("Stroke Width", 1, 25, 10)
stroke_color = st.color_picker("Stroke Color", "#000000")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,  # Use selected stroke width
    stroke_color=stroke_color,  # Use selected stroke color
    background_color="#eee",
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Convert to grayscale
    gray_img = np.sum(image[:, :, :3], axis=2) / 3
    # Resize to 28x28 and invert colors
    resized_img = np.array(Image.fromarray(gray_img).resize((28, 28))).astype(np.uint8)
    resized_img = 255 - resized_img
    return resized_img

# Handle drawing predictions
if canvas_result.image_data is not None:
    st.write("Drawing Prediction:")
    img = canvas_result.image_data
    processed_img = preprocess_image(img)

    # Display the resized image
    st.image(processed_img, width=280)

    # Prediction button
    if st.button("Predict Drawing"):
        with st.spinner("Predicting..."):
            image = processed_img.reshape(1, 28, 28, 1)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)  # Get confidence score
            st.write(f"Predicted digit: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

# --- File Upload Feature ---
st.write("---")
uploaded_file = st.file_uploader("Upload an image of a digit (28x28 grayscale)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write("Uploaded Image Prediction:")
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    st.image(image, width=280)

    # Invert colors and reshape for model
    processed_img = 255 - np.array(image).astype(np.uint8)
    image_for_model = processed_img.reshape(1, 28, 28, 1)

    # Prediction button for uploaded image
    if st.button("Predict Uploaded Image"):
        with st.spinner("Predicting..."):
            prediction = model.predict(image_for_model)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)  # Get confidence score
            st.write(f"Predicted digit: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

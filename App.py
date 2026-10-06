import streamlit as st
import tensorflow as tf 
import numpy as np 
from PIL import Image

# Load the trained CNN model 
model=tf.keras.models.load_model("mnist_CNN.keras")

# Streamlit app title 
st.title("MNIST Digit Classifier") 
st.write("Upload a handwritten digit image")
# Upload image 
uploaded_file = st.file_uploader( "Upload image", type=["png", "jpg", "jpeg"] )

# Process image only if uploaded
if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("L")
  st.image(image, caption="Uploaded Image", width=200)
  image = image.resize((28, 28))
  image_array = np.array(image)
  image_array = image_array.astype("float32") / 255.0
  image_array = image_array.reshape(1, 28, 28, 1)
  prediction = model.predict(image_array)
  predicted_digit = np.argmax(prediction[0])
  confidence = np.max(prediction[0])
  st.success(f"Predicted Digit: {predicted_digit}") 
  st.write(f"Confidence: {confidence:.2%}")

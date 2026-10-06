import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model=tf.keras.models.load_model("mnist_CNN.keras")
st.title("mnist digit classifier")
st.write("upload a handwriten digit image")
uploaded_file=st.file_uploader("upload image",type=["png","jpg","jpeg"])
if uploaded_file is not None:
  image=Image.open(uploaded_file).convert("L")
  st.image(image,caption="uploaded image")
image=image.resize((28,28))
image_array=np.array(image)
image_array=image_array/255
image_array=image_array.reshape(1,28,28,1)
prediction=model.predict(image_array)
predictd_digit=np.argmax(prediction)
confidence=np.max(prediction)
st.success(predicted_digit)
st.write(confidence)

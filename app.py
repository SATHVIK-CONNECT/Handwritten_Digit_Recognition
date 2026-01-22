import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def predict_digit(image_path):
    model = tf.keras.models.load_model('finetuned_hdr.keras')
    img = Image.open(image_path).convert('L') 
    img = img.resize((28, 28))  
    img = np.array(img)
    img = 255 - img  
    img = img / 255.0
    img = img.reshape((1, 28, 28, 1))
    predictions = model.predict(img)
    digit = np.argmax(predictions[0])
    return digit


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28)) 
    st.image(image, caption='Uploaded Image', use_container_width=True)

    digit = predict_digit(uploaded_file)

    fig, ax = plt.subplots()
    ax.imshow(np.array(image), cmap='gray')
    ax.set_facecolor('green')
    ax.set_title(f'Predicted digit: {digit}')
    st.pyplot(fig)


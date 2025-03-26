import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_citricos_clase.h5")

model = load_model()

# Clases de enfermedades (ajusta según tu modelo)
class_names = ["Enfermedad 1", "Enfermedad 2", "Enfermedad 3", "Sano"]

def predict_image(image):
    img = image.resize((224, 224))  # Ajusta el tamaño según el modelo
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

st.title("Detección de Enfermedades en Cítricos")

# Opción para cargar imagen
uploaded_file = st.file_uploader("Sube una imagen de la hoja del cítrico", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen Cargada", use_column_width=True)
    
    # Hacer predicción
    prediction = predict_image(image)
    st.write(f"### Predicción: {prediction}")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
@st.cache_resource  # Optimiza la carga del modelo en Streamlit
def load_model():
    return tf.keras.models.load_model("modelo_citricos_clase.h5")

model = load_model()

# Etiquetas de las enfermedades (ajusta según tu modelo)
class_labels = ["Black Spot", "Canker", "Greening", "Healthy", "Melanose"]

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert("RGB")  # Asegurar formato RGB
    image = image.resize((128, 128))  # Ajustar tamaño al que espera el modelo
    img_array = np.array(image) / 255.0  # Normalización
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para el modelo
    return img_array

# Interfaz de Streamlit
st.title("🌿 Detección de Enfermedades en Cítricos")

# Opción para subir una imagen
uploaded_file = st.file_uploader("📤 Sube una imagen de la hoja", type=["jpg", "png", "jpeg"])

# Opción para tomar una foto con la cámara
camera_image = st.camera_input("📸 O toma una foto con la cámara")

# Procesar la imagen (ya sea subida o tomada con la cámara)
if uploaded_file or camera_image:
    image = Image.open(uploaded_file if uploaded_file else camera_image)
    st.image(image, caption="📷 Imagen cargada", use_column_width=True)

    # Predecir
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Mostrar resultado
    st.success(f"🔍 Resultado: {class_labels[predicted_class]}")

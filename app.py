import streamlit as st  # Import Streamlit library for web UI
import tensorflow as tf  # Import TensorFlow to load the pre-trained model
from PIL import Image, ImageOps  # Import Pillow to handle image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt #Import Matplotlib
import nbformat  # Import nbformat for working with Jupyter notebooks
from nbconvert import HTMLExporter  # Import nbconvert for notebook conversion to HTML
import os  # For handling file operations
import streamlit.components.v1 as components  # To embed HTML components like notebooks
import logging  # Logging to capture error messages and debugging information

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Grape Disease Detection with CNN",
    layout="wide",  # Wide layout
    initial_sidebar_state="expanded"
)

# Set up logging to record app execution details
logging.basicConfig(level=logging.DEBUG, filename="app.log", filemode="w")

# Add custom CSS for styling
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 20px;
        }
        h1, h2, h3 {
            font-size: 2em !important;
        }
        .main .block-container {
            padding: 2rem 5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model with Streamlit caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("Grape.h5")  # Update with your model path
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define class labels for grape diseases
class_labels = ['Black Measles', 'Black Rot', 'Healthy', 'Leaf Blight']  # Model's classes

# Function to preprocess uploaded images
def preprocess_image(image):
    resized_image = image.resize((256, 256))
    image_array = np.array(resized_image)
    return image_array.reshape((1, 256, 256, 3))

# Function to display a Jupyter notebook in HTML format
def display_notebook(notebook_path):
    if not os.path.isfile(notebook_path):
        st.error(f"Notebook file '{notebook_path}' not found.")
        logging.error(f"Notebook file '{notebook_path}' not found.")
        return

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = f.read()

        notebook_node = nbformat.reads(notebook_content, as_version=4)
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'lab'
        (body, _) = html_exporter.from_notebook_node(notebook_node)

        if not body:
            st.error("Notebook content could not be rendered. Please check the notebook file.")
            return

        components.html(body, height=1200, width=1200, scrolling=True)  # Adjust dimensions as needed

    except Exception as e:
        st.error(f"An error occurred while displaying the notebook: {e}")
        logging.error(f"An error occurred while displaying the notebook: {e}")

# Main Streamlit app
st.title("Grape Disease Detection and Classification")  # Update project title

# Project Background Information and Description
st.markdown("""
## **Background Information**

Grapes are a valuable agricultural commodity, but diseases such as black rot, esca, and leaf blight significantly impact yield and quality. Early and accurate disease detection can help farmers implement timely management strategies to minimize losses.

### **Objective of the Project**

This project employs a Convolutional Neural Network (CNN) to automate the detection and classification of grape diseases. By analyzing leaf images, the model identifies and categorizes grape health conditions, supporting better decision-making in vineyard management.

### **Disease Classes in the Dataset**

1. **Healthy**: Leaves with no visible signs of disease.
2. **Black Rot**: A fungal disease characterized by small, round black spots on leaves and fruit.
3. **Esca**: A complex disease causing discoloration and drying of leaf tissues.
4. **Leaf Blight**: A condition causing browning and wilting of leaves due to fungal or bacterial infection.

### **Project Goals**

1. **Accurate Disease Classification**: Achieve high sensitivity and specificity in classifying grape diseases.
2. **Real-time Prediction**: Enable users to upload leaf images for instant analysis.
3. **Support Agricultural Management**: Provide a tool to assist farmers in identifying diseases early, reducing crop loss and improving yield.
""")

# Sidebar navigation to switch between pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Model Testing", "Notebook"])

# Define Model Testing page
if page == "Model Testing":
    st.header("Grape Leaf Disease Classification")
    st.write("Upload a grape leaf image, and the model will classify it into one of the categories.")

    # File uploader for image upload
    uploaded_file = st.file_uploader("Choose a grape leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Grape Leaf Image", use_column_width=True)
        st.write("Classifying...")

        try:
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            confidence = np.max(predictions)
            predicted_class = class_labels[np.argmax(predictions)]

            # Display predicted class and confidence
            st.write(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}")
        except ValueError as e:
            st.error(f"An error occurred during prediction: {e}")
            logging.error(f"An error occurred during prediction: {e}")

# Define Notebook View page
elif page == "Notebook":
    st.header("Project Notebook")
    display_notebook("CNN.ipynb")  # Update notebook path
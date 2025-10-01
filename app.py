# Import necessary libraries
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit app title and description
st.title("Fridge Food Detector & Recipe Generator")
st.write("Upload an image of your open fridge, and the app will detect the food items and suggest a recipe!")

# Function to detect ingredients using YOLOv8
def detect_ingredients(image_path):
    # Use relative path or check if model exists
    model_path = "yolov8s_trained_model.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return []
    
    model = YOLO(model_path)  # Updated path
    model.predict(image_path, save=True, visualize=False, conf=0.5, save_txt=True)

    # Path to the detected image and labels
    detected_image_path = 'runs/detect/predict/uploaded_image.jpg'
    labels_file_path = 'runs/detect/predict/labels/uploaded_image.txt'

    # Display the detected image
    detected_image = Image.open(detected_image_path)
    st.image(detected_image, caption="Detected Image", use_column_width=True)

    # Create a list to store unique labels
    unique_labels = []

    # Open and read the labels file
    with open(labels_file_path, 'r') as file:
        lines = file.readlines()

    # Process each line to extract label IDs and names
    for line in lines:
        label_id, *_ = map(float, line.split())
        label_id = int(label_id)
        label_name = model.names[label_id]
        if label_name not in unique_labels:
            unique_labels.append(label_name)

    return unique_labels

# Function to generate a recipe using Gemini API
def generate_recipe(unique_labels, cuisine_type):
    # Configure the Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Create prompts based on cuisine type
    prompt_indian = "Generate a recipe of Indian food using some or all of these ingredients: " + str(unique_labels)
    prompt_western = "Generate a recipe of Western food using some or all of these ingredients: " + str(unique_labels)
    prompt_chinese = "Generate a recipe of Chinese food using some or all of these ingredients: " + str(unique_labels)

    # Choose the input prompt based on user selection
    if cuisine_type == "Indian":
        input_prompt = prompt_indian
    elif cuisine_type == "Western":
        input_prompt = prompt_western
    elif cuisine_type == "Chinese":
        input_prompt = prompt_chinese

    # Generate the recipe using Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(input_prompt)

    return response.text

# Streamlit file uploader
uploaded_image = st.file_uploader("Upload an image of your fridge", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image for YOLO model inference
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # Detect ingredients using the YOLO model
    with st.spinner("Detecting food items..."):
        unique_labels = detect_ingredients(image_path)
    st.success("Detection complete!")
    st.write("Unique ingredients detected:", unique_labels)

    # Radio button selection for cuisine type
    cuisine_type = st.radio(
        "Select the type of recipe you want:",
        ("Indian", "Western", "Chinese")
    )

    # Generate and display the recipe
    if st.button("Generate Recipe"):
        with st.spinner("Generating recipe..."):
            recipe = generate_recipe(unique_labels, cuisine_type)
        st.success("Recipe generated successfully!")
        st.write(recipe)

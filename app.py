import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define the class names (from the notebook)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load your trained model
def load_model(model_path):
    # Initialize the model (same architecture as used in training)
    model = models.vgg16(pretrained=False)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, len(class_names))
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    # Convert Gradio's numpy array to PIL Image
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Apply transformations
    image = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    # Extract plant name and disease status
    top_pred = max(confidences, key=confidences.get)
    plant = top_pred.split('___')[0]
    disease = top_pred.split('___')[1]
    
    return {
        "Plant": plant,
        "Disease": disease,
        "Confidence": confidences[top_pred],
        "All Predictions": confidences
    }

# Load your trained model
model_path = "model/vgg_model_ft.pth"  # Update this path if needed
model = load_model(model_path)

# Create Gradio interface
title = "Plant Disease Classifier"
description = """
Upload an image of a plant leaf to classify its health status. The model can detect diseases across 14 plant types and 38 disease categories.
"""

examples = [
    ["example_images/healthy_apple.jpg"],  # You should provide some example images
    ["example_images/diseased_tomato.jpg"]
]

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Plant Leaf Image"),
    outputs=[
        gr.Label(label="Plant"),
        gr.Label(label="Disease Status"),
        gr.Label(label="Confidence"),
        gr.Label(label="All Predictions")
    ],
    title=title,
    description=description,
    examples=examples,
    allow_flagging="never"
)

iface.launch()
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import numpy as np

# Define your model class (same as during training)
class Plant_Disease_VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.vgg16(pretrained=True)
        for param in list(self.network.features.parameters())[:-5]:
            param.requires_grad = False
        num_ftrs = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(num_ftrs, 38)  # 38 classes

    def forward(self, xb):
        return self.network(xb)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Plant_Disease_VGG16()
model.load_state_dict(torch.load("model/vgg_model_ft.pth", map_location=device))
model.to(device)
model.eval()

# Class labels with plant and disease information
class_labels = [
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

# Enhanced preprocessing
def preprocess_image(image):
    """Add noise reduction, sharpening, and background removal"""
    # Convert to numpy array for processing
    img = np.array(image)
    
    # Simple background removal (assuming leaf is dominant green object)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))  # Green color range
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    img = cv2.bitwise_and(img, img, mask=mask)
    
    # Convert back to PIL
    image = Image.fromarray(img)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def parse_class_label(class_label):
    """Split class label into plant name and disease status"""
    parts = class_label.split('___')
    plant = parts[0].replace('_', ' ').replace(',', '')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else "healthy"
    return plant, disease

def is_healthy_override(image, predicted_class, confidence):
    """Heuristic check for false disease predictions"""
    # If model predicts disease but image looks "too clean", override to healthy
    if "healthy" not in predicted_class and confidence > 0.9:
        # Simple check: count green pixels vs total
        img = np.array(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        green_pixels = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
        green_ratio = np.sum(green_pixels > 0) / (img.shape[0] * img.shape[1])
        
        if green_ratio > 0.7:  # Mostly green leaf with no visible spots
            return True
    return False

# Prediction function with fixes
def predict(image):
    try:
        # Preprocess
        input_tensor = preprocess_image(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            preds = model(input_tensor)
            probabilities = torch.nn.functional.softmax(preds[0], dim=0)
        
        # Get top prediction
        top_prob, top_idx = torch.max(probabilities, 0)
        top_class = class_labels[top_idx.item()]
        plant, disease = parse_class_label(top_class)
        confidence = top_prob.item()
        
        # Apply fixes
        if is_healthy_override(image, top_class, confidence):
            return f"Plant: {plant}\nDisease: healthy (Override: Original prediction '{disease}' had {confidence:.2%} confidence but leaf appears healthy)"
        
        # Confidence thresholding
        if confidence < 0.7:
            return f"Uncertain prediction for {plant} (Confidence: {confidence:.2%})\nPlease upload a clearer image."
        
        return f"Plant: {plant}\nDisease: {disease} (Confidence: {confidence:.2%})"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI with additional instructions
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Textbox(label="Prediction Results"),
    title="Plant Disease Detection (With Error Correction)",
    description="""Upload a clear image of a plant leaf. Tips:
    - Crop to show only the leaf
    - Use even lighting
    - Avoid shadows/reflections""",
    examples=[
        ["examples/healthy_apple.jpg"],
        ["examples/diseased_tomato.jpg"]
    ],
    allow_flagging="manual"
)

if __name__ == "__main__":
    import cv2  # For image processing
    iface.launch()
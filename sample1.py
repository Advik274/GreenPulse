import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from disease_info import get_disease_info
from flask import Flask, render_template
import threading
import socket
from warnings import filterwarnings

# Suppress deprecation warnings
filterwarnings("ignore", category=UserWarning)

# ========== MODEL DEFINITION ==========
class Plant_Disease_VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.network = models.vgg16(weights=weights)
        # Freeze early layers
        for param in list(self.network.features.parameters())[:-5]:
            param.requires_grad = False
        # Modify final layer
        num_ftrs = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(num_ftrs, 38)  # 38 classes

    def forward(self, xb):
        return self.network(xb)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Plant_Disease_VGG16()
model.load_state_dict(torch.load("model/vgg_model_ft.pth", map_location=device))
model.to(device)
model.eval()

# Class labels
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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def parse_class_label(class_label):
    """Extract plant and disease from class label"""
    parts = class_label.split('___')
    plant = parts[0].replace('_', ' ').replace(',', '')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else "healthy"
    return plant, disease

def predict(image):
    """Make prediction on input image"""
    try:
        if image is None:
            return "Error: No image provided"
            
        # Preprocess and predict
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(image)
            probabilities = torch.nn.functional.softmax(preds[0], dim=0)
        
        # Get top prediction
        top_prob, top_idx = torch.max(probabilities, 0)
        class_name = class_labels[top_idx.item()]
        plant, disease = parse_class_label(class_name)
        
        # Get disease info
        disease_info = get_disease_info(plant, disease)
        
        # Format results
        result = f"""
Plant: {plant}
Disease: {disease}

Description:
{disease_info['description']}

Recommended Treatments:
{disease_info['pesticides']}

Application Timing:
{disease_info['timing']}

Prevention Measures:
{disease_info['prevention']}
"""
        return result
        
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# ========== WEB APPLICATION ==========
def find_available_port(start_port):
    """Find next available port from start_port"""
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            port += 1

app = Flask(__name__)

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Analysis Results", lines=20),
    title="GREEN PULSE - Plant Health Analysis",
    description="Upload an image of a plant leaf to detect health issues.",
    examples=[
        ["examples/healthy_apple.jpg"],
        ["examples/diseased_tomato.jpg"]
    ]
)

def run_gradio():
    """Launch Gradio in separate thread"""
    global gradio_port
    gradio_port = find_available_port(7860)
    print(f"\nGradio interface running on port: {gradio_port}")
    iface.launch(
        server_name="0.0.0.0",
        server_port=gradio_port,
        share=False,
        prevent_thread_lock=True
    )

# Start Gradio thread
gradio_port = 7860  # Default
gradio_thread = threading.Thread(target=run_gradio, daemon=True)
gradio_thread.start()

# Flask Routes
@app.route('/')
def home():
    """Main landing page"""
    return render_template("index.html")

@app.route('/analyze')
def analyze():
    """Page with embedded Gradio interface"""
    return render_template("analyze.html", gradio_port=gradio_port)

@app.route('/results')
def results():
    """Results display page"""
    return render_template("results.html")

if __name__ == '__main__':
    """Main application entry point"""
    flask_port = find_available_port(5000)
    print(f"Flask server running on port: {flask_port}")
    print(f"Access the app at: http://localhost:{flask_port}")
    app.run(debug=True, port=flask_port, use_reloader=False)
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from disease_info import get_disease_info

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

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def parse_class_label(class_label):
    """Split class label into plant name and disease status"""
    parts = class_label.split('___')
    plant = parts[0].replace('_', ' ').replace(',', '')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else "healthy"
    return plant, disease

# Prediction function
def predict(image):
    try:
        print("Starting prediction...")
        if image is None:
            return "Error: No image provided"
            
        print("Preprocessing image...")
        image = transform(image).unsqueeze(0).to(device)
        
        print("Running model prediction...")
        with torch.no_grad():
            preds = model(image)
            probabilities = torch.nn.functional.softmax(preds[0], dim=0)
        
        # Get top prediction
        top_prob, top_idx = torch.max(probabilities, 0)
        class_name = class_labels[top_idx.item()]
        plant, disease = parse_class_label(class_name)
        
        print(f"Detected: {plant} - {disease}")
        
        # Get disease information from Mistral API
        print("Fetching disease information...")
        try:
            disease_info = get_disease_info(plant, disease)
            print("Received disease information successfully")
            
            # Format the output
            result = f"""
Plant: {plant}
Disease: {disease}

{disease_info['description']}

Recommended Pesticides:
{disease_info['pesticides']}

Application Timing:
{disease_info['timing']}

Prevention Measures:
{disease_info['prevention']}
"""
            return result
            
        except Exception as e:
            print(f"Error getting disease info: {str(e)}")
            return f"Error getting disease information: {str(e)}"
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return f"Error in prediction: {str(e)}"

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction Results", lines=20),
    title="Plant Disease Detection",
    description="Upload an image of a plant leaf to detect diseases and get detailed information about treatment and prevention.",
    examples=[
        ["examples/healthy_apple.jpg"],
        ["examples/diseased_tomato.jpg"]
    ],
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(share=True)
import gradio as gr
from PIL import Image
import random
import time
import os

# --- Sample Output (Simulated ML Prediction Result) ---
sample_output = {
    "Apple___healthy": {
        "disease_name": "Healthy",
        "crop": "Apple",
        "description": "This Apple leaf shows no signs of disease. The plant appears healthy.",
        "cause": "No disease detected.",
        "prevention": "Continue with good agricultural practices like clean pruning, proper spacing, and pest monitoring.",
        "pesticide": {
            "name": "No pesticide needed",
            "type": "None",
            "timing": "N/A",
            "image_url": "https://yourcdn.com/images/no_pesticide.jpg"
        },
        "sample_images": [
            "dataset/Apple___healthy/image1.jpg",
            "dataset/Apple___healthy/image2.jpg",
            "dataset/Apple___healthy/image3.jpg"
        ],
        "summary_prompt": "The Apple leaf appears healthy. No signs of disease. Maintain good care and monitor regularly."
    }
}

TIPS = [
    "🩴 Always water plants early in the morning to reduce evaporation.",
    "🌞 Keep leaves dry to prevent fungal diseases.",
    "🩹 Clean tools after pruning to stop disease spread.",
    "🎾 Rotate crops every season to maintain soil health.",
    "🪪 Check for pest damage under the leaves too!"
]

def predict_disease(username, location_method, manual_location, gps_coords, image):
    user_location = manual_location if location_method == "Manual Entry" else gps_coords

    # Simulate Prediction
    time.sleep(2)
    predicted_label = "Apple___healthy"
    confidence = 0.94
    result = sample_output.get(predicted_label)

    if not result:
        return "Could not detect disease.", None, None, None, None, None, None, None, None, None

    # Alerts based on location
    alerts = {
        "Punjab": ["Wheat Rust", "Cotton Leaf Curl"],
        "West Bengal": ["Rice Blast", "Bacterial Leaf Blight"],
        "Maharashtra": ["Powdery Mildew", "Leaf Spot"]
    }
    disease_alerts = alerts.get(user_location, ["No major alerts"])

    return (
        f"✅ Prediction Complete: {result['disease_name']} ({result['crop']})", 
        f"{int(confidence * 100)}%", 
        result['description'],
        result['cause'],
        result['prevention'],
        result['pesticide'],
        result['sample_images'],
        random.choice(TIPS),
        user_location,
        ", ".join(disease_alerts)
    )

def dr_green_chat(user_query):
    q = user_query.lower()
    if "apple" in q and "healthy" in q:
        return "An apple leaf with no spots or discoloration is likely healthy. Continue regular monitoring and good practices."
    elif "pesticide" in q:
        return "Choose pesticides based on the specific disease. Always follow recommended guidelines and timings."
    elif "how to use" in q or "guide" in q:
        return "Upload a clear leaf image and click 'Predict Disease'. Ask anything in the chat!"
    else:
        return "I'm Dr. Green 🌿, your plant health assistant! Ask me about diseases, care, or anything green."

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("# 🌱 GREENPULSE - AI-Powered Leaf Disease Detection")

    with gr.Row():
        username = gr.Textbox(label="Username", placeholder="e.g., farmer123")
        location_method = gr.Radio(["Manual Entry", "Detect via GPS"], label="Location Method", value="Manual Entry")

    with gr.Row():
        manual_location = gr.Textbox(label="Manual Location", placeholder="e.g., Punjab")
        gps_coords = gr.Textbox(label="GPS Coordinates", placeholder="e.g., 30.7333,76.7794")

    image = gr.Image(type="filepath", label="Upload Leaf Image")
    predict_btn = gr.Button("🔍 Predict Disease")

    result_msg = gr.Textbox(label="Result")
    confidence = gr.Textbox(label="Health Confidence")
    description = gr.Textbox(label="Description")
    cause = gr.Textbox(label="Cause")
    prevention = gr.Textbox(label="Prevention")
    pesticide_info = gr.Textbox(label="Pesticide Details")
    sample_gallery = gr.Gallery(label="Sample Images", columns=3, rows=1)
    tip = gr.Textbox(label="💡 Daily Tip")
    detected_location = gr.Textbox(label="Detected Location")
    alerts_output = gr.Textbox(label="Disease Alerts")

    predict_btn.click(
        predict_disease,
        inputs=[username, location_method, manual_location, gps_coords, image],
        outputs=[result_msg, confidence, description, cause, prevention, pesticide_info, sample_gallery, tip, detected_location, alerts_output]
    )

    gr.Markdown("---")
    gr.Markdown("## 🧑‍🌾 Ask Dr. Green")
    user_question = gr.Textbox(label="Ask your question")
    dr_response = gr.Textbox(label="Dr. Green Says")
    user_question.change(dr_green_chat, inputs=user_question, outputs=dr_response)

if __name__ == "__main__":
    demo.launch()

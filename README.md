# Plant Disease Detection System

This system uses a VGG16 model to detect plant diseases from leaf images and provides detailed information about the disease, including treatment recommendations and prevention measures using the Mistral AI API.

## Features

- Plant disease detection using VGG16 model
- Detailed disease information including:
  - Disease description
  - Recommended pesticides
  - Pesticide application timing
  - Prevention measures
- User-friendly Gradio interface

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Mistral API key:
   - Create a `.env` file in the project root
   - Add your Mistral API key:
     ```
     MISTRAL_API_KEY=your_api_key_here
     ```

## Usage

1. Run the application:
   ```bash
   python model.py
   ```
2. Open the Gradio interface in your web browser
3. Upload an image of a plant leaf
4. View the disease detection results and detailed information

## Model Information

The system can detect 38 different conditions across multiple plant species, including:
- Apple diseases
- Tomato diseases
- Potato diseases
- Grape diseases
- And more...

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- Mistral API key (required for detailed disease information)
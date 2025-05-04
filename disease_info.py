import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_KEY = "wfGjHNCbHYuLio66x5CGMANSnD8QVXYy"
API_URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_disease_info(plant_name: str, disease_name: str) -> dict:
    """
    Get detailed information about a plant disease using Mistral API
    """
    try:
        # Construct the prompt
        prompt = f"""
        You are an expert in plant pathology. Please provide detailed information about {disease_name} in {plant_name} plants.
        Provide the following information in a structured format:
        1. Disease description
        2. Recommended pesticides (if any)
        3. Pesticide application timing
        4. Prevention measures
        
        Format the response as a JSON object with these keys:
        - description
        - pesticides
        - timing
        - prevention
        
        Example response format:
        {{
            "description": "Detailed description of the disease",
            "pesticides": "List of recommended pesticides",
            "timing": "When to apply pesticides",
            "prevention": "Prevention measures"
        }}
        """
        
        print(f"Querying Mistral API for {plant_name} - {disease_name}")
        
        # Prepare the request payload
        payload = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        print("Making API request...")
        # Make API call
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API Error Response: {response.text}")
            raise Exception(f"API request failed with status {response.status_code}")
        
        response_data = response.json()
        print("Received API response")
        
        # Extract the content from the response
        content = response_data["choices"][0]["message"]["content"]
        print("Extracted content from response")
        
        # Try to parse the content as JSON
        try:
            # First, try to find JSON in the content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = content[json_start:json_end]
                disease_info = json.loads(json_str)
            else:
                disease_info = json.loads(content)
            
            print("Successfully parsed JSON response")
            
            # Ensure all required fields are present and properly formatted
            required_fields = ["description", "pesticides", "timing", "prevention"]
            for field in required_fields:
                if field not in disease_info:
                    disease_info[field] = f"No {field} information available"
                else:
                    # Clean up the field value
                    value = disease_info[field]
                    if isinstance(value, str):
                        # Remove any JSON formatting if present
                        if value.strip().startswith('{') and value.strip().endswith('}'):
                            try:
                                value = json.loads(value)
                                if isinstance(value, dict) and field in value:
                                    disease_info[field] = value[field]
                            except:
                                pass
                        disease_info[field] = value.strip()
            
            return disease_info
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Raw content: {content}")
            # If not valid JSON, use the raw content
            return {
                "description": content,
                "pesticides": "Please check the description for pesticide information",
                "timing": "Please check the description for timing information",
                "prevention": "Please check the description for prevention measures"
            }
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {str(e)}")
        return {
            "description": f"Error retrieving disease information: {str(e)}",
            "pesticides": "Error retrieving pesticide information",
            "timing": "Error retrieving timing information",
            "prevention": "Error retrieving prevention information"
        }
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return {
            "description": f"Unexpected error: {str(e)}",
            "pesticides": "Error retrieving pesticide information",
            "timing": "Error retrieving timing information",
            "prevention": "Error retrieving prevention information"
        } 
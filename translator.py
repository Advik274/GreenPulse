import requests

class Translator:
    def __init__(self):
        self.api_key = "393faf7faf-bf19-46db-a640-c0f44f844724"  # Your Inference API Key Value
        self.base_url = "https://api.basini.com/v1/translate"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def translate(self, text, target_lang):
        try:
            # Split the text into sections for better translation
            sections = text.split('\n\n')
            translated_sections = []
            
            for section in sections:
                if section.strip():
                    payload = {
                        "text": section,
                        "target_lang": target_lang,
                        "source_lang": "en"
                    }
                    response = requests.post(self.base_url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    translated_sections.append(response.json()["translated_text"])
                else:
                    translated_sections.append("")
            
            return "\n\n".join(translated_sections)
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

translator = Translator() 
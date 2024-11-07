import os
import requests
import base64
from PIL import Image
import numpy as np
from IPython.display import display

api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def get_text_description(image_path, api_key):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Use text to simply explain the content of the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        # 解析响应
        response_data = response.json()
        
        text = response_data['choices'][0]['message']['content']
        print(text)
        
        return text
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def get_importance_score(patch_path, description, api_key):
    base64_patch = encode_image(patch_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given a small patch of an image and a textual description of the entire image, give a importance score from 0 to 1 to show the importance of this patch within the context of the whole image. This is the description of the entire image{description}, and only return the score."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_patch}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        
        score = response_data['choices'][0]['message']['content']
        print(score)
        
        return score
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")



def main(image_path, patch_folder):
    text = get_text_description(image_path, api_key)
    # text = "The image shows a man outdoors holding a large fish with both hands. He is dressed in casual outdoor attire, including a hat, and appears to be pleased with his catch. Behind him, there is greenery suggesting a natural, possibly lakeside or riverside, setting."
    scores = []
    path = os.listdir(patch_folder)
    for p in path:
        score = get_importance_score(os.path.join(patch_folder, p), text, api_key)
        scores.append(float(score))
    return os.path.join(patch_folder, path[np.argmax(scores)])

if __name__ == "__main__":
    image_path = "fish.jpg"
    patch_folder = "list/list"
    
    mip = main(image_path, patch_folder)
    print(mip)

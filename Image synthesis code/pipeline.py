import os
import cv2
import requests
import base64
import numpy as np
from scipy.ndimage import zoom
from autogen import ConversableAgent

api_key = os.environ.get("OPENAI_API_KEY")

def melt_image5(
        image1: np.ndarray,
        image2: np.ndarray, 
        coor1: list, 
        coor2: list, 
        label1: str, 
        label2: str) -> tuple:
    """
    label1, label2 are one-hot label
    coor: Tuple(ndarray[x0,y0],ndarray[x1,y1])
    """
    coor1 = np.array(coor1)
    coor2 = np.array(coor2)

    patch_shape1 = coor1[1] - coor1[0]
    patch_shape2 = coor2[1] - coor2[0]
    zoom_factors = (
        patch_shape1[0] / patch_shape2[0],
        patch_shape1[1] / patch_shape2[1],
        1
    )
    # patch_shape1 = np.append(patch_shape1, 3)
    patch2 = image2[coor2[0][0] : coor2[1][0], coor2[0][1] : coor2[1][1], :]
    # patch2 = np.resize(patch2, patch_shape1)
    patch2 = zoom(patch2, (zoom_factors), order=1)
    image1[coor1[0][0] : coor1[1][0], coor1[0][1] : coor1[1][1], :] = patch2

    label = (label1 + label2) / 2
    return image1, label

def base64_to_ndarray(base64_str: str) -> np.ndarray:
    decoded_data = base64.b64decode(base64_str)
    return np.frombuffer(decoded_data, dtype=np.uint8)

def encode_image(image):
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image as JPEG")
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def get_text_description(base64_image, api_key):

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
        return text
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def get_importance_score(base64_patch, description, cls, api_key=api_key):
    
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
                        "text": f"Given a small patch of an image and a textual description of the entire image, give a importance score from 0 to 1 to show the importance of this patch within the context of the whole image. The category of the entire image should be {cls}. This is the description of the entire image{description}, and only return the score."
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
        return score
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def get_important_patch(base64_image: str, 
                        cls: str, 
                        patch_size: tuple, 
                        api_key: str) -> list:

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
                        "text": f"For the given image with category {cls}, provide the positions of the most important patch and the least important patch. The patch should be square. And the size of the square patch should between {patch_size[0]} and {patch_size[1]}. The output format includes eight numbers: the first four represent the top-left and bottom-right coordinates of the most important patch, and the last four represent the top-left and bottom-right coordinates of the least important patch. Make sure all numbers are between 0 and 223. Return only eight numbers separated by spaces and do not return anything else."
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
        
        coordinate = response_data['choices'][0]['message']['content']
        coordinate = np.array(list(map(int, coordinate.split(" ")))).reshape((4, 2))
        coordinate = adjust_coors(coordinate)
        return coordinate
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred in getting patch: {http_err}")
    except Exception as err:
        print(f"Other error occurred in getting patch: {err}")

def get_best_fuse(base64_fuse1: str, 
                  base64_fuse2: str, 
                  base64_fuse3: str, 
                  label1: str, 
                  label2: str, 
                  api_key: str) -> int:

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
                        "text": f"There are three images obtained after blending. Determine which one is the best, considering it should reflect both {label1} and {label2}. Return only one number: 1, 2, or 3. Return only one number and do not return anything else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_fuse1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_fuse2}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_fuse3}"
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
        
        best = response_data['choices'][0]['message']['content']
        best = int(best)
        return best
     
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred in finding best: {http_err}")
    except Exception as err:
        print(f"Other error occurred in finding best: {err}")

def adjust_coors(coors: np.ndarray) -> np.ndarray:
    output_coors = coors.copy()

    min_val, max_val = 0, 223

    for i in range(0, 4, 2):
        x_values = coors[i:i+2, 0] 
        y_values = coors[i:i+2, 1]

        if any(x_values < min_val):
            adjustment = min_val - np.min(x_values)
            output_coors[i:i+2, 0] += adjustment
        elif any(x_values > max_val):
            adjustment = max_val - np.max(x_values)
            output_coors[i:i+2, 0] += adjustment

        if any(y_values < min_val):
            adjustment = min_val - np.min(y_values)
            output_coors[i:i+2, 1] += adjustment
        elif any(y_values > max_val):
            adjustment = max_val - np.max(y_values)
            output_coors[i:i+2, 1] += adjustment

    return output_coors


def main(image1, image2, label1, label2, label1_one_hot, label2_one_hot, api_key):

    base64_image1 = encode_image(image1)
    base64_image2 = encode_image(image2)

    fused_images = []
    coordinates = []

    patch_sizes = [(52, 76), (76, 100), (100, 124)]

    for i in range(3):
        coordinate1 = get_important_patch(base64_image1, label1, patch_size=patch_sizes[i], api_key=api_key)
        coordinate2 = get_important_patch(base64_image2, label2, patch_size=patch_sizes[i], api_key=api_key)
        coordinate1[:, [0,1]] = coordinate1[:, [1, 0]]
        coordinate2[:, [0,1]] = coordinate2[:, [1, 0]]
        fused_image, merged_label = melt_image5(image1.copy(), image2, coordinate1[2:], coordinate2[:2], label1_one_hot, label2_one_hot)
        fused_images.append(fused_image)

        coordinates.append((coordinate1, coordinate2))

    base64_fused_images = [encode_image(fused_image) for fused_image in fused_images]
    index = get_best_fuse(base64_fused_images[0], base64_fused_images[1], base64_fused_images[2], label1=label1, label2=label2, api_key=api_key)
    index = index - 1

    base64_fused_image = [cv2.resize(fi, (28, 28)) for fi in fused_images]
    base64_fused_image = [encode_image(fi) for fi in base64_fused_image]
    small_image1 = cv2.resize(image1, (28, 28))
    small_image2 = cv2.resize(image2, (28, 28))
    index = autogen_framework(encode_image(small_image1), encode_image(small_image2), base64_fused_image, label1, label2, label1_one_hot, label2_one_hot, index)
    index = int(index)

    return fused_images, coordinates[index], merged_label, index

def autogen_framework(base64_image1, base64_image2, fused_images, label1, label2, label1_one_hot, label2_one_one, index):
    fused_image1, fused_image2, fused_image3 = fused_images[0], fused_images[1], fused_images[2]
    patch_sizes = [(52, 76), (76, 100), (100, 124)]
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    
    entrypoint_agent_system_message = (
        "You are an intelligent assistant that fuse two images together. "
        "Your task is to distribute task to subagent and analyse the results. "
        "when you get base64 image as input, you should deliver it to the expected agent."
        "the only thing you can do is distribute task to the expected agent, do not output any result"
    )
    
    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config, 
                                        human_input_mode="NEVER",
                                        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"])
    

    fuse_small_agent_system_message = (
        "You are a patch extraction agent responsible for finding the coordinates of the most and least important patch "
        "The only thing you can return is receive message and return output"
        f"You should only return {fused_image1}, no more words"
    )
    fuse_small_agent = ConversableAgent("fuse_small_agent", 
                                        system_message=fuse_small_agent_system_message, 
                                        llm_config=llm_config)

    fuse_medium_agent_system_message = (
        "You are a patch extraction agent responsible for finding the coordinates of the most and least important patch "
        "The only thing you can return is receive message and return output"
        f"You should only return {fused_image2}, no more words"
    )
    fuse_medium_agent = ConversableAgent("fuse_medium_agent", 
                                             system_message=fuse_medium_agent_system_message, 
                                             llm_config=llm_config)

    fuse_large_agent_system_message = (
        "You are a patch extraction agent responsible for finding the coordinates of the most and least important patch "
        "The only thing you can return is receive message and return output"
        f"You should only return {fused_image3}, no more words"
    )
    fuse_large_agent = ConversableAgent("fuse_large_agent", 
                                     system_message=fuse_large_agent_system_message, 
                                     llm_config=llm_config)

    scoring_agent_system_message = (
        "You are a scoring agent responsible for finding the coordinates of the most and least important patch "
        "The only thing you can return is receive message and return output"
        f"You should only return {index}, no more words"
    )
    scoring_agent = ConversableAgent("scoring_agent", 
                                        system_message=scoring_agent_system_message, 
                                        llm_config=llm_config)
    
    chat_results = entrypoint_agent.initiate_chats(
    [
        {
            "recipient": entrypoint_agent,  
            "message": f"The two images to be fused {(base64_image1, base64_image2)}, two labels corresponding to images{label1, label2}, patch_sizes{patch_sizes}, the api_key is {api_key}",  
            "max_turns": 1,
            "summary_method": "last_msg"
        },
        {
            "recipient": fuse_small_agent, 
            "message": f"given the {(base64_image1, base64_image2)}, two labels{(label1, label2)} and patch size{patch_sizes[0]} and api_key{api_key}, return the result. You should only return {fused_image1}, no more words",
            "max_turns": 1, 
        },
        {
            "recipient": fuse_medium_agent, 
            "message": f"given the {(base64_image1, base64_image2)}, two labels{(label1, label2)} and patch size{patch_sizes[1]} and api_key{api_key}, return the result. You should only return {fused_image2}, no more words", 
            "max_turns": 1,
        },
        {
            "recipient": fuse_large_agent,
            "message": f"given the {(base64_image1, base64_image2)}, two labels{(label1, label2)} and patch size{patch_sizes[2]} and api_key{api_key}, return the result. You should only return {fused_image2}, no more words",
            "max_turns": 1,
        },
        {
            "recipient": scoring_agent,
            "message": f"given the {(fused_image1, fused_image2, fused_image3)}, two labels{(label1, label2)}and api_key{api_key}, return the result. You should only return {index}, no more words",
            "max_turns": 1,
        }
    ])

    with open("chat.txt", 'w') as f:
        f.write(str(chat_results[-1].chat_history))

    return chat_results[-1].chat_history[-1]["content"]

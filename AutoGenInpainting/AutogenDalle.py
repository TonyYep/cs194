import os
import re
from typing import Dict, Optional

from IPython.display import display
from PIL.Image import Image

import autogen
from autogen.agentchat.contrib import img_utils
from autogen.agentchat.contrib.capabilities import generate_images
from autogen.cache import Cache
from autogen.oai import openai_utils

import inpaint_images

import json
import numpy as np

image_info_test2017 = json.load(open("annotations/image_info_test2017.json"))

# Learning Batch
learning_batch = np.random.choice(image_info_test2017["images"], size=20, replace=False)

gpt_config = {
    "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}],
    "timeout": 120,
    "temperature": 0.7,
}
gpt_vision_config = {
    "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}],
    "timeout": 120,
    "temperature": 0.7,
}
dalle_config = {
    "config_list": [{"model": "dall-e-2", "api_key": os.environ["OPENAI_API_KEY"]}],
    "timeout": 120,
    "temperature": 0.7,
}

CRITIC_SYSTEM_MESSAGE = """You need to help improve the image you saw. 
As an critic agent, you will first be given a batch of images to learn and analyze in general. Focus on what objects, scenes, and details are present in those images and how. 
After that, say BEGIN_EDIT to ask for the image you will work on.
Then, you will be shown with the image needed to improve through directing the Dalle model to do image inpainting. To improve the image, you need to analyze how you think the image should be improved in terms of enhancing the overall quality and values of the image batch. You must give clear direction of inpainting. 
You answer must include the best location to draw the mask box (left, top, right, bottom) in pixels, where the image will be inpainted, and the prompt for dalle model to generate. Note the prompt should describe the full new image, not just the inpainted area.
The mask box should be narrow enough and focused on the detail area. It should not be too broad. 
For example, your response should be like:

The image is showing..., but need to improve.... Hence, the mask box will have left-top corner at (left: 103, top: 245) and right-bottom corner at (right: 392, bottom: 532). The prompt is: ...

The first element is the X coordinate along the horizontal line from left to right. The second element is the Y coordinate along the verticle from top to down. The origin is the left-top corner of the image. Be cautious! Make sure the mask box is always within the range of the given image size.
If you have no critique or further advise for improvement, just say TERMINATE
"""


def _is_termination_message(msg) -> bool:
    # Detects if we should terminate the conversation
    if isinstance(msg.get("content"), str):
        return msg["content"].rstrip().endswith("TERMINATE")
    elif isinstance(msg.get("content"), list):
        for content in msg["content"]:
            if isinstance(content, dict) and "text" in content:
                return content["text"].rstrip().endswith("TERMINATE")
    return False


def critic_agent() -> autogen.ConversableAgent:
    return autogen.ConversableAgent(
        name="critic",
        llm_config=gpt_vision_config,
        system_message=CRITIC_SYSTEM_MESSAGE,
        max_consecutive_auto_reply=3,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: _is_termination_message(msg),
    )


def image_editing_agent(working_image_url: str) -> autogen.ConversableAgent:
    # Create the agent
    agent = autogen.ConversableAgent(
        name="dalle",
        llm_config=gpt_vision_config,
        max_consecutive_auto_reply=3,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: _is_termination_message(msg),
    )

    # Add image generation ability to the agent
    dalle_gen = inpaint_images.DalleImageInpainter(llm_config=dalle_config)
    image_gen_capability = inpaint_images.ImageEditing(
        image_generator=dalle_gen, 
        working_image_url=working_image_url,
        text_analyzer_llm_config=gpt_config
    )

    image_gen_capability.add_to_agent(agent)
    return agent

learning_batch_url = [image["coco_url"] for image in learning_batch]

# Iterate over images
for url in learning_batch_url:

    # Begin Autogen
    dalle = image_editing_agent(url)
    critic = critic_agent()

    initial_msg = {
      "role": "user",
      "content": [{"type": "text", "text": "These are the image. Learn and Analyze. You will be working on improving one of them. Say BEGIN_EDIT"}] + 
        [{
          "type": "image_url",
          "image_url": {
            "url": img,
          },
        } for img in learning_batch_url]
    }

    result = dalle.initiate_chat(critic, message=initial_msg)
    
    input(f"Now finished {url}")
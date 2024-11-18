from typing import List, Dict
from autogen import ConversableAgent
import os
import json
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

image_description_system_message = "This agent provides a description of the image content."
scoring_agent_system_message = "This agent calculates the importance score of a given patch based on its metadata. The score is between 0 and 10."

def main(image_path: str, patches: str):
    llm_config = {"config_list": [{"model": "gpt-4-turbo", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    entrypoint_agent = ConversableAgent("entrypoint_agent", system_message="", llm_config=llm_config)

    image_description_agent = ConversableAgent("patch_extraction_agent", 
                                               system_message=image_description_system_message,  
                                               llm_config=llm_config)
    scoring_agent = ConversableAgent("scoring_agent", 
                                     system_message=scoring_agent_system_message, 
                                     llm_config=llm_config)

    image_description_result = entrypoint_agent.initiate_chats(
        [
            {
                "recipient": image_description_agent,
                "message": f"Please use text to simply explain the content of the image at {image_path}.",
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )
    description = image_description_result[0].chat_history[-1]["content"]

    patch_scores = {}
    for patch in patches:
        scoring_result = entrypoint_agent.initiate_chats(
            [
                {
                    "recipient": scoring_agent,
                    "message": f"Given a small patch of an image and a textual description of the entire image, give a importance score from 0 to 1 to show the importance of this patch within the context of the whole image. This is the description of the entire image{description}, and this is the url of the patch{patch}. Only return the score.",
                    "max_turns": 1,
                    "summary_method": "last_msg",
                }
            ]
        )
    score = scoring_result[0].chat_history[-1]["content"]


    return score


if __name__ == "__main__":
    image_path = ""  # Replace with actual image url
    patches_path = [] # Replace with actual urls of patch paths

    score = main(image_path, patches_path)
    print(score)

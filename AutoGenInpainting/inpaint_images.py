import re
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

from openai import OpenAI
from PIL.Image import Image

from autogen import Agent, ConversableAgent, code_utils
from autogen.agentchat.contrib import img_utils
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.cache import AbstractCache

import image_processor

SYSTEM_MESSAGE = "You've been given the special ability to generate images inpainting. You will receive improving advice about how to generate images. A good advice should tell you where on the image to edit and what prompt should be used for generation. Ask for clarification if you think the advice is not clear."
DESCRIPTION_MESSAGE = "This agent has the ability to generate images inpainting."

PROMPT_INSTRUCTIONS = """Extract the locations of the top-left corner and the right-bottom corner of the mask box (in pixels), and the prompt for generation from TEXT. For the prompt, DO NOT include any advice, use descriptive phrases like "Blue background, 3D shapes,...". Your response should be a line strictly following this format:
left; top; right; bottom; prompt
For example:
120; 224; 400; 384; Blue background, 3D shapes,...
"""


class ImageInpainter(Protocol):
    """This class defines an interface for image generators.

    Concrete implementations of this protocol must provide a `generate_image` method that takes a string prompt as
    input and returns a PIL Image object.

    NOTE: Current implementation does not allow you to edit a previously existing image.
    """

    def generate_image(self, prompt: str, src_img: str, mask: List[int]) -> Image:
        """Generates an image based on the provided prompt.

        Args:
          prompt: A string describing the desired image.

        Returns:
          A PIL Image object representing the generated image.

        Raises:
          ValueError: If the image generation fails.
        """
        ...

    def cache_key(self, prompt: str, src_img: str, mask: List[int]) -> str:
        """Generates a unique cache key for the given prompt.

        This key can be used to store and retrieve generated images based on the prompt.

        Args:
          prompt: A string describing the desired image.

        Returns:
          A unique string that can be used as a cache key.
        """
        ...


class DalleImageInpainter(ImageInpainter):
    """Generates images using OpenAI's DALL-E models.

    This class provides a convenient interface for generating images based on textual prompts using OpenAI's DALL-E
    models. It allows you to specify the DALL-E model, resolution, quality, and the number of images to generate.

    Note: Current implementation does not allow you to edit a previously existing image.
    """

    def __init__(
        self,
        llm_config: Dict,
        resolution: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "512x512",
        quality: Literal["standard", "hd"] = "standard",
        num_images: int = 1,
    ):
        """
        Args:
            llm_config (dict): llm config, must contain a valid dalle model and OpenAI API key in config_list.
            resolution (str): The resolution of the image you want to generate. Must be one of "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792".
            quality (str): The quality of the image you want to generate. Must be one of "standard", "hd".
            num_images (int): The number of images to generate.
        """
        config_list = llm_config["config_list"]
        _validate_dalle_model(config_list[0]["model"])
        _validate_resolution_format(resolution)

        self._model = config_list[0]["model"]
        self._resolution = resolution
        self._quality = quality
        self._num_images = num_images
        self._dalle_client = OpenAI(api_key=config_list[0]["api_key"])

    def generate_image(self, prompt: str, src_img: str, mask: List[int]) -> Image:

        src_img_bytes, mask_bytes, crop_box = image_processor.get_img_and_mask(src_img, mask)

        response = self._dalle_client.images.edit(
            model=self._model,
            image=src_img_bytes.getvalue(),
            mask=mask_bytes.getvalue(),
            prompt=prompt,
            size=self._resolution,
            n=self._num_images,
        )

        image_url = response.data[0].url
        if image_url is None:
            raise ValueError("Failed to generate image.")


        return image_processor.restore_original_ratio(src_img, image_url, crop_box)

    def cache_key(self, prompt: str, src_img: str, mask: List[int]) -> str:
        keys = (prompt, src_img, mask, self._model, self._resolution, self._num_images)
        return ",".join([str(k) for k in keys])


class ImageEditing(AgentCapability):
    """This capability allows a ConversableAgent to generate images based on the message received from other Agents.

    1. Utilizes a TextAnalyzerAgent to analyze incoming messages to identify requests for image generation and
        extract relevant details.
    2. Leverages the provided ImageGenerator (e.g., DalleImageGenerator) to create the image.
    3. Optionally caches generated images for faster retrieval in future conversations.

    NOTE: This capability increases the token usage of the agent, as it uses TextAnalyzerAgent to analyze every
        message received by the agent.

    Example:
        ```python
        import autogen
        from autogen.agentchat.contrib.capabilities.image_generation import ImageGeneration

        # Assuming you have llm configs configured for the LLMs you want to use and Dalle.
        # Create the agent
        agent = autogen.ConversableAgent(
            name="dalle", llm_config={...}, max_consecutive_auto_reply=3, human_input_mode="NEVER"
        )

        # Create an ImageGenerator with desired settings
        dalle_gen = generate_images.DalleImageGenerator(llm_config={...})

        # Add the ImageGeneration capability to the agent
        agent.add_capability(ImageGeneration(image_generator=dalle_gen))
        ```
    """

    def __init__(
        self,
        image_generator: ImageInpainter,
        working_image_url: str,
        working_image: Optional[Image] = None,
        cache: Optional[AbstractCache] = None,
        text_analyzer_llm_config: Optional[Dict] = None,
        text_analyzer_instructions: str = PROMPT_INSTRUCTIONS,
        verbosity: int = 0,
        register_reply_position: int = 2,
    ):
        """
        Args:
            image_generator (ImageGenerator): The image generator you would like to use to generate images.
            cache (None or AbstractCache): The cache client to use to store and retrieve generated images. If None,
                no caching will be used.
            text_analyzer_llm_config (Dict or None): The LLM config for the text analyzer. If None, the LLM config will
                be retrieved from the agent you're adding the ability to.
            text_analyzer_instructions (str): Instructions provided to the TextAnalyzerAgent used to analyze
                incoming messages and extract the prompt for image generation. The default instructions focus on
                summarizing the prompt. You can customize the instructions to achieve more granular control over prompt
                extraction.
                Example: 'Extract specific details from the message, like desired objects, styles, or backgrounds.'
            verbosity (int): The verbosity level. Defaults to 0 and must be greater than or equal to 0. The text
                analyzer llm calls will be silent if verbosity is less than 2.
            register_reply_position (int): The position of the reply function in the agent's list of reply functions.
                This capability registers a new reply function to handle messages with image generation requests.
                Defaults to 2 to place it after the check termination and human reply for a ConversableAgent.
        """
        self._image_generator = image_generator
        self._cache = cache
        self._text_analyzer_llm_config = text_analyzer_llm_config
        self._text_analyzer_instructions = text_analyzer_instructions
        self._verbosity = verbosity
        self._register_reply_position = register_reply_position

        self._agent: Optional[ConversableAgent] = None
        self._text_analyzer: Optional[TextAnalyzerAgent] = None

        self._working_image_url = working_image_url
        self._working_image = working_image
        if working_image is None:
            self._working_image = img_utils.get_pil_image(working_image_url)
        self._working_image_size = self._working_image.size

    def add_to_agent(self, agent: ConversableAgent):
        """Adds the Image Generation capability to the specified ConversableAgent.

        This function performs the following modifications to the agent:

        1. Registers a reply function: A new reply function is registered with the agent to handle messages that
           potentially request image generation. This function analyzes the message and triggers image generation if
           necessary.
        2. Creates an Agent (TextAnalyzerAgent): This is used to analyze messages for image generation requirements.
        3. Updates System Message: The agent's system message is updated to include a message indicating the
           capability to generate images has been added.
        4. Updates Description: The agent's description is updated to reflect the addition of the Image Generation
           capability. This might be helpful in certain use cases, like group chats.

        Args:
          agent (ConversableAgent): The ConversableAgent to add the capability to.
        """
        self._agent = agent

        agent.register_reply([Agent, None], self._image_gen_reply, position=self._register_reply_position)

        self._text_analyzer_llm_config = self._text_analyzer_llm_config or agent.llm_config
        self._text_analyzer = TextAnalyzerAgent(llm_config=self._text_analyzer_llm_config)

        agent.update_system_message(agent.system_message + "\n" + SYSTEM_MESSAGE)
        agent.description += "\n" + DESCRIPTION_MESSAGE

    def _image_gen_reply(
        self,
        recipient: ConversableAgent,
        messages: Optional[List[Dict]],
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        if messages is None:
            return False, None

        last_message = code_utils.content_str(messages[-1]["content"])
        images_in_chat = image_processor.extract_images_from_chat(messages)

        if not last_message:
            return False, None
        
        if "BEGIN_EDIT" in last_message:
            return True, {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"The image is of the size {self._working_image_size}"},
                    {"type": "image_url", "image_url": {"url": self._working_image_url}},
                ]
            }
        
        if not images_in_chat:
            return False, None
        
        latest_image_version = images_in_chat[0]

        if self._should_generate_image(last_message):
            instr = self._extract_inpainting_instruction(last_message)

            if instr is None:
                return False, None
            
            left,top,right,bottom,prompt = instr

            if left < 0 or top < 0 or right > self._working_image_size[0] or bottom > self._working_image_size[1]:
                return False, None
            
            mask = [left, top, right, bottom]

            image = self._cache_get(prompt, latest_image_version, mask)
            if image is None:
                image = self._image_generator.generate_image(prompt, latest_image_version, mask)
                self._cache_set(prompt, latest_image_version, mask, image)

            return True, self._generate_content_message(prompt, mask, image)

        else:
            return False, None

    def _should_generate_image(self, message: str) -> bool:
        assert self._text_analyzer is not None

        instructions = """
        Does the TEXT ask the agent to improve the image?
        The TEXT must explicitly give the coordinates of the editing box (left, top, right, bottom) in pixels, the prompt for generating editing. Otherwise, don't do image editing.
        Answer with just one word, yes or no.
        """
        analysis = self._text_analyzer.analyze_text(message, instructions)

        return "yes" in self._extract_analysis(analysis).lower()

    def _extract_inpainting_instruction(self, last_message) -> Tuple[int, int, int, int, str]:
        assert self._text_analyzer is not None

        analysis = self._text_analyzer.analyze_text(last_message, self._text_analyzer_instructions)
        instruction = self._extract_analysis(analysis)
        try:
            instruction = instruction.split(';')
            return int(instruction[0]), int(instruction[1]), int(instruction[2]), int(instruction[3]), instruction[-1]
        except:
            return None

    def _cache_get(self, prompt: str, src_img: str, mask: List[str]) -> Optional[Image]:
        if self._cache:
            key = self._image_generator.cache_key(prompt, src_img, mask)
            cached_value = self._cache.get(key)

            if cached_value:
                return img_utils.get_pil_image(cached_value)

    def _cache_set(self, prompt: str, src_img: str, mask: List[str], image: Image):
        if self._cache:
            key = self._image_generator.cache_key(prompt, src_img, mask)
            self._cache.set(key, img_utils.pil_to_data_uri(image))

    def _extract_analysis(self, analysis: Union[str, Dict, None]) -> str:
        if isinstance(analysis, Dict):
            return code_utils.content_str(analysis["content"])
        else:
            return code_utils.content_str(analysis)

    def _generate_content_message(self, prompt: str, mask: List[int], image: Image) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"I edited the image with the prompt: {prompt} within the region bounded by {mask}."},
                {"type": "image_url", "image_url": {"url": img_utils.pil_to_data_uri(image)}},
            ]
        }


### Helpers
def _validate_resolution_format(resolution: str):
    """Checks if a string is in a valid resolution format (e.g., "1024x768")."""
    pattern = r"^\d+x\d+$"  # Matches a pattern of digits, "x", and digits
    matched_resolution = re.match(pattern, resolution)
    if matched_resolution is None:
        raise ValueError(f"Invalid resolution format: {resolution}")


def _validate_dalle_model(model: str):
    if model not in ["dall-e-3", "dall-e-2"]:
        raise ValueError(f"Invalid DALL-E model: {model}. Must be 'dall-e-3' or 'dall-e-2'")
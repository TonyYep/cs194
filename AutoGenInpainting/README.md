# Image Inpainting with DALL-E Enabled Conversable Agent

This repository contains a collection of scripts to perform **automatic image inpainting** using OpenAI's **DALL·E** models and Autogen frameworks. It facilitates the generation and improvement of images through inpainting workflows while leveraging conversational agents for analysis and enhancement.

---

## **Files Overview**

### 1. **AutogenDalle.py**
   - Integrates **Autogen agents** and OpenAI's DALL·E model.
   - Performs image generation and inpainting using prompts and automated analysis.
   - Facilitates collaboration between agents for improving generated images.

### 2. **image_processor.py**
   - Handles image preprocessing tasks:
     - Fetches and crops images.
     - Resizes them to a standard size for DALL·E input.
   - Extracts masks and regions to focus on for inpainting.

### 3. **inpaint_images.py**
   - Core implementation of **image inpainting** using OpenAI's DALL·E:
     - Defines an `ImageInpainter` base interface.
     - Implements `DalleImageInpainter` for OpenAI DALL·E models.
     - Uses caching for efficient retrieval of previously generated images.
   - Integrates **multi-agent workflows** to automate prompt analysis and generation tasks.

---

## **Steps to Run**

### 1. **Set Up the Environment**

   Clone the repository:
   ```bash
   git clone https://github.com/TonyYep/cs194.git
   cd cs194/AutoGenInpainting
   ```

   Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### 2. **Set Up OpenAI API Key**

Ensure you have an OpenAI API key. Export it as an environment variable:
```bash
export OPENAI_API_KEY='your_openai_api_key'
```

Alternatively, include the key in the scripts.

---

### 3. **Run the Inpainting Script**

To perform image inpainting:
```bash
python AutogenDalle.py
```

This script automatically chooses a batch of images and the image to be edited. The URL for processed image is shown at the end of each iteration. 

---


## **Dependencies**
- Python 3.11 or above
- OpenAI API
- Pillow
- Autogen

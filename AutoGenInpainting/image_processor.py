from io import BytesIO
from typing import List, Tuple, Any
from PIL import Image
from autogen.agentchat.contrib import img_utils

def get_img_and_mask(url: str, maskbox: List[int]) -> Tuple[BytesIO, BytesIO, Tuple]:
    assert len(maskbox) == 4 and maskbox[2] >= maskbox[0] and maskbox[3] >= maskbox[1]
    center_x = (maskbox[2] + maskbox[0])//2
    center_y = (maskbox[3] + maskbox[1])//2
    DALLESIZE = 512


    image = img_utils.get_pil_image(url)

    width, height = image.size
            
    # Determine the size of the largest square that fits within the image
    square_size = min(width, height)

    # Calculate cropping coordinates to center the square around the given position
    left = max(0, center_x - square_size // 2)
    top = max(0, center_y - square_size // 2)
    right = left + square_size
    bottom = top + square_size

    # Ensure the cropping box fits within the image boundaries
    if right > width:
        left = width - square_size
        right = width
    if bottom > height:
        top = height - square_size
        bottom = height

    crop_box = (left, top, right, bottom)

    # Crop the image
    cropped_image = image.crop(crop_box)

    resized_image = cropped_image.resize((DALLESIZE, DALLESIZE))


    mask = Image.new('RGBA', (width, height), (0, 0, 0, 255))

    for x in range(maskbox[0], maskbox[2]):
        for y in range(maskbox[1], maskbox[3]):
            mask.putpixel((x, y), (0, 0, 0, 0))  # Opaque (non-transparent)

    cropped_mask = mask.crop(crop_box)
    resized_mask = cropped_mask.resize((DALLESIZE, DALLESIZE))


    # This is the BytesIO object that contains your image data
    src_img_bytes: BytesIO = BytesIO()
    mask_bytes: BytesIO = BytesIO()
    resized_image.save(src_img_bytes, format='PNG')
    resized_mask.save(mask_bytes, format='PNG')
    src_img_bytes.seek(0)
    mask_bytes.seek(0)

    return src_img_bytes, mask_bytes, crop_box

def restore_original_ratio(original_url: str, cropped_url: str, crop_box: Tuple):
    cropsize = crop_box[2] - crop_box[0]
    image = img_utils.get_pil_image(cropped_url)
    image = image.resize((cropsize, cropsize))

    origin = img_utils.get_pil_image(original_url)
    origin.paste(image, (crop_box[0], crop_box[1]))

    return origin


def extract_images_from_chat(messages: List[Any], by_url = True):
    images = []

    for message in reversed(messages):
        # The GPT-4V format, where the content is an array of data
        contents = message.get("content", [])
        for content in contents:
            if isinstance(content, str):
                continue
            if content.get("type", "") == "image_url":
                img_data = content["image_url"]["url"]
                if by_url:
                    images.append(img_data)
                else:
                    images.append(img_utils.get_pil_image(img_data))

    return images
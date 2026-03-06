from pathlib import Path
import base64
import json
import cv2
import time
import numpy as np
from openai import OpenAI
from utils.general import strip_code_fences


def get_response(messages, text_format=None):

    client = OpenAI()
    if text_format is None:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
        )
        return response.choices[0].message.content
    response = client.chat.completions.parse(
        model="gemini-2.5-flash",
        messages=messages,
        response_format=text_format,
    )
    message = response.choices[0].message
    if message.parsed is not None:
        return message.parsed
    # Fallback if provider doesn't set parsed (e.g. refusal or raw content)
    if message.content:
        data = json.loads(strip_code_fences(message.content))
        return text_format.model_validate(data)
    raise ValueError("No parsed output and no content in response")

def generate_messages(images, prompt, max_frames=None):
    """
    Build messages from images (numpy arrays) or image paths.
    Args:
        images: np.ndarray, path, directory, or iterable of these
        prompt: text prompt
        max_frames: If set, use at most this many frames (evenly subsampled). Helps avoid 503 when folder has many images.
    """
    # Normalize to list
    if isinstance(images, (str, Path, np.ndarray)):
        images = [images]

    # Collect image arrays (BGR)
    imgs = []
    for item in images:
        if isinstance(item, np.ndarray):
            imgs.append(item)
        else:
            p = Path(item)
            if p.is_dir():
                paths = sorted([x for x in p.iterdir() if x.suffix.lower() in [".jpg", ".jpeg"]])
                for img_path in paths:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    imgs.append(img)
            else:
                img = cv2.imread(str(p))
                if img is None:
                    raise ValueError(f"Could not read image: {p}")
                imgs.append(img)

    if not imgs:
        raise ValueError("No images provided.")

    # Use only half the images
    if len(imgs) > 1:
        imgs = imgs[::2]

    if max_frames is not None and len(imgs) > max_frames:
        step = len(imgs) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        imgs = [imgs[i] for i in indices]

    # Encode images to base64
    base64Frames = []
    for img in imgs:
        success, buffer = cv2.imencode(".jpg", img)
        if not success:
            raise ValueError("Failed to encode image array to JPG.")
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    content = [
        {
            "type": "text",
            "text": prompt
        },
        *[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            }
            for frame in base64Frames
        ]
    ]

    messages = [{
        "role": "user",
        "content": content
    }]
    return messages


if __name__ == "__main__":
    start_time = time.time()

    from prompts import prompt_generate_episodic_memory, character_matching_information
    
    messages = generate_messages("../data/frames/bedroom_01_10min", character_matching_information + prompt_generate_episodic_memory)
    response = get_response(messages)
    print(response)
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time} seconds")

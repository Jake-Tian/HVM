from pathlib import Path
import base64
import cv2
import time
import numpy as np
from openai import OpenAI
from typing import Optional

from utils.token_monitor import TokenMonitor


def get_response(messages, token_monitor: Optional[TokenMonitor] = None):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
    )
    content = response.choices[0].message.content
    if token_monitor is not None and response.usage:
        token_monitor.add_vision_usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    return content


def generate_messages(images, prompt):
    """
    Build messages from images (numpy arrays) or image paths.
    Args:
        images: np.ndarray, path, directory, or iterable of these
        prompt: text prompt
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

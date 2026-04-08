from pathlib import Path
import base64
import cv2
import time
import numpy as np
from openai import OpenAI

def get_response(messages, text_format=None):
    client = OpenAI()
    if text_format is None:
        response = client.responses.create(
            model="gpt-5-mini",
            input=messages,
        )
        return response.output_text, getattr(response.usage, "total_tokens", None) or 0
    else:
        response = client.responses.parse(
            model="gpt-5-mini",
            input=messages,
            text_format=text_format,
        )
        return response.output_parsed, getattr(response.usage, "total_tokens", None) or 0


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
            "type": "input_text",
            "text": prompt
        },
        *[
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame}"
            }
            for frame in base64Frames
        ]
    ]

    messages = [{
        "role": "user",
        "content": content
    }]
    return messages


def generate_audio_messages(audio_path, prompt):
    """
    Build messages from a single audio file.
    Args:
        audio_path: path to the .wav file
        prompt: text prompt
    """
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    base64_audio = base64.b64encode(audio_data).decode("utf-8")

    content = [
        {
            "type": "input_text",
            "text": prompt
        },
        {
            "type": "input_audio",
            "audio": {
                "data": base64_audio,
                "format": "wav"
            }
        }
    ]

    messages = [{
        "role": "user",
        "content": content
    }]
    return messages

"""Multimodal LLM via Qwen-VL's OpenAI-compatible endpoint."""

import base64
from pathlib import Path

import cv2
import numpy as np

from utils.llm_qwen import _client

MODEL = "qwen3-vl-flash"


def get_response(messages, text_format=None):
    client = _client()
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "extra_body": {"enable_thinking": False},
    }
    if text_format is None:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content, response.usage.total_tokens

    response = client.chat.completions.parse(
        **kwargs,
        response_format=text_format,
    )
    return response.choices[0].message.parsed, response.usage.total_tokens


def generate_messages(images, prompt):
    """Build OpenAI-style multimodal messages from images and a prompt."""
    if isinstance(images, (str, Path, np.ndarray)):
        images = [images]

    frames = []
    for item in images:
        if isinstance(item, np.ndarray):
            frames.append(item)
            continue
        path = Path(item)
        image_paths = (
            sorted(x for x in path.iterdir() if x.suffix.lower() in {".jpg", ".jpeg"})
            if path.is_dir()
            else [path]
        )
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            frames.append(image)

    if not frames:
        raise ValueError("No images provided.")

    encoded_frames = []
    for image in frames:
        height, width = image.shape[:2]
        image = cv2.resize(
            image,
            (width // 2, height // 2),
            interpolation=cv2.INTER_AREA,
        )
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image array to JPG.")
        encoded_frames.append(base64.b64encode(buffer).decode("utf-8"))

    content = [
        {"type": "text", "text": prompt},
        *[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
            for frame in encoded_frames
        ],
    ]
    return [{"role": "user", "content": content}]

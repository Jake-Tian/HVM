"""Multimodal LLM via OpenAI GPT.

API key and base URL are read from OPENAI_API_KEY / OPENAI_BASE_URL env vars
by default.
"""

import os
import sys
from pathlib import Path
import base64
import cv2
import numpy as np
from openai import OpenAI
from utils.llm_gpt import MODEL
from utils.token_usage import usage_from_response


def _client(api_key=None, base_url=None):
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
    )


def _report_llm_failure(model, exc):
    """Print LLM API failures to stderr (visible even when stdout is quiet-logged)."""
    try:
        sys.stderr.write(f"[LLM ERROR] model={model}: {exc}\n")
        sys.stderr.flush()
    except Exception:
        pass


def get_response(messages, text_format=None, model=None,
                  api_key=None, base_url=None, temperature=None):
    """Call a GPT multimodal LLM. Returns (content, token_usage) or (parsed, token_usage)."""
    model = model or MODEL

    client = _client(api_key, base_url)
    create_kwargs = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature

    try:
        if text_format is None:
            response = client.chat.completions.create(**create_kwargs)
            return (
                response.choices[0].message.content,
                usage_from_response(response, model, "openai"),
            )
        response = client.chat.completions.parse(
            **create_kwargs, response_format=text_format,
        )
        return (
            response.choices[0].message.parsed,
            usage_from_response(response, model, "openai"),
        )
    except Exception as e:
        _report_llm_failure(model, e)
        raise


def generate_messages(images, prompt):
    """Build OpenAI-style multimodal messages from images + a text prompt.

    images: np.ndarray, path, directory, or iterable of these.
    Images are resized to half resolution to stay within request size limits.
    """
    if isinstance(images, (str, Path, np.ndarray)):
        images = [images]

    imgs = []
    for item in images:
        if isinstance(item, np.ndarray):
            imgs.append(item)
            continue
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

    base64_frames = []
    for img in imgs:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(".jpg", img)
        if not success:
            raise ValueError("Failed to encode image array to JPG.")
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    content = [
        {"type": "text", "text": prompt},
        *[
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
            for frame in base64_frames
        ],
    ]
    return [{"role": "user", "content": content}]

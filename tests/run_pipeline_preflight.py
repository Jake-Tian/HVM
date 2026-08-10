#!/usr/bin/env python3
"""Run network, model, storage, and OpenCV checks before run_video_pipeline.sh."""

import importlib
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BLAS_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
ORIGINAL_BLAS_THREADS = {
    variable: os.environ.get(variable) for variable in BLAS_THREAD_VARIABLES
}

# Keep numerical libraries from exhausting the per-user thread allowance.
for variable in BLAS_THREAD_VARIABLES:
    os.environ[variable] = "1"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_required_files(context):
    required = [
        "run_video_pipeline.sh",
        "video_list.txt",
        "process_full_video.py",
        "reason.py",
        "abstraction_ablation.py",
        "preprocessing/download_hf_folder.py",
        "preprocessing/download_hf_videos.py",
        "preprocessing/add_subtitles_and_extract_frames.py",
        "configs/abs_30_10.json",
        "configs/abs_50_30.json",
        "configs/abs_100_60.json",
        "utils/embedding.py",
        "utils/llm_gpt.py",
        "utils/mllm_gpt.py",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing files: {', '.join(missing)}")
    return f"{len(required)} required files found"


def check_dependencies(context):
    modules = [
        "cv2",
        "huggingface_hub",
        "langchain_core",
        "langgraph",
        "numpy",
        "openai",
        "pydantic",
        "requests",
        "tqdm",
    ]
    for module in modules:
        importlib.import_module(module)
    return f"{len(modules)} Python dependencies imported"


def check_environment(context):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL", "OpenAI default")
    return f"OPENAI_API_KEY is set; base URL: {base_url}"


def check_storage(context):
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=data_dir, prefix=".preflight_") as handle:
        handle.write(b"ok")
        handle.flush()

    free_gib = shutil.disk_usage(data_dir).free / (1024 ** 3)
    if free_gib < 10:
        context["warnings"].append(
            f"Only {free_gib:.1f} GiB is free on the data filesystem"
        )
    return f"data/ is writable; {free_gib:.1f} GiB free"


def check_thread_capacity(context):
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    result = subprocess.run(
        ["ps", "-u", str(os.getuid()), "-L", "--no-headers"],
        check=True,
        capture_output=True,
        text=True,
    )
    thread_count = sum(bool(line.strip()) for line in result.stdout.splitlines())
    uncapped = [
        f"{name}={value or 'unset'}"
        for name, value in ORIGINAL_BLAS_THREADS.items()
        if value != "1"
    ]
    if soft_limit != resource.RLIM_INFINITY:
        remaining = soft_limit - thread_count
        if uncapped and remaining < 128:
            raise RuntimeError(
                "too few thread slots for uncapped numerical libraries "
                f"({remaining} remain; {', '.join(uncapped)}). Export "
                "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
            )
        if uncapped:
            context["warnings"].append(
                "Numerical-library thread limits are not all set to 1: "
                + ", ".join(uncapped)
            )
        if remaining < 32:
            raise RuntimeError(
                f"only {remaining} user process/thread slots remain "
                f"({thread_count}/{soft_limit} used)"
            )
        if remaining < 128:
            context["warnings"].append(
                f"Only {remaining} user process/thread slots remain "
                f"({thread_count}/{soft_limit} used)"
            )
        return f"{thread_count}/{soft_limit} user process/thread slots used"
    return f"{thread_count} user processes/threads; no finite RLIMIT_NPROC"


def check_opencv(context):
    import cv2
    import numpy as np

    from preprocessing.add_subtitles_and_extract_frames import (
        draw_subtitle_on_frame,
    )

    build_info = cv2.getBuildInformation()
    if not re.search(r"FFMPEG:\s+YES", build_info):
        raise RuntimeError("OpenCV was built without FFmpeg video support")

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 180)
    rendered = draw_subtitle_on_frame(frame, "HVM preflight")
    image_path = context["temp_dir"] / "opencv_test.jpg"
    if not cv2.imwrite(str(image_path), rendered):
        raise RuntimeError("cv2.imwrite failed")
    decoded = cv2.imread(str(image_path))
    if decoded is None or decoded.shape != frame.shape:
        raise RuntimeError("OpenCV JPEG round trip failed")
    context["opencv_image"] = image_path
    return f"OpenCV {cv2.__version__}; FFmpeg and JPEG round trip available"


def check_huggingface(context):
    import cv2
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "huggingface/documentation-images"
    m3_subtitle = Path(
        hf_hub_download(
            repo_id="ByteDance-Seed/M3-Bench",
            filename="subtitles/robot/bedroom_01.srt",
            repo_type="dataset",
            cache_dir=context["temp_dir"] / "huggingface",
        )
    )
    if m3_subtitle.stat().st_size == 0:
        raise RuntimeError("downloaded M3-Bench subtitle is empty")

    entries = HfApi().list_repo_tree(
        repo_id,
        repo_type="dataset",
        recursive=True,
        expand=True,
    )
    candidates = [
        entry
        for entry in entries
        if getattr(entry, "path", "").lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
        and 5_000 <= (getattr(entry, "size", 0) or 0) <= 2_000_000
    ]
    candidates.sort(key=lambda entry: entry.size)
    if len(candidates) < 2:
        raise RuntimeError("public Hugging Face test repo has fewer than two images")

    image_paths = []
    cache_dir = context["temp_dir"] / "huggingface"
    for entry in candidates[:2]:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=entry.path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        )
        if cv2.imread(str(path)) is None:
            raise RuntimeError(f"downloaded image cannot be decoded: {entry.path}")
        image_paths.append(path)

    context["hf_images"] = image_paths
    names = ", ".join(path.name for path in image_paths)
    return f"M3-Bench is accessible; downloaded and decoded 2 images: {names}"


def check_embedding(context):
    from utils.embedding import get_multiple_embeddings

    embeddings = get_multiple_embeddings(
        ["HVM pipeline preflight one", "HVM pipeline preflight two"]
    )
    if len(embeddings) != 2 or not embeddings[0]:
        raise RuntimeError("embedding response has the wrong shape")
    if len(embeddings[0]) != len(embeddings[1]):
        raise RuntimeError("embedding dimensions do not match")
    if not all(math.isfinite(value) for value in embeddings[0]):
        raise RuntimeError("embedding contains non-finite values")
    return f"text-embedding-3-small batch works; dimension={len(embeddings[0])}"


def check_llm(context):
    from typing import Literal

    from pydantic import BaseModel

    from utils.llm_gpt import MODEL, generate_text_response

    class TextCheck(BaseModel):
        status: Literal["ok"]

    response, tokens = generate_text_response(
        'Return JSON with exactly one field: {"status": "ok"}.',
        text_format=TextCheck,
    )
    if response.status != "ok":
        raise RuntimeError(f"unexpected structured response: {response}")
    return f"{MODEL} text structured output works; tokens={tokens}"


def check_mllm(context):
    from pydantic import BaseModel

    from utils.llm_gpt import MODEL
    from utils.mllm_gpt import generate_messages, get_response

    class VisionCheck(BaseModel):
        description: str

    hf_images = context.get("hf_images")
    image_path = hf_images[0] if hf_images else context.get("opencv_image")
    if image_path is None:
        raise RuntimeError("no image is available for the multimodal check")
    messages = generate_messages(
        image_path,
        "Describe the main visible content in a few words.",
    )
    response, tokens = get_response(messages, text_format=VisionCheck)
    if not response.description.strip():
        raise RuntimeError("multimodal response description is empty")
    return f"{MODEL} vision structured output works; tokens={tokens}"


def main():
    checks = [
        ("required files", check_required_files),
        ("Python dependencies", check_dependencies),
        ("environment", check_environment),
        ("storage and permissions", check_storage),
        ("process/thread capacity", check_thread_capacity),
        ("OpenCV", check_opencv),
        ("Hugging Face download", check_huggingface),
        ("embedding API", check_embedding),
        ("text LLM", check_llm),
        ("multimodal LLM", check_mllm),
    ]
    failures = []

    print("HVM pipeline preflight")
    print("This makes 1 embedding, 1 text-LLM, and 1 multimodal-LLM request.\n")

    with tempfile.TemporaryDirectory(prefix="hvm_pipeline_preflight_") as temp:
        context = {"temp_dir": Path(temp), "warnings": []}
        for name, check in checks:
            try:
                detail = check(context)
                print(f"[PASS] {name}: {detail}")
            except Exception as exc:
                failures.append((name, exc))
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

        for warning in context["warnings"]:
            print(f"[WARN] {warning}")

    print()
    if failures:
        print(f"Preflight failed: {len(failures)}/{len(checks)} checks failed.")
        return 1
    print(f"Preflight passed: {len(checks)}/{len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

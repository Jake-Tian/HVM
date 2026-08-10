# Hierarchical Video Memory

Pipeline for building graph memory from videos and answering QA with graph + video reasoning.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API credentials**
   - Ensure the required OpenAI/HF credentials are available in your environment.

3. **Run the full pipeline**
   ```bash
   bash run_video_pipeline.sh
   ```
   - Uses `video_list.txt` by default.
   - You can also pass specific video names:
     ```bash
     bash run_video_pipeline.sh part1 part2
     ```

## What `run_video_pipeline.sh` does

1. Download shared data via `preprocessing/download_hf_folder.py`
2. Download each video via `preprocessing/download_hf_videos.py`
3. Build graph memory via `process_full_video.py`
4. Run QA reasoning via `reason.py`
5. Clean up downloaded MP4s and extracted frames

## Outputs

Per successfully processed video:
- `data/graphs/<video>.pkl` (graph memory)
- `data/memorization/<video>.json` (episodic memory + memory token summary)
- `data/reasoning/<video>.json` (QA reasoning results)

## Data Structure

Expected layout in `data/` (some folders are created/populated during pipeline runtime):

```text
data/
├── frames/
│   └── <video>/
│       └── <clip_id>/
│           └── *.jpg
├── videos/
│   └── <video>.mp4
├── subtitles/
│   └── robot/
│       └── <video>.srt
├── graphs/
│   └── <video>.pkl
├── memorization/
│   └── <video>.json
├── reasoning/
│   └── <video>.json
└── robot.json
```

Notes:
- `frames/` and `videos/` are temporary runtime artifacts and are cleaned up by `run_video_pipeline.sh`.
- `graphs/`, `memorization/`, and `reasoning/` are persistent outputs.

## Project Structure

Core pipeline:

- `process_full_video.py` - build memory graph from frames
- `reason.py` - answer questions from graph memory and selected video clips
- `run_video_pipeline.sh` - end-to-end pipeline runner
- `preprocessing/` - dataset/video download and preprocessing
- `classes/` - graph, node, edge, conversation, and structured-output models
- `utils/` - prompts, search, agent tools, and LLM/MLLM helpers

Completed experiment entrypoints retained for reproducibility:

- `abstraction_ablation.py`, `scripts/ablation/reason_ablation.py`, `scripts/ablation/run_ablation_experiment.sh`
- `noise_injection.py`, `scripts/ablation/run_noise_experiment.sh`
- `configs/` - abstraction experiment configurations

Supporting code:

- `scripts/` - experiment entrypoints, result aggregation, evaluation, reporting, and graph inspection
- `tests/` - offline automated unit tests only; no API calls or fixed local datasets
- `docs/` - architecture notes and experiment analysis

## Tests

Run the offline test suite from the project root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

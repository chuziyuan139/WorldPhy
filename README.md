<h1 align="center">WorldPhy</h1>

<p align="center">
  <strong>Benchmarking Fine-Grained Physical Plausibility of World Models in Robotics</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.34130"><img alt="Paper on arXiv" src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white"></a>
  <a href="https://huggingface.co/pandabear139/WorldPhyReward"><img alt="WorldPhyReward model on Hugging Face" src="https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#data-and-results">Data &amp; Results</a> ·
  <a href="#checkpoint">Checkpoint</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#citation">Citation</a>
</p>

## Overview

WorldPhy provides 300 initial-image–instruction pairs for evaluating video world models in robotics. The tasks cover gripper and hand manipulation, single and dual arms, and third-person, wrist, and head viewpoints.

WorldPhyReward is a fully fine-tuned Qwen3.5-9B evaluator that produces a binary score and an explanatory reason for each of seven dimensions:

| ID | Dimension |
|---|---|
| R1 / EIC | Entity and Identity Consistency |
| R2 / GMP | Geometric and Material Plausibility |
| R3 / CJC | Connection and Joint Constraints |
| R4 / Penetr. | Impenetrability |
| R5 / CIC | Contact and Interaction Causality |
| R6 / SMD | Support and Motion Dynamics |
| R7 / IF | Instruction Following |

## Data and results

```text
task/
  image/                       # 300 initial images
  instruction/tasks.json       # Instructions, image paths, task IDs, and categories
evaluation/
  evaluate.py                  # Evaluate generated videos
  overall.csv                  # Results for 13 world models
worldphyreward/
  inference/score.py            # Standalone WorldPhyReward inference
```

Each task contains `task_id`, `caption`, `image`, and `classification`. Image paths are relative to `task/instruction/tasks.json`. Images are numbered `000001.png`–`000300.png` in task order; the original task IDs are retained. Name generated videos `<task_id>.mp4`, for example `RV000015.mp4`.

The overall benchmark metric is **7/7 pass rate**: a video passes only when all seven scores are 1. [Overall results](evaluation/overall.csv) include per-dimension means, mean total scores, and the mean and sample standard deviation of pass rates over three scoring runs. These are three actual runs with the same seed and temperature 0. Twelve models have 300 videos each; Seedance has 293.

## Checkpoint

Download the default full-parameter E4 checkpoint from **[pandabear139/WorldPhyReward](https://huggingface.co/pandabear139/WorldPhyReward)**:

```bash
hf download pandabear139/WorldPhyReward --local-dir ./WorldPhyReward-checkpoint
```

Use this local directory with `--model`, or pass `--model pandabear139/WorldPhyReward` to download the checkpoint automatically. Use the provided inference script to retain the benchmark's prompt, preprocessing, and decoding configuration.

## Usage

### Environment

Use a CUDA-capable Linux environment with the reference dependencies below.

Reference inference environment: CUDA 12.8 with the following packages:

```bash
pip install vllm==0.19.1 torch==2.10.0 transformers==5.5.4 \
  av==16.0.0 numpy==2.2.6 Pillow==11.3.0 tokenizers==0.22.2
```

### Evaluate a directory of generated videos

```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluation/evaluate.py \
  --videos /path/to/generated-videos \
  --model ./WorldPhyReward-checkpoint \
  --output /path/to/evaluation-results --tp 2 --batch-size 16
```

### Score a single video

```bash
CUDA_VISIBLE_DEVICES=0,1 python worldphyreward/inference/score.py \
  --model ./WorldPhyReward-checkpoint \
  --video /path/to/video.mp4 --image /path/to/initial-image.png \
  --caption "Task instruction." --output /path/to/scores --tp 2
```

For batch inference, use `--records records.json`, a list of objects containing `video`, `image`, and `caption`; media paths are relative to that file. Set `--tp` to the number of visible GPUs.

Inference uses the fixed prompt, 5 fps sampling with up to 128 frames, 640×480 bicubic resizing, BF16, a 24,576-token context, and at most 1,024 output tokens. The original decoding bias and output parser are embedded in `score.py`. Outputs retain raw responses, seven scores and reasons, and parsing errors. The evaluator writes all three runs and aggregate CSV files to `--output`; missing task videos are listed, and pass rates use the number of available videos as their denominator.

## Citation

```bibtex
@misc{chu2026worldphy,
  title         = {{WorldPhy}: Benchmarking Fine-Grained Physical Plausibility of World Models in Robotics},
  author        = {Ziyuan Chu and Mingzhi Zheng and Kaiyue Zhang and Jinan Zhang and Linchao Zhu},
  year          = {2026},
  eprint        = {2611.34130},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.34130}
}
```

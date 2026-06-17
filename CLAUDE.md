# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptZip is a research project for prompt compression — reducing long prompts before sending them to LLMs. It implements multiple compression strategies (reranker-based, PPL-based, attention-based, embedding-based, token-level via LLMLingua) and evaluates them on the LongBench benchmark using vLLM-backed inference.

The repository also contains `draft-based-approx-llm/`, a separate subproject implementing SpecKV/SpecPC/SpecKV-PC (arXiv 2506.08373) with its own dependencies and evaluation pipeline. Treat it as an independent project — do not mix its dependencies with the main PromptZip code.

## Build & Run

- **Package manager**: `uv` (pyproject.toml + uv.lock)
- **Python**: pinned to 3.12.3 (`requires-python == "3.12.3"`)
- **Install**: `uv sync`
- **flash-attn** has a special build dependency on torch, configured via `[tool.uv.extra-build-dependencies]`
- **GPU required**: CUDA needed for flash-attn, vllm, onnxruntime-gpu

### Running Evaluations

Evaluations are driven by YAML configs. Template at `experiments/config/templete.yaml`.

```bash
# Main LongBench evaluation
cd experiments/evaluation
python eval_longbench.py --config ../config/your_config.yaml

# End-to-end latency benchmark
python benchmark_e2e.py --config ../config/your_config.yaml

# Simple shell wrapper
bash experiments/evaluation/run.sh
```

### Running Tests

No formal test framework. Standalone assert-based tests only:

```bash
python experiments/evaluation/test_parse_baseline_output.py
```

## Architecture

### Core Library (`src/`)

- **`compressor.py`** (~3800 lines) — All compressor implementations inherit from `BaseCompressor`:
  - `NullCompressor` — no-op baseline
  - `RerankCompressor` — chunk-level selection via Qwen3-Reranker (HF or vLLM engine); selection modes: topk, topp, cluster, cluster-zscore, mmr
  - `PPLCompressor` — perplexity-based chunk filtering
  - `AttnScoreCompressor` — attention score-based compression
  - `EmbeddingCompressor` — embedding similarity-based chunk selection
  - `LongLLMLinguaTokenCompressor` — token-level iterative compression
  - `PromptCompressor` — comprehensive LLMLingua/LLMLingua-2 compressor (context, sentence, and token-level filtering, structured JSON support)
  - `QwenCompressor` / `QwenVLLMCompressor` — summarization-based compression using Qwen models

- **`util.py`** — `TokenClfDataset`, `seed_everything`, token/text processing helpers
- **`count_chunks.py`** — Dataset chunk size analysis utility

### Evaluation Pipeline (`experiments/evaluation/`)

- **`eval_longbench.py`** — Main eval script: loads dataset, builds compressor/LLM from YAML config, runs async prediction, computes per-dataset metrics, writes JSON/CSV results
- **`benchmark_e2e.py`** — Latency benchmark at various target token lengths
- **`metrics.py`** — Scoring functions: F1, ROUGE, classification, count, retrieval, code similarity (with Chinese variants)
- **`utils.py`** — LLM construction helpers (`construct_llm`, `setup_logging`)
- **`extract_dataset.py`** — Download and prepare LongBench data
- **`parse_baseline_output.py`** / **`parse_and_collect.py`** — Parse and aggregate evaluation results
- **`compare_gpu_device.py`** — Compare GPU vs device-side inference

### Config System (`experiments/config/`)

YAML configs define three component blocks:
- `reranker_config` — model, chunk_size, rate, engine (hf/vllm), selection_mode
- `compressor_config` — model_type (longllmlingua/llmlingua2/llm/rerank/PPL/null), rate, chunk_size
- `llm_config` — target LLM with vLLM sampling params

## Conventions

- **Language**: Comments and commit messages are primarily in Chinese (中文). Follow this pattern: `feat(scope): 中文描述` or `fix(scope): 中文描述`
- **No linting/formattering configured**: No ruff, flake8, or pre-commit hooks. Follow existing code style when editing.
- **No CI/CD**: All testing and evaluation is manual/local.
- **`sys.path.append`** is used in evaluation scripts to import from `src/` — the project does not use formal package installation for evaluation runs.
- **Multiprocessing**: Evaluation scripts use `multiprocessing.set_start_method("spawn")` due to vLLM/cuda requirements.
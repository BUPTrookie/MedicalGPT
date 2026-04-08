# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedicalGPT implements the full ChatGPT/InstructGPT training pipeline for large language models. Despite the medical naming, it is a general-purpose LLM training framework. It supports 20+ model families (Qwen, LLaMA, DeepSeek, ChatGLM, Baichuan, etc.) through a pluggable conversation template system.

## Repository Layout

```
training/          Core training scripts (one per stage)
scripts/           Shell launch scripts + DeepSpeed configs (zero1/2/3.json)
tools/             Utilities: merge LoRA, convert datasets, quantize models
demo/              Inference & deployment: CLI, Gradio, FastAPI, vLLM
data/              Sample data organized by stage (pretrain/, finetune/, reward/, grpo/)
notebooks/         End-to-end Colab pipelines (DPO path, PPO path)
role_play_data/    Medical dialogue generation scripts using GPT-4o/Doubao/MiniMax
```

## Training Pipeline

The stages run independently. Each has a training script and a corresponding `scripts/run_*.sh` launcher.

```
Stage 1: training/pretraining.py          (PT)   Continue pretraining on raw text
Stage 2: training/supervised_finetuning.py (SFT)  Instruction fine-tuning on conversation data
Stage 3: Alignment (pick one path):
  ├─ training/dpo_training.py              (DPO)   Direct preference optimization
  ├─ training/orpo_training.py             (ORPO)  Odds ratio preference optimization
  ├─ training/grpo_training.py             (GRPO)  Group relative policy optimization (math/code)
  └─ training/reward_modeling.py → training/ppo_training.py  (RM→RLOO)  Classic RLHF
```

Between stages, merge LoRA adapters with `tools/merge_peft_adapter.py` before feeding the model to the next stage.

**Important**: `ppo_training.py` uses TRL's `RLOOTrainer` (REINFORCE Leave-One-Out), not PPO, despite the filename.

## Running Training

All launch scripts are in `scripts/`. They assume CUDA GPUs.

```bash
# SFT (2 GPUs, DDP)
bash scripts/run_sft.sh

# DPO
bash scripts/run_dpo.sh

# GRPO (2 GPUs, DDP)
bash scripts/run_grpo.sh

# Reward model (single process, torchrun not supported)
bash scripts/run_rm.sh

# PPO/RLOO (requires trained reward model)
bash scripts/run_ppo.sh
```

Key patterns in launch scripts:
- SFT/PT/GRPO use `torchrun --nproc_per_node 2` (DDP)
- RM/DPO/ORPO/PPO use plain `python` (model parallelism via `device_map=auto`)
- All default to LoRA with `--use_peft True`, rank 8, targeting all linear layers

## Merge and Inference

```bash
# Merge LoRA adapter back into base model
python tools/merge_peft_adapter.py \
    --model_name_or_path <base_model> \
    --peft_path <outputs-sft-xxx> \
    --output_dir <merged_model>

# CLI inference
python demo/inference.py --model_name_or_path <merged_model>
```

## Conversation Template System

`training/template.py` contains 33 registered templates. The template name is passed via `--template_name` and must match the base model family:

| Model Family | Template |
|---|---|
| Qwen 1.5/2/2.5 | `qwen` |
| Qwen 3 | `qwen3` or `qwen3_nothink` |
| Qwen 3.5 | `qwen3_5` or `qwen3_5_nothink` |
| LLaMA-2 | `llama2` |
| LLaMA-3 | `llama3` |
| ChatGLM 1/2/3 | `chatglm` / `chatglm2` / `chatglm3` |
| DeepSeek V3 | `deepseek3` |

Using the wrong template silently produces bad results. Always verify the template matches your model.

## Data Formats

**SFT** (`data/finetune/`): ShareGPT conversation format
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

**DPO/RM/ORPO** (`data/reward/`): Preference pairs
```json
{"system": "", "history": [], "question": "...", "response_chosen": "...", "response_rejected": "..."}
```

**GRPO** (`data/grpo/`): Math problems with answers
```json
{"question": "...", "answer": "..."}
```

Use `tools/convert_dataset.py` to convert alpaca/QA format into ShareGPT format. Use `tools/validate_jsonl.py` to check dataset validity.

## Architecture Notes

- All training scripts share the same pattern: dataclass args → load tokenizer → load data → load model → apply LoRA → train → save
- LoRA targets all linear layers by default (`--target_modules all`). The `find_all_linear_names()` function (present in each script) auto-discovers linear layers, excluding `lm_head` (and `score` for reward models)
- SFT uses label masking (`IGNORE_INDEX = -100`) on query tokens so the model only learns to predict responses
- Reward model replaces `lm_head` with a single-output `score` layer via `AutoModelForSequenceClassification(num_labels=1)`
- GRPO uses rule-based reward functions (math verification via `math_verify` library), not a trained reward model
- `SavePeftModelTrainer` (in SFT/RM scripts) overrides `save_model()` to only save LoRA adapter weights

## Environment

```bash
pip install -r requirements.txt
# bitsandbytes is commented out in requirements.txt; install manually for quantization:
# pip install bitsandbytes
```

Python 3.10+ required. Key dependencies: transformers>=5.1.0, trl>=0.27.0, peft>=0.14.0, torch>=2.0.

## DeepSpeed

ZeRO configs are in `scripts/zero1.json`, `zero2.json`, `zero3.json`. Pass via `--deepspeed scripts/zero2.json` in launch scripts. Note: QLoRA is incompatible with FSDP and DeepSpeed ZeRO-3.

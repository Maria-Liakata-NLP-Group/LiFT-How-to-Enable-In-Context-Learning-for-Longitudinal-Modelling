#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
IFT TRAINER (S3-first) — Longitudinal iFT (LoRA + Temporal/Label Embeddings + CLS)
================================================================================

This script is TRAINER-ONLY. It assumes you already ran the BUILDER which wrote
stage JSONL files to S3. The trainer:
  - reads those JSONLs from S3,
  - tokenizes + chunks them with strict masking rules,
  - trains a LoRA-adapted causal LM plus a temporal/label embedding module and a
    global multi-task classification head,
  - uploads checkpoints, final weights, BEST weights, and summaries to S3,
  - resumes safely using an S3 state file.

-------------------------------------------------------------------------------
WHAT THE BUILDER PRODUCES (INPUT DATA TO THIS TRAINER)
-------------------------------------------------------------------------------
The builder writes one JSONL per stage (stage1..stage3; optionally stage4):
  {CURRICULUM_PREFIX_S3}_stage1.jsonl
  {CURRICULUM_PREFIX_S3}_stage2.jsonl
  {CURRICULUM_PREFIX_S3}_stage3.jsonl
  (optional) {CURRICULUM_PREFIX_S3}_stage4_high_shot.jsonl

Each JSONL line is a training example with this structure (simplified):

{
  "input": "<instruction> ... </instruction>\n"
           "<few-shot> ... </few-shot>\n"
           "<query> ... </query>\n"
           "<output>\n",
  "output": "GOLD_LABEL_TEXT",         # the label token(s) used as generation target
  "task_type": "annomi|lrs|mood|...",   # dataset/task identity
  "raw_label_id": int,                 # label id within that task (0..K-1)
  "k_shot_used": int,                  # how many true in-group demos were used
  "stage": int,                        # stage index
  "meta": {...},                       # original row metadata (optional)
  "example_id": "stable_hash_id"       # stable ID used for dedup + auditing
}

Important builder behavior:
- Few-shot demonstrations are REAL longitudinal in-group demos.
- If the example would exceed token budget, demos are dropped until it fits,
  otherwise the example is dropped.
- Staging is group-locked (same group goes to one stage) to prevent leakage.

-------------------------------------------------------------------------------
WHAT THE MODEL LEARNS (OBJECTIVES)
-------------------------------------------------------------------------------
The model is trained with THREE coupled objectives:

(A) Prompt CE loss (teacher-forcing next-token) — BUT NOT inside <HIST>
    - We compute next-token CE over prompt tokens that are OUTSIDE <HIST>... </HIST>.
    - This makes the model focus on instruction + query formatting without
      overfitting to raw history.

(B) Output Focal LM loss (teacher-forcing next-token) — ONLY on output region
    - We compute a focal loss only over tokens that correspond to:
        response tokens + EOS.
    - This strongly emphasizes learning to produce the correct output label.

(C) Classification focal loss (global label head) — stage-aware multi-task labels
    - A classification head predicts a GLOBAL label id, not task-local.
    - We stamp the global label id across the output region (label_index_seq),
      pool the final hidden state at the last stamped position, and classify.
    - This yields a shared label space across tasks (avoids collisions).

The architecture additions:
- LoRA adapters on attention + MLP proj modules (r changes per stage).
- RelativeTemporalEmbedding module:
    - per-token absolute position embedding
    - per-token relative position-to-end embedding
    - per-token label embedding (global label id stamped on output region)
  These are combined and projected into the LM hidden space, then added to token
  embeddings before the base LM forward pass.
- Classification head (Linear hidden -> num_global_classes)

-------------------------------------------------------------------------------
TOKENIZATION + MASKING RULES (CRITICAL)
-------------------------------------------------------------------------------
Given:
  prompt_ids = tokenize(example["prompt"])
  response_ids = tokenize(example["response"])
We create:
  input_ids = prompt_ids + response_ids + [eos]

We create masks/labels:
- labels_ce[t]      = next token id, for prompt positions excluding <HIST> region
- labels_focal[t]   = next token id, for output region only (response + eos)
- focal_mask[t]     = 1 if output region, else 0
- label_index_seq   = global label id stamped over output region positions

Then we chunk the sequence with overlap (stride) so long samples fit:
- chunk_max_len = min(model_max_len, MAX_TOKENS_PER_INPUT)
- dyn_stride ~ chunk_max_len/4 (bounded by min_stride/max_stride)
This yields a dataset of chunks for training.

-------------------------------------------------------------------------------
STAGE SCHEDULE (CURRICULUM)
-------------------------------------------------------------------------------
Stages are trained sequentially. Each stage uses its own LoRA rank + LR:
- Stage 1: LoRA r=4  lr=2e-4  scheduler=cosine
- Stage 2: LoRA r=8  lr=1e-4  scheduler=cosine
- Stage 3: LoRA r=16 lr=5e-5  scheduler=constant

Resume behavior:
- `OUTPUT_S3/curriculum_state.json` tracks the last completed stage plus pointers
  to the last stage’s LoRA + wrapper heads in S3.
- When resuming, the trainer downloads those and continues at the next stage.
- If LoRA rank increases, it loads the previous weights with rank resize and
  freezes the old rank slice so only the new rank learns additional capacity.

-------------------------------------------------------------------------------
WHAT GETS WRITTEN TO S3 (ARTIFACTS)
-------------------------------------------------------------------------------
Global:
- OUTPUT_S3/tokenizer/                      (tokenizer + manifest)
- OUTPUT_S3/LATEST.json                     (points to most recent stage manifest)
- OUTPUT_S3/training_report.json            (single global report of all stages)
- OUTPUT_S3/curriculum_state.json           (resume state)

Per stage: OUTPUT_S3/stage_{idx}_{stagetag}/
- checkpoints/checkpoint-{step}/            (HF checkpoint uploads during training)
- lora_model/                               (END-OF-STAGE LoRA weights)
- wrapper_heads.pt                          (END-OF-STAGE temporal + cls head weights)
- BEST.json                                 (BEST checkpoint pointer + best metric)
- best_export/lora_model/                   (BEST weights export)
- best_export/wrapper_heads.pt              (BEST wrapper heads export)
- lora_model_best/                          (BEST weights in CANONICAL path)  ✅
- wrapper_heads_best.pt                     (BEST heads in CANONICAL path)    ✅
- curriculum_summary.json                   (sanity summary: counts + token stats) ✅
- eval_loss.json                            (final eval loss payload)         ✅
- eval_metrics.json                         (classification acc/macro-f1)     (if eval)
- log_history.json                          (trainer.state.log_history)
- inference_manifest.json                   (full manifest for inference)

Canonical best paths matter because deployment can always read:
  stage_s3_dir/lora_model_best
  stage_s3_dir/wrapper_heads_best.pt
without needing to know about best_export/.

-------------------------------------------------------------------------------
CURRICULUM SANITY SUMMARY (WHY IT’S IMPORTANT)
-------------------------------------------------------------------------------
For each stage we write stage_s3_dir/curriculum_summary.json containing:
- raw example counts by:
    task_type, k_shot_used, raw_label_id
- balanced example counts by:
    task_type, k_shot_used, raw_label_id, global_label_id
- chunk count + token length stats (min/max/mean)
This makes it obvious if a stage file accidentally contains the wrong tasks or
if balancing removed too much data.

-------------------------------------------------------------------------------
GLOBAL TRAINING REPORT
-------------------------------------------------------------------------------
At the end of each stage we update:
  OUTPUT_S3/training_report.json
with a per-stage entry including:
- best pointer path (BEST.json)
- best metric + best step
- best canonical artifact paths
- final eval_loss + eval metrics
This enables quick comparison across runs without hunting through directories.

UPDATED ACCORDINGLY (Longitudinal upgrades):
1) Explicit time index + history length markers
   - Assumed already added by the builder into the prompt text (header + t-# lines).
   - Trainer additionally reads group_id + timestep from JSONL/meta if present.

5) Explicit history attention incentive
   - Adds an auxiliary classification head that must predict the global label using ONLY
     the pooled hidden state over tokens inside <HIST> ... </HIST>.
   - Adds hist classification focal loss term, weighted by HIST_CLS_WEIGHT.

6) Timestep embedding (absolute time) separately from relative
   - Adds a timestep embedding lookup (timestep_id_seq) that is broadcast across tokens
     and injected alongside temporal/label/relative embeddings.

7) Sequence-consistent batching (same group contiguous)
   - Uses a custom BatchSampler that sorts chunks by (group_id, timestep, chunk_start)
     so batches tend to be time-contiguous within each group.
-------------------------------------------------------------------------------
ENVIRONMENT VARIABLES
-------------------------------------------------------------------------------
Required:
  FSSPEC_S3_ENDPOINT_URL   (S3 endpoint)
  FSSPEC_S3_KEY            (access key)
  FSSPEC_S3_SECRET         (secret key)
  IFT_OUTPUT_S3            (where trainer writes outputs)
  IFT_NEWIFTDATA_S3        (where curriculum prefix usually lives)

Optional:
  CURRICULUM_PREFIX_S3     (explicit prefix for stage JSONLs)
  BASE_MODEL_NAME          (HF model name, default "meta-llama/Llama-3.1-8B-Instruct")
  IFT_WORKDIR              (ephemeral local working dir, default /tmp/ift_run)
  BUILDER_CAP              (token cap safety)
  MAX_TOKENS_PER_INPUT     (max token budget for chunking)
  TEMPORAL_MAX_LEN         (cap temporal embedding max length)
  BALANCE_OFF=1            (disable balancing)
  LOSS_LOG_EVERY           (loss debug print interval)
  ALLOW_MISSING_STAGE=1    (do not crash if stage JSONL missing)

-------------------------------------------------------------------------------
SAFE DEFAULTS + INTENDED USAGE
-------------------------------------------------------------------------------
- Run builder once (writes stage JSONLs to S3).
- Run trainer in "train" mode pointing to the same CURRICULUM_PREFIX_S3.
- Inspect S3:
    - curriculum_summary.json per stage
    - training_report.json (global)
    - BEST.json per stage (best checkpoint pointer)
    - lora_model_best + wrapper_heads_best.pt for deployment/inference.

================================================================================
"""


import os
import json
import random
import datetime
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from datasets import Dataset as HFDataset

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.utils import ModelOutput

from peft import LoraConfig, get_peft_model

import fsspec

try:
    from safetensors.torch import load_file as safetensors_load_file
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False


# =============================================================================
# 1) Repro + ENV paths
# =============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
IFT_WORKDIR = os.environ.get("IFT_WORKDIR", "/tmp/ift_run")

HF_HOME = os.environ.get("HF_HOME", os.path.join(IFT_WORKDIR, "hf_home"))
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))

OUTPUT_BASE_DIR_LOCAL = os.path.join(IFT_WORKDIR, "outputs_local")
os.makedirs(OUTPUT_BASE_DIR_LOCAL, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)

OUTPUT_S3 = os.environ["IFT_OUTPUT_S3"].rstrip("/")
CURRICULUM_S3 = os.environ["IFT_NEWIFTDATA_S3"].rstrip("/")

CURRICULUM_PREFIX_S3 = os.environ.get(
    "CURRICULUM_PREFIX_S3",
    f"{CURRICULUM_S3}/ICL_curriculum_IFT_llama8bnew",
).rstrip("/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOW_MISSING_STAGE = os.environ.get("ALLOW_MISSING_STAGE", "0") == "1"


# =============================================================================
# 2) S3 FS
# =============================================================================
def env_required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise ValueError(f"Missing required env var: {name}")
    return v

storage_options = {
    "key": env_required("FSSPEC_S3_KEY"),
    "secret": env_required("FSSPEC_S3_SECRET"),
    "client_kwargs": {"endpoint_url": env_required("FSSPEC_S3_ENDPOINT_URL")},
}
fs = fsspec.filesystem("s3", **storage_options)

def s3_join(prefix: str, *parts: str) -> str:
    prefix = prefix.rstrip("/")
    rest = "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None])
    return f"{prefix}/{rest}" if rest else prefix

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def log_event(message: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)

def safe_rmtree(path: str) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def s3_put_dir(local_dir: str, s3_dir: str, recursive: bool = True) -> None:
    if not os.path.exists(local_dir):
        return
    log_event(f"📤 S3 UPLOAD DIR: {local_dir} -> {s3_dir}")
    fs.put(local_dir, s3_dir, recursive=recursive)

def s3_put_file(local_file: str, s3_file: str) -> None:
    if not os.path.exists(local_file):
        return
    log_event(f"📤 S3 UPLOAD FILE: {local_file} -> {s3_file}")
    fs.put(local_file, s3_file)

def s3_get_dir(s3_dir: str, local_dir: str) -> None:
    ensure_dir(local_dir)
    log_event(f"📥 S3 DOWNLOAD DIR: {s3_dir} -> {local_dir}")
    fs.get(s3_dir, local_dir, recursive=True)

def s3_get_file(s3_file: str, local_file: str) -> None:
    ensure_dir(os.path.dirname(local_file) or ".")
    log_event(f"📥 S3 DOWNLOAD FILE: {s3_file} -> {local_file}")
    fs.get(s3_file, local_file)

def s3_write_text_atomic(s3_path: str, text: str) -> None:
    tmp = s3_path + ".tmp"
    with fs.open(tmp, "w") as f:
        f.write(text)
    try:
        fs.copy(tmp, s3_path)
        fs.rm(tmp)
    except Exception:
        with fs.open(s3_path, "w") as f:
            f.write(text)
        try:
            fs.rm(tmp)
        except Exception:
            pass

def s3_read_text(s3_path: str) -> Optional[str]:
    if not fs.exists(s3_path):
        return None
    with fs.open(s3_path, "r") as f:
        return f.read()

def s3_exists(s3_path: str) -> bool:
    return bool(fs.exists(s3_path))


# =============================================================================
# 3) Tokenizer + caps
# =============================================================================
if "HF_TOKEN" in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

SPECIAL_TOKENS = [
    "<instruction>", "</instruction>",
    "<few-shot>", "</few-shot>",
    "<query>", "</query>",
    "<output>", "</output>",
    "<HIST>", "</HIST>",
    "<CURR>", "</CURR>",
    "<TASK_MI>", "<TASK_MOOD>", "<TASK_LRS>",
]

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

lm_config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
LM_CONTEXT_WINDOW = getattr(lm_config, "max_position_embeddings", 4096)
if not isinstance(LM_CONTEXT_WINDOW, int) or LM_CONTEXT_WINDOW <= 0 or LM_CONTEXT_WINDOW > 10_000_000:
    LM_CONTEXT_WINDOW = 4096

BUILDER_CAP = int(os.environ.get("BUILDER_CAP", "4096"))
MAX_TOKENS_PER_INPUT_ENV = int(os.environ.get("MAX_TOKENS_PER_INPUT", "4096"))
TEMPORAL_MAX_LEN = int(os.environ.get("TEMPORAL_MAX_LEN", "4096"))

MAX_CONTEXT_FOR_BUILDER = min(LM_CONTEXT_WINDOW - 32, BUILDER_CAP)
MAX_TOKENS_PER_INPUT = min(MAX_TOKENS_PER_INPUT_ENV, MAX_CONTEXT_FOR_BUILDER)

# NEW: absolute timestep embedding vocab cap (how many distinct timestep ids)
TIMESTEP_MAX_VOCAB = int(os.environ.get("TIMESTEP_MAX_VOCAB", str(TEMPORAL_MAX_LEN)))

# NEW: history attention incentive weight
HIST_CLS_WEIGHT = float(os.environ.get("HIST_CLS_WEIGHT", "0.2"))

log_event(f"CUDA available={torch.cuda.is_available()} | device_count={torch.cuda.device_count()}")
log_event(f"LM_CONTEXT_WINDOW={LM_CONTEXT_WINDOW}")
log_event(f"MAX_CONTEXT_FOR_BUILDER={MAX_CONTEXT_FOR_BUILDER} | MAX_TOKENS_PER_INPUT={MAX_TOKENS_PER_INPUT} | TEMPORAL_MAX_LEN={TEMPORAL_MAX_LEN}")
log_event(f"TIMESTEP_MAX_VOCAB={TIMESTEP_MAX_VOCAB} | HIST_CLS_WEIGHT={HIST_CLS_WEIGHT}")


# =============================================================================
# 4) Curriculum sources (ONLY 3 STAGES) + outputs
# =============================================================================
STAGE_STATE_S3 = s3_join(OUTPUT_S3, "curriculum_state.json")
TOKENIZER_S3_DIR = s3_join(OUTPUT_S3, "tokenizer")
LATEST_MANIFEST_S3 = s3_join(OUTPUT_S3, "LATEST.json")
BUILDER_STATS_S3 = s3_join(OUTPUT_S3, "builder_stats.json")
TRAINING_REPORT_S3 = s3_join(OUTPUT_S3, "training_report.json")

CURRICULUM_STAGE_FILES_S3 = [
    f"{CURRICULUM_PREFIX_S3}_stage1.jsonl",
    f"{CURRICULUM_PREFIX_S3}_stage2.jsonl",
    f"{CURRICULUM_PREFIX_S3}_stage3.jsonl",
]


# =============================================================================
# 5) Config
# =============================================================================
@dataclass
class TrainConfig:
    model_name: str = BASE_MODEL_NAME
    output_base_dir_local: str = OUTPUT_BASE_DIR_LOCAL
    json_sources_s3: List[str] = None

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16

    min_stride: int = 512
    max_stride: int = 2048

    max_timestep_emb: int = 4096
    temporal_embed_dim: int = 128

    num_classes: int = 1
    val_fraction: float = 0.1

    lora_dropout: float = 0.05

    ce_prompt_weight: float = 1.0
    focal_output_weight: float = 1.0
    class_focal_weight: float = 1.0
    hist_class_weight: float = HIST_CLS_WEIGHT  # NEW
    mse_weight: float = 0.0
    focal_gamma: float = 1.0

CONFIG = TrainConfig(json_sources_s3=CURRICULUM_STAGE_FILES_S3)

def read_jsonl_s3(path: str) -> List[Dict[str, Any]]:
    if not s3_exists(path):
        msg = f"Stage file not found in S3: {path}"
        if ALLOW_MISSING_STAGE:
            log_event(f"[WARN] {msg} (ALLOW_MISSING_STAGE=1 -> returning empty)")
            return []
        raise FileNotFoundError(msg)
    out = []
    with fs.open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def get_dynamic_model_max_length(tokenizer, model=None) -> Optional[int]:
    if model is not None and hasattr(model, "config"):
        max_pos = getattr(model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and 0 < max_pos < 10_000_000:
            return max_pos
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and 0 < tmax < 10_000_000:
        return tmax
    return None


# =============================================================================
# 6) Global label namespace (annomi + lrs + mood)
# =============================================================================
LABEL_PAD_ID = 0

TASK_LABELS = {
    "annomi": ["neutral", "sustain", "change"],
    "lrs":    ["0", "1"],
    "mood":   ["0", "IS", "IE"],
}

def build_global_label_map() -> Tuple[Dict[Tuple[str, int], int], Dict[int, Tuple[str, int]]]:
    m, inv = {}, {}
    nxt = 1
    for task, names in TASK_LABELS.items():
        for raw_id in range(len(names)):
            m[(task, raw_id)] = nxt
            inv[nxt] = (task, raw_id)
            nxt += 1
    return m, inv

GLOBAL_LABEL_MAP, GLOBAL_LABEL_INV = build_global_label_map()
CONFIG.num_classes = 1 + len(GLOBAL_LABEL_INV)

def _get_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    m = item.get("meta", None)
    return m if isinstance(m, dict) else {}

def standardize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    UPDATED: pulls group_id + timestep (for sequence-consistent batching + timestep embedding)
    Supports either top-level fields (preferred) or meta fields.
    """
    prompt = item.get("input", "").strip() + "\n"
    response = str(item.get("output", "")).strip()
    task_type = str(item.get("task_type", "generic")).strip()
    raw_label_id = int(item.get("raw_label_id", item.get("label_id", 0)))
    label_id = GLOBAL_LABEL_MAP.get((task_type, raw_label_id), LABEL_PAD_ID)

    meta = _get_meta(item)
    group_id = str(item.get("group_id", meta.get("group_id", "unknown")))
    timestep = int(item.get("timestep", meta.get("timestep", 0)))

    return {
        "prompt": prompt,
        "response": response,
        "label_id": int(label_id),
        "raw_label_id": int(raw_label_id),
        "task_type": task_type,
        "target_value": 0.0,
        "k_shot_used": int(item.get("k_shot_used", 0)),
        "stage": int(item.get("stage", 0)),
        "meta": item.get("meta", None),
        "example_id": item.get("example_id", None),
        "group_id": group_id,
        "timestep": timestep,
    }

def balance_dataset(raw_items: List[Dict[str, Any]], rng: random.Random, kshot_bins=None) -> List[Dict[str, Any]]:
    if os.environ.get("BALANCE_OFF", "0") == "1":
        log_event("Balancing is disabled via BALANCE_OFF=1")
        return raw_items

    from collections import defaultdict
    if not raw_items:
        return raw_items

    if kshot_bins is None:
        ks = set([x.get("k_shot_used", None) for x in raw_items])
        kshot_bins = sorted(k for k in ks if k is not None)

    buckets = defaultdict(list)
    for item in raw_items:
        t = item.get("task_type", "unknown")
        k = item.get("k_shot_used", None)
        if k in kshot_bins:
            buckets[(t, k)].append(item)

    if not buckets:
        return raw_items

    target = min(len(v) for v in buckets.values())
    log_event(f"Balancing buckets to size={target}, {len(buckets)} buckets")

    balanced = []
    for _, vals in buckets.items():
        if len(vals) > target:
            balanced.extend(rng.sample(vals, target))
        else:
            balanced.extend(vals)
    return balanced


# =============================================================================
# 7) Tokenization + chunking (UPDATED: hist_mask_full + timestep_id_seq + group/time keys)
# =============================================================================
def tokenize_dataset_dynamic(dataset_items, tokenizer, min_stride, max_stride, model=None):
    chunked = []
    effective_max_len = get_dynamic_model_max_length(tokenizer, model)
    chunk_max_len = MAX_TOKENS_PER_INPUT if effective_max_len is None else min(effective_max_len, MAX_TOKENS_PER_INPUT)

    hist_start_id = tokenizer.convert_tokens_to_ids("<HIST>")
    hist_end_id = tokenizer.convert_tokens_to_ids("</HIST>")

    for ex in dataset_items:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(ex["response"], add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
        seq_len = len(input_ids)

        # prompt-only hist mask, then extend to full seq
        hist_mask_prompt = [0] * len(prompt_ids)
        in_hist = False
        for i, tid in enumerate(prompt_ids):
            if tid == hist_start_id:
                in_hist = True
                hist_mask_prompt[i] = 1
            elif tid == hist_end_id:
                hist_mask_prompt[i] = 1
                in_hist = False
            elif in_hist:
                hist_mask_prompt[i] = 1

        hist_mask_full = hist_mask_prompt + [0] * (len(response_ids) + 1)

        output_start = len(prompt_ids)
        output_positions = set(range(output_start, output_start + len(response_ids) + 1))

        labels_ce = [-100] * seq_len
        labels_focal = [-100] * seq_len
        focal_mask = [0] * seq_len

        for t in range(seq_len - 1):
            next_pos = t + 1
            next_tok = input_ids[next_pos]

            # prompt CE outside history only
            if next_pos < len(prompt_ids) and hist_mask_prompt[next_pos] == 0:
                labels_ce[t] = next_tok

            # output focal only
            if next_pos in output_positions:
                labels_focal[t] = next_tok
                focal_mask[t] = 1

        # stamp global label id on output region for CLS pooling
        label_id = int(ex["label_id"])
        label_index_seq = [0] * seq_len
        if label_id != 0:
            for pos in output_positions:
                if 0 <= pos < seq_len:
                    label_index_seq[pos] = label_id

        # NEW: absolute timestep sequence (broadcast)
        timestep = int(ex.get("timestep", 0))
        timestep_clamped = max(0, min(timestep, TIMESTEP_MAX_VOCAB - 1))
        timestep_id_seq = [timestep_clamped] * seq_len

        dyn_stride = min(max(min_stride, max(1, min(seq_len, chunk_max_len) // 4)), max_stride)

        start = 0
        while start < seq_len:
            end = min(start + chunk_max_len, seq_len)

            ids = input_ids[start:end]
            attn = [1] * len(ids)

            ce = labels_ce[start:end]
            foc = labels_focal[start:end]
            fmask = focal_mask[start:end]
            idx_seq = label_index_seq[start:end]
            hmask = hist_mask_full[start:end]
            ts_seq = timestep_id_seq[start:end]

            pad = chunk_max_len - len(ids)
            if pad > 0:
                ids += [tokenizer.pad_token_id] * pad
                attn += [0] * pad
                ce += [-100] * pad
                foc += [-100] * pad
                fmask += [0] * pad
                idx_seq += [0] * pad
                hmask += [0] * pad
                ts_seq += [0] * pad

            chunked.append({
                "input_ids": ids,
                "attention_mask": attn,
                "labels_ce": ce,
                "labels_focal": foc,
                "focal_mask": fmask,
                "label_index_seq": idx_seq,
                "hist_mask": hmask,               # NEW
                "timestep_id_seq": ts_seq,         # NEW
                "target_value": 0.0,
                "task_type": ex["task_type"],
                "group_id": ex.get("group_id", "unknown"),  # NEW
                "timestep": int(ex.get("timestep", 0)),     # NEW
                "chunk_start": int(start),                  # NEW (for sorting)
            })

            if end == seq_len:
                break
            start += chunk_max_len - dyn_stride

    return HFDataset.from_list(chunked)

def custom_data_collator(batch):
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)

    ids = torch.full((B, maxlen), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((B, maxlen), dtype=torch.long)

    labels_ce = torch.full((B, maxlen), -100, dtype=torch.long)
    labels_focal = torch.full((B, maxlen), -100, dtype=torch.long)
    focal_mask = torch.zeros((B, maxlen), dtype=torch.float)

    lbl_idx = torch.zeros((B, maxlen), dtype=torch.long)
    hist_mask = torch.zeros((B, maxlen), dtype=torch.float)         # NEW
    timestep_id_seq = torch.zeros((B, maxlen), dtype=torch.long)     # NEW

    targets = torch.zeros(B, dtype=torch.float)
    types: List[str] = []

    # NEW: keep these for sequence batching debug / sorting
    group_ids: List[str] = []
    timesteps: List[int] = []
    chunk_starts: List[int] = []

    for i, ex in enumerate(batch):
        l = len(ex["input_ids"])
        ids[i, :l] = torch.tensor(ex["input_ids"], dtype=torch.long)
        mask[i, :l] = torch.tensor(ex["attention_mask"], dtype=torch.long)

        labels_ce[i, :l] = torch.tensor(ex["labels_ce"], dtype=torch.long)
        labels_focal[i, :l] = torch.tensor(ex["labels_focal"], dtype=torch.long)
        focal_mask[i, :l] = torch.tensor(ex["focal_mask"], dtype=torch.float)

        lbl_idx[i, :l] = torch.tensor(ex["label_index_seq"], dtype=torch.long)
        hist_mask[i, :l] = torch.tensor(ex.get("hist_mask", [0]*l), dtype=torch.float)
        timestep_id_seq[i, :l] = torch.tensor(ex.get("timestep_id_seq", [0]*l), dtype=torch.long)

        targets[i] = float(ex.get("target_value", 0.0))
        types.append(ex["task_type"])

        group_ids.append(str(ex.get("group_id", "unknown")))
        timesteps.append(int(ex.get("timestep", 0)))
        chunk_starts.append(int(ex.get("chunk_start", 0)))

    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels_ce": labels_ce,
        "labels_focal": labels_focal,
        "focal_mask": focal_mask,
        "label_index_seq": lbl_idx,
        "hist_mask": hist_mask,                 # NEW
        "timestep_id_seq": timestep_id_seq,     # NEW
        "target_value": targets,
        "task_type": types,
        "group_id": group_ids,
        "timestep": timesteps,
        "chunk_start": chunk_starts,
    }


# =============================================================================
# 8) Temporal embedding + wrapper model (UPDATED: timestep emb + hist cls head)
# =============================================================================
class RelativeTemporalEmbedding(nn.Module):
    """
    OLD: temporal(pos), label, rel_pos
    NEW: + timestep embedding (absolute time id, broadcast per token)
    """
    def __init__(self, max_len: int, dim: int, max_label_vocab: int, timestep_vocab: int):
        super().__init__()
        self.temporal_emb = nn.Embedding(max_len, dim)
        self.label_emb = nn.Embedding(max_label_vocab, dim)
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, dim)
        self.timestep_emb = nn.Embedding(timestep_vocab, dim)  # NEW

        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.max_len = max_len
        self.timestep_vocab = timestep_vocab

    def forward(self, attention_mask: torch.Tensor, label_indices: torch.Tensor, timestep_id_seq: torch.Tensor) -> torch.Tensor:
        B, L = attention_mask.shape
        device = attention_mask.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        seq_len = (attention_mask.sum(1) - 1).clamp(min=0)
        rel = pos - seq_len.unsqueeze(1)

        pos_clamped = pos.clamp(0, self.max_len - 1)
        rel_shifted = (rel + self.max_len - 1).clamp(0, 2 * self.max_len - 2)

        tmp = self.temporal_emb(pos_clamped)
        lbl = self.label_emb(label_indices.clamp(0, self.label_emb.num_embeddings - 1))
        rpe = self.rel_pos_emb(rel_shifted)
        ts = self.timestep_emb(timestep_id_seq.clamp(0, self.timestep_vocab - 1))  # NEW

        h = torch.cat([tmp, lbl, rpe, ts], dim=-1)
        return self.mlp(h)

@dataclass
class LongitudinalOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    class_logits: Optional[torch.FloatTensor] = None
    hist_class_logits: Optional[torch.FloatTensor] = None  # NEW
    numeric_preds: Optional[torch.FloatTensor] = None

class LongitudinalMultiTaskModel(nn.Module):
    def __init__(self, lm, max_timestep_emb: int, temporal_dim: int, num_classes: int):
        super().__init__()
        self.lm = lm
        self.hidden = lm.config.hidden_size

        self.temporal = RelativeTemporalEmbedding(
            max_len=max_timestep_emb,
            dim=temporal_dim,
            max_label_vocab=num_classes,
            timestep_vocab=TIMESTEP_MAX_VOCAB,  # NEW
        )

        lm_dtype = lm.get_input_embeddings().weight.dtype
        self.temp_proj = nn.Linear(temporal_dim, self.hidden).to(dtype=lm_dtype)

        self.cls_head = nn.Linear(self.hidden, num_classes).to(dtype=lm_dtype)
        self.hist_cls_head = nn.Linear(self.hidden, num_classes).to(dtype=lm_dtype)  # NEW

        self.reg_head = nn.Linear(self.hidden, 1).to(dtype=lm_dtype)

        self.temporal = self.temporal.to(dtype=lm_dtype)

    def forward(self, input_ids, attention_mask, label_index_seq=None, hist_mask=None, timestep_id_seq=None):
        if label_index_seq is None:
            label_index_seq = torch.zeros_like(input_ids)
        if hist_mask is None:
            hist_mask = torch.zeros_like(input_ids, dtype=torch.float)
        if timestep_id_seq is None:
            timestep_id_seq = torch.zeros_like(input_ids)

        label_index_seq = label_index_seq.long()
        timestep_id_seq = timestep_id_seq.long()

        B, L = input_ids.shape
        device = input_ids.device

        inp_emb = self.lm.get_input_embeddings()(input_ids)
        t_emb = self.temporal(attention_mask, label_index_seq, timestep_id_seq)
        t_emb = self.temp_proj(t_emb).to(dtype=inp_emb.dtype, device=inp_emb.device)

        out = self.lm(
            inputs_embeds=inp_emb + t_emb,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden = out.hidden_states[-1]
        lm_logits = out.logits

        # CLS pooling at last stamped label position (fallback to last token)
        last_token_positions = (attention_mask.sum(1) - 1).clamp(min=0)
        mask_lbl = (label_index_seq != 0)
        idx = torch.arange(L, device=device).unsqueeze(0).expand_as(label_index_seq)
        idx_masked = idx * mask_lbl.long()
        last_label_positions = idx_masked.max(dim=1).values
        has_label = mask_lbl.any(dim=1)
        pool_positions = torch.where(has_label, last_label_positions, last_token_positions)

        pooled = hidden[torch.arange(B, device=device), pool_positions].to(dtype=self.cls_head.weight.dtype)
        c_logits = self.cls_head(pooled)

        # NEW: history-only pooled representation (mean over hist tokens)
        hm = (hist_mask > 0.5) & (attention_mask > 0)
        denom = hm.sum(dim=1).clamp(min=1).unsqueeze(1)
        hist_vec = (hidden * hm.unsqueeze(-1)).sum(dim=1) / denom
        hist_vec = hist_vec.to(dtype=self.hist_cls_head.weight.dtype)
        h_logits = self.hist_cls_head(hist_vec)

        n_preds = self.reg_head(pooled).squeeze(-1)

        return LongitudinalOutput(
            logits=lm_logits,
            hidden_states=out.hidden_states,
            class_logits=c_logits,
            hist_class_logits=h_logits,
            numeric_preds=n_preds,
        )


# =============================================================================
# 9) Loss (UPDATED: add hist focal term)
# =============================================================================
def compute_class_balanced_weights(label_ids: List[int], num_classes: int, beta: float = 0.99) -> torch.Tensor:
    import numpy as np
    counts = np.zeros(num_classes, dtype=np.int64)
    for lbl in label_ids:
        if 0 < lbl < num_classes:
            counts[lbl] += 1

    weights = np.zeros(num_classes, dtype=np.float32)
    weights[0] = 0.0
    for c in range(1, num_classes):
        if counts[c] > 0:
            eff_num = 1.0 - pow(beta, counts[c])
            weights[c] = (1 - beta) / eff_num
        else:
            weights[c] = 0.0

    real = weights[1:]
    mean_w = real.mean() if real.mean() > 0 else 1.0
    weights[1:] = weights[1:] / mean_w
    return torch.tensor(weights, dtype=torch.float)

def focal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, focal_mask: torch.Tensor, gamma: float) -> torch.Tensor:
    B, T, V = logits.shape
    logits_2d = logits.view(B * T, V)
    labels_1d = labels.view(B * T)
    mask_1d = focal_mask.view(B * T)

    valid = (labels_1d != -100) & (mask_1d > 0.5)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)

    ce = F.cross_entropy(logits_2d[valid], labels_1d[valid], reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()

def focal_classification_loss(class_logits: torch.Tensor, targets: torch.Tensor, gamma: float, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    ce = F.cross_entropy(class_logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        ce_prompt_weight: float,
        focal_output_weight: float,
        class_focal_weight: float,
        hist_class_weight: float,   # NEW
        mse_weight: float,
        gamma: float,
        class_weights: Optional[torch.Tensor],
        log_breakdown: bool = True,
    ):
        super().__init__()
        self.ce_prompt_weight = ce_prompt_weight
        self.focal_output_weight = focal_output_weight
        self.class_focal_weight = class_focal_weight
        self.hist_class_weight = hist_class_weight  # NEW
        self.mse_weight = mse_weight
        self.gamma = gamma
        self.class_weights = class_weights
        self.log_breakdown = log_breakdown

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels_ce: torch.Tensor,
        labels_focal: torch.Tensor,
        focal_mask: torch.Tensor,
        class_logits: Optional[torch.Tensor],
        hist_class_logits: Optional[torch.Tensor],   # NEW
        class_targets: Optional[torch.Tensor],
        numeric_preds: Optional[torch.Tensor],
        numeric_targets: Optional[torch.Tensor],
        class_mask: torch.Tensor,
        hist_present_mask: torch.Tensor,             # NEW
        numeric_mask: torch.Tensor,
    ):
        device = lm_logits.device
        total = torch.tensor(0.0, device=device)

        ce_prompt_val = torch.tensor(0.0, device=device)
        if self.ce_prompt_weight > 0:
            ce_prompt_val = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels_ce.view(-1),
                ignore_index=-100,
            )
            total = total + self.ce_prompt_weight * ce_prompt_val

        focal_lm_val = torch.tensor(0.0, device=device)
        if self.focal_output_weight > 0:
            focal_lm_val = focal_lm_loss(lm_logits, labels_focal, focal_mask, self.gamma)
            total = total + self.focal_output_weight * focal_lm_val

        class_focal_val = torch.tensor(0.0, device=device)
        hist_focal_val = torch.tensor(0.0, device=device)

        w = (
            self.class_weights.to(device=device, dtype=(class_logits.dtype if class_logits is not None else torch.float))
            if self.class_weights is not None
            else None
        )

        if class_logits is not None and class_targets is not None and self.class_focal_weight > 0:
            active = class_mask > 0.5
            if active.any():
                class_focal_val = focal_classification_loss(
                    class_logits=class_logits[active],
                    targets=class_targets[active],
                    gamma=self.gamma,
                    weight=w,
                )
                total = total + self.class_focal_weight * class_focal_val

        # NEW: history-only classification incentive
        if hist_class_logits is not None and class_targets is not None and self.hist_class_weight > 0:
            active = (class_mask > 0.5) & (hist_present_mask > 0.5)
            if active.any():
                hist_focal_val = focal_classification_loss(
                    class_logits=hist_class_logits[active],
                    targets=class_targets[active],
                    gamma=self.gamma,
                    weight=w,
                )
                total = total + self.hist_class_weight * hist_focal_val

        mse_val = torch.tensor(0.0, device=device)
        if numeric_preds is not None and numeric_targets is not None and self.mse_weight > 0:
            active = numeric_mask > 0.5
            if active.any():
                mse_val = F.mse_loss(numeric_preds[active], numeric_targets[active])
                total = total + self.mse_weight * mse_val

        if self.log_breakdown:
            return total, {
                "ce_prompt": float(ce_prompt_val.detach().cpu()),
                "focal_lm": float(focal_lm_val.detach().cpu()),
                "class_focal": float(class_focal_val.detach().cpu()),
                "hist_class_focal": float(hist_focal_val.detach().cpu()),
                "mse": float(mse_val.detach().cpu()),
                "total": float(total.detach().cpu()),
            }
        return total


# =============================================================================
# 10) Sequence-consistent batching (NEW)
# =============================================================================
class GroupTimeBatchSampler(Sampler[List[int]]):
    """
    Sorts dataset indices by (group_id, timestep, chunk_start), then emits batches.
    This pushes batches to be time-contiguous within the same group.
    """
    def __init__(self, dataset: HFDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        keys = []
        for i in range(len(dataset)):
            ex = dataset[i]
            keys.append((
                str(ex.get("group_id", "unknown")),
                int(ex.get("timestep", 0)),
                int(ex.get("chunk_start", 0)),
                i
            ))
        keys.sort(key=lambda x: (x[0], x[1], x[2]))
        self.sorted_indices = [k[-1] for k in keys]

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sorted_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.sorted_indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# =============================================================================
# 11) Custom Trainer (UPDATED: override train dataloader + hist loss inputs)
# =============================================================================
def _count_active_tokens(labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> int:
    active = (labels != -100)
    if mask is not None:
        active = active & (mask > 0.5)
    return int(active.sum().detach().cpu().item())

class LongitudinalTrainer(Trainer):
    def __init__(self, loss_fn: MultiTaskLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def get_train_dataloader(self):
        # NEW: sequence-consistent batching
        train_dataset = self.train_dataset
        bs = self.args.per_device_train_batch_size
        sampler = GroupTimeBatchSampler(train_dataset, batch_size=bs, drop_last=False)
        return DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_ce = inputs["labels_ce"]
        labels_focal = inputs["labels_focal"]
        focal_mask = inputs["focal_mask"]
        lbl_idx = inputs["label_index_seq"]

        hist_mask = inputs.get("hist_mask", None)
        timestep_id_seq = inputs.get("timestep_id_seq", None)

        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            label_index_seq=lbl_idx,
            hist_mask=hist_mask,
            timestep_id_seq=timestep_id_seq,
        )

        lm_logits = out.logits
        class_logits = out.class_logits
        hist_class_logits = out.hist_class_logits

        device = lm_logits.device
        B, T = labels_ce.shape

        numeric_mask = torch.zeros(B, device=device, dtype=torch.float)

        # classify only if any label stamped
        has_label = (lbl_idx != 0).any(dim=1).float().to(device)
        class_mask = has_label

        # target = last stamped label id
        mask_lbl = (lbl_idx != 0)
        idx = torch.arange(T, device=device).unsqueeze(0).expand_as(lbl_idx)
        idx_masked = idx * mask_lbl.long()
        last_label_positions = idx_masked.max(dim=1).values.clamp(min=0)
        class_targets = lbl_idx[torch.arange(B, device=device), last_label_positions].long()

        # NEW: hist present?
        if hist_mask is None:
            hist_present = torch.zeros(B, device=device, dtype=torch.float)
        else:
            hist_present = (hist_mask.to(device) > 0.5).any(dim=1).float()

        loss_out = self.loss_fn(
            lm_logits=lm_logits,
            labels_ce=labels_ce.to(device),
            labels_focal=labels_focal.to(device),
            focal_mask=focal_mask.to(device),
            class_logits=class_logits,
            hist_class_logits=hist_class_logits,
            class_targets=class_targets,
            numeric_preds=None,
            numeric_targets=None,
            class_mask=class_mask,
            hist_present_mask=hist_present,
            numeric_mask=numeric_mask,
        )

        if isinstance(loss_out, tuple) and len(loss_out) == 2:
            loss, breakdown = loss_out
        else:
            loss, breakdown = loss_out, None

        ce_tokens = _count_active_tokens(labels_ce.to(device))
        out_tokens = _count_active_tokens(labels_focal.to(device), focal_mask.to(device))
        class_chunks = int((class_mask > 0.5).sum().detach().cpu().item())
        hist_chunks = int((hist_present > 0.5).sum().detach().cpu().item())

        step = int(self.state.global_step) if self.state is not None else 0
        LOG_EVERY = int(os.environ.get("LOSS_LOG_EVERY", "5"))
        if LOG_EVERY > 0 and (step % LOG_EVERY == 0):
            msg = f"[LOSS DBG] step={step} B={B} ce_tokens={ce_tokens} out_tokens={out_tokens} class_chunks={class_chunks} hist_present={hist_chunks}"
            if breakdown is not None:
                msg += (
                    f" | total={breakdown['total']:.4f} "
                    f"ce_prompt={breakdown['ce_prompt']:.4f} "
                    f"focal_lm={breakdown['focal_lm']:.4f} "
                    f"class_focal={breakdown['class_focal']:.4f} "
                    f"hist_class_focal={breakdown['hist_class_focal']:.4f}"
                )
            print(msg, flush=True)

        return (loss, out) if return_outputs else loss


# =============================================================================
# 12) Eval helper (classification)  (unchanged except pass hist/timestep fields)
# =============================================================================
def compute_classification_metrics(preds, labels):
    if not preds:
        return {"acc": 0.0, "macro_f1": 0.0}
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    acc = correct / len(preds)
    classes = sorted(set(labels))
    f1s = []
    for c in classes:
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return {"acc": acc, "macro_f1": sum(f1s) / len(f1s)}

def evaluate_stage_model(model, eval_dataset, batch_size=4):
    model.eval()
    model.to(DEVICE)
    loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=custom_data_collator)

    preds_cls, lbls_cls = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl_idx = batch["label_index_seq"].to(DEVICE)
            hmask = batch.get("hist_mask", None)
            ts_seq = batch.get("timestep_id_seq", None)

            out = model(
                input_ids=ids,
                attention_mask=mask,
                label_index_seq=lbl_idx,
                hist_mask=(hmask.to(DEVICE) if hmask is not None else None),
                timestep_id_seq=(ts_seq.to(DEVICE) if ts_seq is not None else None),
            )
            class_logits = out.class_logits

            B = ids.size(0)
            T = ids.size(1)
            mask_lbl = (lbl_idx != 0)
            idx = torch.arange(T, device=DEVICE).unsqueeze(0).expand_as(lbl_idx)
            idx_masked = idx * mask_lbl.long()
            last_label_positions = idx_masked.max(dim=1).values.clamp(min=0)
            class_targets = lbl_idx[torch.arange(B, device=DEVICE), last_label_positions]

            for i in range(B):
                if class_targets[i].item() == 0:
                    continue
                preds_cls.append(int(torch.argmax(class_logits[i]).item()))
                lbls_cls.append(int(class_targets[i].item()))

    out_metrics = {}
    if lbls_cls:
        out_metrics.update({f"class_{k}": v for k, v in compute_classification_metrics(preds_cls, lbls_cls).items()})
    return out_metrics


# =============================================================================
# 13) LoRA schedule + resume state (UNCHANGED)
# =============================================================================
STAGE_SCHEDULE = {
    1: {"lora_r": 4,  "lr": 2e-4, "scheduler": "cosine"},
    2: {"lora_r": 8,  "lr": 1e-4, "scheduler": "cosine"},
    3: {"lora_r": 16, "lr": 5e-5, "scheduler": "constant"},
}

def stage_cfg(stage_idx: int) -> Dict[str, Any]:
    return STAGE_SCHEDULE.get(stage_idx, STAGE_SCHEDULE[max(STAGE_SCHEDULE)])

def build_lora_config_for_stage(stage_idx: int) -> LoraConfig:
    cfg = stage_cfg(stage_idx)
    r = int(cfg["lora_r"])
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=CONFIG.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_stage_state_s3() -> Dict[str, Any]:
    txt = s3_read_text(STAGE_STATE_S3)
    if not txt:
        return {"last_completed_stage": 0}
    try:
        return json.loads(txt)
    except Exception:
        return {"last_completed_stage": 0}

def save_stage_state_s3(last_completed_stage: int, last_stage_s3_dir: Optional[str] = None,
                        last_lora_s3_dir: Optional[str] = None, last_heads_s3_path: Optional[str] = None) -> None:
    payload = {"last_completed_stage": int(last_completed_stage)}
    if last_stage_s3_dir: payload["last_stage_s3_dir"] = str(last_stage_s3_dir)
    if last_lora_s3_dir: payload["last_lora_s3_dir"] = str(last_lora_s3_dir)
    if last_heads_s3_path: payload["last_heads_s3_path"] = str(last_heads_s3_path)
    payload["updated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    s3_write_text_atomic(STAGE_STATE_S3, json.dumps(payload, indent=2))

def _load_adapter_state_dict(adapter_dir: str) -> Dict[str, torch.Tensor]:
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if _HAS_SAFETENSORS and os.path.exists(st_path):
        return safetensors_load_file(st_path)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

def load_lora_with_rank_resize(peft_model, prev_adapter_dir: str) -> None:
    prev_sd = _load_adapter_state_dict(prev_adapter_dir)
    curr_sd = peft_model.state_dict()
    updated = {}

    for k, curr_v in curr_sd.items():
        if k not in prev_sd:
            continue
        prev_v = prev_sd[k]
        if not (isinstance(curr_v, torch.Tensor) and isinstance(prev_v, torch.Tensor)):
            continue
        if curr_v.dtype != prev_v.dtype:
            prev_v = prev_v.to(curr_v.dtype)

        if curr_v.shape == prev_v.shape:
            updated[k] = prev_v
            continue

        if curr_v.ndim == 2 and prev_v.ndim == 2:
            c0, c1 = curr_v.shape
            p0, p1 = prev_v.shape
            new_v = curr_v.detach().clone()
            rr0 = min(c0, p0)
            rr1 = min(c1, p1)
            new_v[:rr0, :rr1] = prev_v[:rr0, :rr1]
            updated[k] = new_v

    peft_model.load_state_dict(updated, strict=False)
    log_event(f"Loaded previous LoRA with rank-resize from: {prev_adapter_dir} (updated {len(updated)} tensors)")

def freeze_old_lora_ranks(peft_model, old_rank: int):
    log_event(f"Freezing old LoRA ranks [0:{old_rank}]")
    for name, param in peft_model.named_parameters():
        if "lora_A" in name and param.ndim == 2 and param.shape[0] >= old_rank:
            param.register_hook(lambda grad, old_rank=old_rank: torch.cat([torch.zeros_like(grad[:old_rank]), grad[old_rank:]], dim=0))
        if "lora_B" in name and param.ndim == 2 and param.shape[1] >= old_rank:
            param.register_hook(lambda grad, old_rank=old_rank: torch.cat([torch.zeros_like(grad[:, :old_rank]), grad[:, old_rank:]], dim=1))


# =============================================================================
# 14) Wrapper heads save/load (UPDATED: include hist_cls_head + timestep vocab metadata)
# =============================================================================
def atomic_torch_save(obj, final_path: str) -> None:
    ensure_dir(os.path.dirname(final_path) or ".")
    tmp_path = final_path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)

def get_wrapper_heads_state(model: nn.Module) -> Dict[str, Any]:
    return {
        "temporal": model.temporal.state_dict(),
        "temp_proj": model.temp_proj.state_dict(),
        "cls_head": model.cls_head.state_dict(),
        "hist_cls_head": model.hist_cls_head.state_dict(),  # NEW
        "reg_head": model.reg_head.state_dict(),
        "meta": {
            "num_classes": int(getattr(model.cls_head, "out_features", -1)),
            "temporal_dim": int(getattr(model.temp_proj, "in_features", -1)),
            "hidden": int(getattr(model.temp_proj, "out_features", -1)),
            "temporal_max_len": int(getattr(model.temporal, "max_len", -1)),
            "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),     # NEW
        }
    }

def load_wrapper_heads_state(model: nn.Module, path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            if "temporal" in sd: model.temporal.load_state_dict(sd["temporal"], strict=False)
            if "temp_proj" in sd: model.temp_proj.load_state_dict(sd["temp_proj"], strict=False)
            if "cls_head" in sd: model.cls_head.load_state_dict(sd["cls_head"], strict=False)
            if "hist_cls_head" in sd and hasattr(model, "hist_cls_head"):
                model.hist_cls_head.load_state_dict(sd["hist_cls_head"], strict=False)
            if "reg_head" in sd: model.reg_head.load_state_dict(sd["reg_head"], strict=False)
        return True
    except Exception as e:
        log_event(f"[WARN] Failed to load wrapper heads from {path}: {repr(e)}")
        return False


# =============================================================================
# 15) S3 checkpoint callback (UNCHANGED)
# =============================================================================
class S3CheckpointCallback(TrainerCallback):
    def __init__(self, stage_s3_dir: str, keep_last_local: int = 1):
        self.stage_s3_dir = stage_s3_dir.rstrip("/")
        self.keep_last_local = keep_last_local

    def on_save(self, args, state, control, **kwargs):
        out_dir = args.output_dir
        ckpt_name = f"checkpoint-{state.global_step}"
        local_ckpt = os.path.join(out_dir, ckpt_name)

        if os.path.isdir(local_ckpt):
            s3_ckpt = s3_join(self.stage_s3_dir, "checkpoints", ckpt_name)
            s3_put_dir(local_ckpt, s3_ckpt, recursive=True)

        try:
            ckpts = []
            for name in os.listdir(out_dir):
                if name.startswith("checkpoint-"):
                    step = int(name.split("-")[-1])
                    ckpts.append((step, name))
            ckpts.sort(key=lambda x: x[0])

            to_delete = ckpts[:-self.keep_last_local] if self.keep_last_local > 0 else ckpts
            for _, nm in to_delete:
                safe_rmtree(os.path.join(out_dir, nm))
        except Exception:
            pass

        return control


# =============================================================================
# 16) Tokenizer upload + latest manifest (UNCHANGED)
# =============================================================================
def upload_tokenizer_to_s3_once() -> None:
    manifest_path = s3_join(TOKENIZER_S3_DIR, "tokenizer_manifest.json")
    if s3_exists(manifest_path):
        log_event(f"Tokenizer already present in S3; skipping upload: {TOKENIZER_S3_DIR}")
        return

    local_tok = os.path.join(IFT_WORKDIR, "tokenizer_local")
    safe_rmtree(local_tok)
    ensure_dir(local_tok)

    tokenizer.save_pretrained(local_tok)
    s3_put_dir(local_tok, TOKENIZER_S3_DIR, recursive=True)
    safe_rmtree(local_tok)

    extra = {
        "base_model_name": BASE_MODEL_NAME,
        "len_tokenizer": int(len(tokenizer)),
        "special_tokens": SPECIAL_TOKENS,
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    s3_write_text_atomic(manifest_path, json.dumps(extra, indent=2))
    log_event(f"Uploaded tokenizer to S3: {TOKENIZER_S3_DIR}")

def write_latest_manifest(stage_s3_dir: str, info: Dict[str, Any]) -> None:
    payload = dict(info)
    payload["updated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    s3_write_text_atomic(LATEST_MANIFEST_S3, json.dumps(payload, indent=2))
    log_event(f"Wrote latest manifest: {LATEST_MANIFEST_S3}")


# =============================================================================
# 17) Builder sanity check (UNCHANGED)
# =============================================================================
def sanity_check_builder_signature() -> None:
    txt = s3_read_text(BUILDER_STATS_S3)
    if not txt:
        log_event(f"[WARN] builder_stats.json not found at {BUILDER_STATS_S3} (skipping signature checks)")
        return
    try:
        stats = json.loads(txt)
    except Exception:
        log_event(f"[WARN] builder_stats.json unreadable at {BUILDER_STATS_S3} (skipping signature checks)")
        return

    expected_prefix = stats.get("curriculum_prefix_s3")
    if expected_prefix and str(expected_prefix).rstrip("/") != CURRICULUM_PREFIX_S3.rstrip("/"):
        raise RuntimeError(
            f"CURRICULUM_PREFIX_S3 mismatch.\ntrainer={CURRICULUM_PREFIX_S3}\nbuilder_stats={expected_prefix}"
        )

    for p in CURRICULUM_STAGE_FILES_S3:
        if not s3_exists(p):
            raise FileNotFoundError(f"Expected curriculum stage file missing: {p}")

    log_event("Builder signature sanity checks: OK")


# =============================================================================
# 18) Training report helpers (UNCHANGED)
# =============================================================================
def _load_training_report() -> Dict[str, Any]:
    txt = s3_read_text(TRAINING_REPORT_S3)
    if not txt:
        return {"runs": [], "latest": None}
    try:
        return json.loads(txt)
    except Exception:
        return {"runs": [], "latest": None}

def _save_training_report(report: Dict[str, Any]) -> None:
    s3_write_text_atomic(TRAINING_REPORT_S3, json.dumps(report, indent=2))

def _update_training_report(stage_entry: Dict[str, Any]) -> None:
    report = _load_training_report()
    runs = report.get("runs", [])
    runs = [r for r in runs if not (r.get("stage_idx") == stage_entry.get("stage_idx") and r.get("stage_s3_dir") == stage_entry.get("stage_s3_dir"))]
    runs.append(stage_entry)
    runs.sort(key=lambda x: (int(x.get("stage_idx", 0)), str(x.get("stage_s3_dir", ""))))
    report["runs"] = runs
    report["latest"] = stage_entry
    report["updated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    _save_training_report(report)


# =============================================================================
# 19) Curriculum sanity summary (UNCHANGED except it now sees group/timestep in items)
# =============================================================================
def _write_curriculum_summary(
    stage_s3_dir: str,
    builder_items: List[Dict[str, Any]],
    balanced_items: List[Dict[str, Any]],
    chunks: HFDataset,
) -> None:
    from collections import Counter

    def _cnt(items, key_fn):
        return dict(Counter([key_fn(x) for x in items]))

    tasks_raw = _cnt(builder_items, lambda x: str(x.get("task_type", "unknown")))
    k_raw = _cnt(builder_items, lambda x: int(x.get("k_shot_used", 0)))
    raw_label_raw = _cnt(builder_items, lambda x: int(x.get("raw_label_id", x.get("label_id", 0))))

    tasks_bal = _cnt(balanced_items, lambda x: str(x.get("task_type", "unknown")))
    k_bal = _cnt(balanced_items, lambda x: int(x.get("k_shot_used", 0)))
    raw_label_bal = _cnt(balanced_items, lambda x: int(x.get("raw_label_id", x.get("label_id", 0))))

    std = [standardize_item(x) for x in balanced_items]
    glob_label_bal = _cnt(std, lambda x: int(x.get("label_id", 0)))

    token_lens = []
    try:
        for ex in chunks:
            token_lens.append(int(sum(ex["attention_mask"])))
    except Exception:
        pass

    summary = {
        "stage_s3_dir": stage_s3_dir,
        "curriculum_prefix_s3": CURRICULUM_PREFIX_S3,
        "counts_raw": {
            "examples": int(len(builder_items)),
            "by_task_type": tasks_raw,
            "by_k_shot_used": k_raw,
            "by_raw_label_id": raw_label_raw,
        },
        "counts_balanced": {
            "examples": int(len(balanced_items)),
            "by_task_type": tasks_bal,
            "by_k_shot_used": k_bal,
            "by_raw_label_id": raw_label_bal,
            "by_global_label_id": glob_label_bal,
        },
        "chunking": {
            "chunks": int(len(chunks)),
            "tokens_min": int(min(token_lens)) if token_lens else 0,
            "tokens_max": int(max(token_lens)) if token_lens else 0,
            "tokens_mean": float(sum(token_lens) / len(token_lens)) if token_lens else 0.0,
        },
        "label_namespace": {
            "label_pad_id": 0,
            "global_label_map_size": int(len(GLOBAL_LABEL_INV)),
            "task_labels": TASK_LABELS,
        },
        "longitudinal": {
            "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
            "hist_cls_weight": float(HIST_CLS_WEIGHT),
            "sequence_consistent_batching": True,
        },
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    s3_write_text_atomic(s3_join(stage_s3_dir, "curriculum_summary.json"), json.dumps(summary, indent=2))


# =============================================================================
# 20) Best pointer + best export + canonical best paths (UNCHANGED)
# =============================================================================
def _save_log_history_to_s3(trainer: Trainer, stage_s3_dir: str, out_dir_local: str) -> None:
    hist = getattr(trainer.state, "log_history", None)
    if not hist:
        return
    local_path = os.path.join(out_dir_local, "log_history.json")
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    s3_put_file(local_path, s3_join(stage_s3_dir, "log_history.json"))

def _write_best_pointer(stage_s3_dir: str, trainer: Trainer) -> Dict[str, Any]:
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    best_step = None
    if best_ckpt and isinstance(best_ckpt, str) and "checkpoint-" in best_ckpt:
        try:
            best_step = int(best_ckpt.split("checkpoint-")[-1].split("/")[0])
        except Exception:
            best_step = None

    payload = {
        "best_model_checkpoint": best_ckpt,
        "best_metric": float(best_metric) if isinstance(best_metric, (int, float)) else best_metric,
        "best_global_step": best_step,
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    s3_write_text_atomic(s3_join(stage_s3_dir, "BEST.json"), json.dumps(payload, indent=2))
    return payload

def _export_best_weights_to_s3(trainer: Trainer, stage_s3_dir: str, out_dir_local: str) -> None:
    best_export_local = os.path.join(out_dir_local, "best_export")
    safe_rmtree(best_export_local)
    ensure_dir(best_export_local)

    lora_local = os.path.join(best_export_local, "lora_model")
    ensure_dir(lora_local)
    trainer.model.lm.save_pretrained(lora_local)
    s3_put_dir(lora_local, s3_join(stage_s3_dir, "best_export", "lora_model"), recursive=True)

    heads_local = os.path.join(best_export_local, "wrapper_heads.pt")
    atomic_torch_save(get_wrapper_heads_state(trainer.model), heads_local)
    s3_put_file(heads_local, s3_join(stage_s3_dir, "best_export", "wrapper_heads.pt"))

def _export_best_weights_canonical_to_s3(trainer: Trainer, stage_s3_dir: str, out_dir_local: str) -> None:
    best_canon_local = os.path.join(out_dir_local, "best_canonical")
    safe_rmtree(best_canon_local)
    ensure_dir(best_canon_local)

    lora_best_local = os.path.join(best_canon_local, "lora_model_best")
    ensure_dir(lora_best_local)
    trainer.model.lm.save_pretrained(lora_best_local)
    s3_put_dir(lora_best_local, s3_join(stage_s3_dir, "lora_model_best"), recursive=True)

    heads_best_local = os.path.join(best_canon_local, "wrapper_heads_best.pt")
    atomic_torch_save(get_wrapper_heads_state(trainer.model), heads_best_local)
    s3_put_file(heads_best_local, s3_join(stage_s3_dir, "wrapper_heads_best.pt"))


# =============================================================================
# 21) Train loop + resume safety (UPDATED only where needed)
# =============================================================================
def _assert_no_bitsandbytes(model) -> None:
    for _, m in model.named_modules():
        mod_name = m.__class__.__name__.lower()
        mod_pkg = m.__class__.__module__.lower()
        if "bitsandbytes" in mod_pkg or "8bit" in mod_name or "4bit" in mod_name:
            raise RuntimeError("Detected bitsandbytes module. This pipeline forbids bitsandbytes.")

def rehydrate_previous_artifacts_from_state(state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    last_done = int(state.get("last_completed_stage", 0))
    lora_s3 = state.get("last_lora_s3_dir")
    heads_s3 = state.get("last_heads_s3_path")

    if last_done <= 0 or not lora_s3 or not heads_s3:
        return None, None

    prev_lora_local = os.path.join(IFT_WORKDIR, "rehydrated_prev_lora")
    prev_heads_local = os.path.join(IFT_WORKDIR, "rehydrated_prev_wrapper_heads.pt")

    safe_rmtree(prev_lora_local)
    try:
        os.remove(prev_heads_local)
    except Exception:
        pass

    s3_get_dir(lora_s3, prev_lora_local)
    s3_get_file(heads_s3, prev_heads_local)
    log_event(f"Rehydrated previous artifacts from S3: stage={last_done}")
    return prev_lora_local, prev_heads_local

def train_curriculum():
    sanity_check_builder_signature()
    upload_tokenizer_to_s3_once()

    state = load_stage_state_s3()
    last_done = int(state.get("last_completed_stage", 0))
    prev_lora_local, prev_heads_local = rehydrate_previous_artifacts_from_state(state)

    for idx, s3_jfile in enumerate(CONFIG.json_sources_s3):
        stage_idx = idx + 1
        stage_tag = os.path.basename(s3_jfile).replace(".jsonl", "").replace(".json", "")
        stage_s3_dir = s3_join(OUTPUT_S3, f"stage_{stage_idx}_{stage_tag}")
        out_dir_local = os.path.join(CONFIG.output_base_dir_local, f"stage_{stage_idx}_{stage_tag}")
        ensure_dir(out_dir_local)

        if stage_idx <= last_done:
            log_event(f"Skipping stage {stage_idx} (already completed).")
            continue

        cfg = stage_cfg(stage_idx)
        log_event(f"================ stage {stage_idx} ================")
        log_event(f"Using S3 file: {s3_jfile}")
        log_event(f"LoRA r={cfg['lora_r']} | LR={cfg['lr']} | scheduler={cfg['scheduler']}")
        log_event(f"Local out_dir: {out_dir_local}")
        log_event(f"S3 stage dir:  {stage_s3_dir}")

        builder_items = read_jsonl_s3(s3_jfile)
        if not builder_items:
            log_event("No items in this stage file; marking stage complete.")
            save_stage_state_s3(
                last_completed_stage=stage_idx,
                last_stage_s3_dir=stage_s3_dir,
                last_lora_s3_dir=state.get("last_lora_s3_dir"),
                last_heads_s3_path=state.get("last_heads_s3_path"),
            )
            continue

        rng = random.Random(10_000 + stage_idx)
        balanced = balance_dataset(builder_items, rng=rng, kshot_bins=list(range(0, 15)))
        raw_data = [standardize_item(x) for x in balanced]

        # class weights
        label_ids_for_weights = [ex["label_id"] for ex in raw_data if ex["label_id"] != 0]
        class_weights = compute_class_balanced_weights(label_ids_for_weights, CONFIG.num_classes) if label_ids_for_weights else None

        base = AutoModelForCausalLM.from_pretrained(
            CONFIG.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        base.resize_token_embeddings(len(tokenizer))
        try:
            base.gradient_checkpointing_enable()
            log_event("Enabled gradient checkpointing.")
        except Exception:
            pass

        lora_cfg = build_lora_config_for_stage(stage_idx)
        base = get_peft_model(base, lora_cfg)
        _assert_no_bitsandbytes(base)

        if prev_lora_local and os.path.isdir(prev_lora_local):
            try:
                load_lora_with_rank_resize(base, prev_lora_local)
                if stage_idx > 1:
                    prev_rank = stage_cfg(stage_idx - 1)["lora_r"]
                    curr_rank = stage_cfg(stage_idx)["lora_r"]
                    if curr_rank > prev_rank:
                        freeze_old_lora_ranks(base, old_rank=prev_rank)
            except Exception as e:
                log_event(f"[WARN] LoRA load failed: {e}")

        lm_ctx = getattr(base.config, "max_position_embeddings", LM_CONTEXT_WINDOW)
        if not isinstance(lm_ctx, int) or lm_ctx <= 0 or lm_ctx > 10_000_000:
            lm_ctx = LM_CONTEXT_WINDOW
        CONFIG.max_timestep_emb = min(int(lm_ctx), int(TEMPORAL_MAX_LEN))

        model = LongitudinalMultiTaskModel(
            lm=base,
            max_timestep_emb=CONFIG.max_timestep_emb,
            temporal_dim=CONFIG.temporal_embed_dim,
            num_classes=CONFIG.num_classes,
        )

        if prev_heads_local and os.path.exists(prev_heads_local):
            ok = load_wrapper_heads_state(model, prev_heads_local)
            log_event(f"Loaded previous wrapper heads: {ok}")

        chunks = tokenize_dataset_dynamic(
            raw_data,
            tokenizer,
            min_stride=CONFIG.min_stride,
            max_stride=CONFIG.max_stride,
            model=model.lm,
        )
        log_event(f"Tokenized chunks={len(chunks)}")

        _write_curriculum_summary(
            stage_s3_dir=stage_s3_dir,
            builder_items=builder_items,
            balanced_items=balanced,
            chunks=chunks,
        )

        if len(chunks) > 1 and CONFIG.val_fraction > 0:
            split = chunks.train_test_split(CONFIG.val_fraction, seed=42)
            train_ds, eval_ds = split["train"], split["test"]
            log_event(f"Train={len(train_ds)}, Eval={len(eval_ds)}")
        else:
            train_ds, eval_ds = chunks, None
            log_event("No validation split.")

        mt_loss = MultiTaskLoss(
            ce_prompt_weight=CONFIG.ce_prompt_weight,
            focal_output_weight=CONFIG.focal_output_weight,
            class_focal_weight=CONFIG.class_focal_weight,
            hist_class_weight=CONFIG.hist_class_weight,   # NEW
            mse_weight=CONFIG.mse_weight,
            gamma=CONFIG.focal_gamma,
            class_weights=class_weights,
            log_breakdown=True,
        )

        save_steps = 5
        args = TrainingArguments(
            output_dir=out_dir_local,

            save_strategy="steps",
            save_steps=save_steps,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=save_steps if eval_ds is not None else None,
            logging_steps=save_steps,

            load_best_model_at_end=True if eval_ds is not None else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # save_safetensors=False,
            save_only_model=True,
            save_total_limit=2,

            learning_rate=float(cfg["lr"]),
            lr_scheduler_type=str(cfg["scheduler"]),
            num_train_epochs=CONFIG.num_train_epochs,
            per_device_train_batch_size=CONFIG.per_device_train_batch_size,
            gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,

            logging_dir=os.path.join(out_dir_local, "logs"),
            report_to=["tensorboard"],

            bf16=torch.cuda.is_available(),
            fp16=False,
            optim="adamw_torch",
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            max_grad_norm=1.0,
        )

        s3_cb = S3CheckpointCallback(stage_s3_dir=stage_s3_dir, keep_last_local=1)

        trainer = LongitudinalTrainer(
            loss_fn=mt_loss,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds if eval_ds is not None else None,
            data_collator=custom_data_collator,
            callbacks=[s3_cb],
        )

        log_event(f"Training stage {stage_idx}... (LOSS_LOG_EVERY={os.environ.get('LOSS_LOG_EVERY','5')})")
        trainer.train()

        _save_log_history_to_s3(trainer, stage_s3_dir=stage_s3_dir, out_dir_local=out_dir_local)

        best_payload = _write_best_pointer(stage_s3_dir, trainer)

        if eval_ds is not None and args.load_best_model_at_end:
            log_event("Exporting BEST weights (best_export + canonical best paths) ...")
            _export_best_weights_to_s3(trainer, stage_s3_dir=stage_s3_dir, out_dir_local=out_dir_local)
            _export_best_weights_canonical_to_s3(trainer, stage_s3_dir=stage_s3_dir, out_dir_local=out_dir_local)

        eval_loss_payload: Dict[str, Any] = {}
        if eval_ds is not None:
            log_event("Computing final validation loss via trainer.evaluate() ...")
            eval_out = trainer.evaluate(eval_dataset=eval_ds)
            eval_loss_payload = {
                "eval_loss": float(eval_out.get("eval_loss", float("nan"))),
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in eval_out.items() if k != "eval_loss"},
                "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
            }
            local_eval_loss = os.path.join(out_dir_local, "eval_loss.json")
            with open(local_eval_loss, "w", encoding="utf-8") as f:
                json.dump(eval_loss_payload, f, indent=2)
            s3_put_file(local_eval_loss, s3_join(stage_s3_dir, "eval_loss.json"))
            log_event(f"Final eval_loss={eval_loss_payload['eval_loss']}")

        metrics = {}
        if eval_ds is not None:
            log_event("Evaluating classification metrics...")
            metrics = evaluate_stage_model(model, eval_ds)
            local_metrics = os.path.join(out_dir_local, "eval_metrics.json")
            with open(local_metrics, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            s3_put_file(local_metrics, s3_join(stage_s3_dir, "eval_metrics.json"))

        # Save end-of-stage LoRA + heads
        lora_local = os.path.join(out_dir_local, "lora_model")
        ensure_dir(lora_local)
        trainer.model.lm.save_pretrained(lora_local)
        s3_put_dir(lora_local, s3_join(stage_s3_dir, "lora_model"), recursive=True)

        heads_local = os.path.join(out_dir_local, "wrapper_heads.pt")
        atomic_torch_save(get_wrapper_heads_state(trainer.model), heads_local)
        s3_put_file(heads_local, s3_join(stage_s3_dir, "wrapper_heads.pt"))

        logs_local = os.path.join(out_dir_local, "logs")
        if os.path.isdir(logs_local):
            s3_put_dir(logs_local, s3_join(stage_s3_dir, "logs"), recursive=True)

        manifest = {
            "base_model_name": CONFIG.model_name,
            "tokenizer_s3_dir": TOKENIZER_S3_DIR,
            "stage_s3_dir": stage_s3_dir,
            "lora_s3_dir": s3_join(stage_s3_dir, "lora_model"),
            "wrapper_heads_s3_path": s3_join(stage_s3_dir, "wrapper_heads.pt"),

            "best_pointer_s3": s3_join(stage_s3_dir, "BEST.json"),
            "best_model_checkpoint_local": best_payload.get("best_model_checkpoint"),
            "best_metric": best_payload.get("best_metric"),
            "best_global_step": best_payload.get("best_global_step"),

            "best_export_s3_dir": s3_join(stage_s3_dir, "best_export"),
            "lora_s3_dir_best": s3_join(stage_s3_dir, "lora_model_best"),
            "wrapper_heads_s3_path_best": s3_join(stage_s3_dir, "wrapper_heads_best.pt"),

            "curriculum_summary_s3": s3_join(stage_s3_dir, "curriculum_summary.json"),

            "num_classes": int(CONFIG.num_classes),
            "temporal_dim": int(CONFIG.temporal_embed_dim),
            "temporal_max_len": int(CONFIG.max_timestep_emb),
            "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),           # NEW
            "hist_cls_weight": float(CONFIG.hist_class_weight),      # NEW
            "sequence_consistent_batching": True,                    # NEW

            "max_tokens_per_input": int(MAX_TOKENS_PER_INPUT),
            "builder_cap": int(MAX_CONTEXT_FOR_BUILDER),
            "special_tokens": SPECIAL_TOKENS,
            "global_label_map_size": int(len(GLOBAL_LABEL_INV)),

            "metrics": metrics,
            "eval_loss": eval_loss_payload.get("eval_loss", None),
            "eval_loss_details": eval_loss_payload,

            "curriculum_prefix_s3": CURRICULUM_PREFIX_S3,
            "stage_file_s3": s3_jfile,
            "log_history_s3": s3_join(stage_s3_dir, "log_history.json"),
        }

        s3_write_text_atomic(s3_join(stage_s3_dir, "inference_manifest.json"), json.dumps(manifest, indent=2))
        write_latest_manifest(stage_s3_dir=stage_s3_dir, info=manifest)

        stage_entry = {
            "stage_idx": int(stage_idx),
            "stage_file_s3": s3_jfile,
            "stage_s3_dir": stage_s3_dir,
            "curriculum_prefix_s3": CURRICULUM_PREFIX_S3,
            "best_pointer_s3": s3_join(stage_s3_dir, "BEST.json"),
            "best_model_checkpoint_local": best_payload.get("best_model_checkpoint"),
            "best_metric": best_payload.get("best_metric"),
            "best_global_step": best_payload.get("best_global_step"),
            "lora_s3_dir_best": s3_join(stage_s3_dir, "lora_model_best"),
            "wrapper_heads_s3_path_best": s3_join(stage_s3_dir, "wrapper_heads_best.pt"),
            "lora_s3_dir_end": s3_join(stage_s3_dir, "lora_model"),
            "wrapper_heads_s3_path_end": s3_join(stage_s3_dir, "wrapper_heads.pt"),
            "eval_loss": eval_loss_payload.get("eval_loss", None),
            "metrics": metrics,
            "curriculum_summary_s3": s3_join(stage_s3_dir, "curriculum_summary.json"),
            "longitudinal": {
                "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
                "hist_cls_weight": float(CONFIG.hist_class_weight),
                "sequence_consistent_batching": True,
            },
            "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        _update_training_report(stage_entry)

        prev_lora_local = lora_local
        prev_heads_local = heads_local

        save_stage_state_s3(
            last_completed_stage=stage_idx,
            last_stage_s3_dir=stage_s3_dir,
            last_lora_s3_dir=s3_join(stage_s3_dir, "lora_model"),
            last_heads_s3_path=s3_join(stage_s3_dir, "wrapper_heads.pt"),
        )
        log_event(f"Completed stage {stage_idx}; updated S3 state: {STAGE_STATE_S3}")

        # Clean local (keep only lora + heads)
        for name in os.listdir(out_dir_local):
            p = os.path.join(out_dir_local, name)
            if p in [lora_local, heads_local]:
                continue
            if os.path.isdir(p):
                safe_rmtree(p)
            elif os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    log_event("Curriculum training complete.")


# =============================================================================
# 22) Main
# =============================================================================
def main():
    log_event("START: iFT TRAINER ONLY (S3-first) — LONGITUDINAL UPDATED")
    log_event(f"CURRICULUM_PREFIX_S3={CURRICULUM_PREFIX_S3}")
    log_event(f"Stage files: {CONFIG.json_sources_s3}")
    train_curriculum()
    log_event(f"Wrote training report: {TRAINING_REPORT_S3}")
    log_event("DONE.")

if __name__ == "__main__":
    main()


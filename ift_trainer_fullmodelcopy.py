#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
IFT TRAINER (S3-first) — Longitudinal iFT
LoRA + Temporal/Label Embeddings + CLS + FULL MERGED MODEL EXPORT
+ FINAL CONTINUOUS MODEL EXPORT
+ INTERRUPTION-SAFE RESUME
================================================================================

TRAINER-ONLY SCRIPT

This script:
  1) Reads curriculum JSONL stage files from S3
  2) Tokenizes + chunks them with strict masking
  3) Trains a LoRA-adapted causal LM with:
       - temporal embeddings
       - timestep embeddings
       - label embeddings
       - global classification head
       - history-only auxiliary classification head
  4) Uploads all training artifacts to S3
  5) Safely resumes from S3 state
  6) Exports a full merged standalone LM in addition to LoRA adapters
  7) Exports the final continuously trained model across stage1->stage2->stage3
     into a canonical Final_Model folder
  8) NEW: Resumes correctly after interruption at TWO levels:
       - stage-level resume
       - within-stage checkpoint resume

-------------------------------------------------------------------------------
INTERRUPTION / RESTART SAFETY
-------------------------------------------------------------------------------
This script now supports robust restart behavior.

A) Stage-level resume
   If stage 1 completed and the process later restarts, training continues from
   stage 2. If stage 2 completed, it continues from stage 3.

B) Mid-stage resume
   If training crashes or stops DURING a stage, the script will:
     - discover the latest uploaded HF checkpoint on S3 for that stage
     - download it locally
     - resume Trainer from that checkpoint
     - restore model, optimizer, scheduler, RNG, and global_step state

This only works correctly because:
  - save_only_model=False
  - checkpoints are uploaded to S3 recursively
  - Trainer resumes from resume_from_checkpoint=...

-------------------------------------------------------------------------------
WHAT "FINAL CONTINUOUS MODEL" MEANS
-------------------------------------------------------------------------------
Training is sequential:
  stage 1 -> stage 2 -> stage 3

Each later stage starts from the previous stage's trained LoRA state + wrapper heads.
So the stage 3 end state represents the continuously learned model across all stages.

This script exports that final continuous state to:

  Final_Model/
    lora_model/
    wrapper_heads.pt
    full_model/                    # merged base + LoRA baked in
    wrapper_heads_full.pt
    final_inference_manifest.json
    LATEST_FINAL.json

-------------------------------------------------------------------------------
IMPORTANT EXPORT NOTES
-------------------------------------------------------------------------------
There are now 3 export layers:

1) END-OF-STAGE EXPORTS
   Per stage:
     - lora_model/
     - wrapper_heads.pt
     - full_model/
     - wrapper_heads_full.pt

2) BEST-WEIGHTS EXPORTS
   Per stage:
     - lora_model_best/
     - wrapper_heads_best.pt
     - full_model_best/
     - wrapper_heads_best_full.pt

3) FINAL CONTINUOUS EXPORT
   Single canonical output:
     - Final_Model/

The "full_model/" folders only contain the merged causal LM.
Your custom wrapper heads remain separate sidecar weights because the top-level
wrapper is a plain nn.Module rather than a Hugging Face PreTrainedModel subclass.

So a full deployable package is:
  - full_model/
  - wrapper_heads_full.pt

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

from peft import LoraConfig, get_peft_model, PeftModel

import fsspec

try:
    from safetensors.torch import load_file as safetensors_load_file
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False


# =============================================================================
# 1) Reproducibility + environment paths
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
    f"{CURRICULUM_S3}/ICL_curriculum_IFT_llama8b",
).rstrip("/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOW_MISSING_STAGE = os.environ.get("ALLOW_MISSING_STAGE", "0") == "1"

# Whether to produce merged full-model exports in addition to adapter exports.
SAVE_FULL_MODEL_EXPORTS = os.environ.get("SAVE_FULL_MODEL_EXPORTS", "1") == "1"


# =============================================================================
# 2) S3 filesystem
# =============================================================================
def env_required(name: str) -> str:
    """
    Read a required environment variable or raise a clear error.
    """
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
    """
    Join S3 path fragments without accidental duplicate slashes.
    """
    prefix = prefix.rstrip("/")
    rest = "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None])
    return f"{prefix}/{rest}" if rest else prefix


def ensure_dir(path: str) -> None:
    """
    Create a local directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def log_event(message: str) -> None:
    """
    Timestamped log helper for long-running jobs.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def safe_rmtree(path: str) -> None:
    """
    Best-effort recursive delete. Never raises.
    """
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def s3_put_dir(local_dir: str, s3_dir: str, recursive: bool = True) -> None:
    """
    Upload a local directory to S3.
    """
    if not os.path.exists(local_dir):
        return
    log_event(f"📤 S3 UPLOAD DIR: {local_dir} -> {s3_dir}")
    fs.put(local_dir, s3_dir, recursive=recursive)


def s3_put_file(local_file: str, s3_file: str) -> None:
    """
    Upload a local file to S3.
    """
    if not os.path.exists(local_file):
        return
    log_event(f"📤 S3 UPLOAD FILE: {local_file} -> {s3_file}")
    fs.put(local_file, s3_file)


def s3_get_dir(s3_dir: str, local_dir: str) -> None:
    """
    Download an S3 directory recursively to local storage.
    """
    ensure_dir(local_dir)
    log_event(f"📥 S3 DOWNLOAD DIR: {s3_dir} -> {local_dir}")
    fs.get(s3_dir, local_dir, recursive=True)


def s3_get_file(s3_file: str, local_file: str) -> None:
    """
    Download a single S3 file to local storage.
    """
    ensure_dir(os.path.dirname(local_file) or ".")
    log_event(f"📥 S3 DOWNLOAD FILE: {s3_file} -> {local_file}")
    fs.get(s3_file, local_file)


def s3_write_text_atomic(s3_path: str, text: str) -> None:
    """
    Best-effort atomic text write to S3 using a temporary object first.
    """
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
    """
    Read a text object from S3, or return None if it does not exist.
    """
    if not fs.exists(s3_path):
        return None
    with fs.open(s3_path, "r") as f:
        return f.read()


def s3_exists(s3_path: str) -> bool:
    """
    Check whether an S3 path exists.
    """
    return bool(fs.exists(s3_path))


# =============================================================================
# 3) Tokenizer + sequence length caps
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

# Timestep embedding vocabulary cap.
TIMESTEP_MAX_VOCAB = int(os.environ.get("TIMESTEP_MAX_VOCAB", str(TEMPORAL_MAX_LEN)))

# Auxiliary history-only classification loss weight.
HIST_CLS_WEIGHT = float(os.environ.get("HIST_CLS_WEIGHT", "0.2"))

log_event(f"CUDA available={torch.cuda.is_available()} | device_count={torch.cuda.device_count()}")
log_event(f"LM_CONTEXT_WINDOW={LM_CONTEXT_WINDOW}")
log_event(
    f"MAX_CONTEXT_FOR_BUILDER={MAX_CONTEXT_FOR_BUILDER} | "
    f"MAX_TOKENS_PER_INPUT={MAX_TOKENS_PER_INPUT} | TEMPORAL_MAX_LEN={TEMPORAL_MAX_LEN}"
)
log_event(f"TIMESTEP_MAX_VOCAB={TIMESTEP_MAX_VOCAB} | HIST_CLS_WEIGHT={HIST_CLS_WEIGHT}")
log_event(f"SAVE_FULL_MODEL_EXPORTS={SAVE_FULL_MODEL_EXPORTS}")


# =============================================================================
# 4) Curriculum sources + output paths
# =============================================================================
STAGE_STATE_S3 = s3_join(OUTPUT_S3, "curriculum_state.json")
TOKENIZER_S3_DIR = s3_join(OUTPUT_S3, "tokenizer")
LATEST_MANIFEST_S3 = s3_join(OUTPUT_S3, "LATEST.json")
BUILDER_STATS_S3 = s3_join(OUTPUT_S3, "builder_stats.json")
TRAINING_REPORT_S3 = s3_join(OUTPUT_S3, "training_report.json")

# Canonical final-model output folder.
FINAL_MODEL_S3_DIR = s3_join(OUTPUT_S3, "Final_Model")
FINAL_MODEL_LOCAL_DIR = os.path.join(OUTPUT_BASE_DIR_LOCAL, "Final_Model")

CURRICULUM_STAGE_FILES_S3 = [
    f"{CURRICULUM_PREFIX_S3}_stage1.jsonl",
    f"{CURRICULUM_PREFIX_S3}_stage2.jsonl",
    f"{CURRICULUM_PREFIX_S3}_stage3.jsonl",
]


# =============================================================================
# 5) Training config
# =============================================================================
@dataclass
class TrainConfig:
    """
    Top-level training configuration.
    """
    model_name: str = BASE_MODEL_NAME
    output_base_dir_local: str = OUTPUT_BASE_DIR_LOCAL
    json_sources_s3: List[str] = None

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8

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
    hist_class_weight: float = HIST_CLS_WEIGHT
    mse_weight: float = 0.0
    focal_gamma: float = 1.0


CONFIG = TrainConfig(json_sources_s3=CURRICULUM_STAGE_FILES_S3)


def read_jsonl_s3(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file directly from S3.

    If ALLOW_MISSING_STAGE=1 and the file does not exist, return [].
    """
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
    """
    Infer usable max sequence length from the model config or tokenizer.
    """
    if model is not None and hasattr(model, "config"):
        max_pos = getattr(model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and 0 < max_pos < 10_000_000:
            return max_pos
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and 0 < tmax < 10_000_000:
        return tmax
    return None


# =============================================================================
# 6) Global label namespace
# =============================================================================
LABEL_PAD_ID = 0

TASK_LABELS = {
    "annomi": ["neutral", "sustain", "change"],
    "lrs":    ["0", "1"],
    "mood":   ["0", "IS", "IE"],
}


def build_global_label_map() -> Tuple[Dict[Tuple[str, int], int], Dict[int, Tuple[str, int]]]:
    """
    Build a global label namespace over all tasks.

    Label id 0 is reserved as padding / absent.
    """
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
    """
    Safely return item['meta'] if it is a dict.
    """
    m = item.get("meta", None)
    return m if isinstance(m, dict) else {}


def standardize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw builder examples into a trainer-friendly format.

    Pulls:
      - input / output
      - task_type
      - raw_label_id -> global label_id
      - group_id / timestep for longitudinal training
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
    """
    Downsample examples to balance buckets by (task_type, k_shot_used).

    Disable with BALANCE_OFF=1.
    """
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
# 7) Tokenization + chunking
# =============================================================================
def tokenize_dataset_dynamic(dataset_items, tokenizer, min_stride, max_stride, model=None):
    """
    Convert standardized examples into fixed-length chunked training examples.

    For each example:
      input_ids = prompt_ids + response_ids + [eos]

    Supervision tensors:
      - labels_ce:
          prompt-only next-token CE, excluding <HIST> span
      - labels_focal:
          output-only next-token focal LM loss
      - label_index_seq:
          global label id stamped across output region
      - hist_mask:
          marks <HIST> span
      - timestep_id_seq:
          broadcasts timestep across sequence
    """
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

        # Build history mask over prompt only.
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

        # Output region includes response tokens + EOS.
        output_start = len(prompt_ids)
        output_positions = set(range(output_start, output_start + len(response_ids) + 1))

        labels_ce = [-100] * seq_len
        labels_focal = [-100] * seq_len
        focal_mask = [0] * seq_len

        for t in range(seq_len - 1):
            next_pos = t + 1
            next_tok = input_ids[next_pos]

            # Prompt next-token CE outside the <HIST> region.
            if next_pos < len(prompt_ids) and hist_mask_prompt[next_pos] == 0:
                labels_ce[t] = next_tok

            # Output next-token focal LM.
            if next_pos in output_positions:
                labels_focal[t] = next_tok
                focal_mask[t] = 1

        # Stamp global label across the output region.
        label_id = int(ex["label_id"])
        label_index_seq = [0] * seq_len
        if label_id != 0:
            for pos in output_positions:
                if 0 <= pos < seq_len:
                    label_index_seq[pos] = label_id

        timestep = int(ex.get("timestep", 0))
        timestep_clamped = max(0, min(timestep, TIMESTEP_MAX_VOCAB - 1))
        timestep_id_seq = [timestep_clamped] * seq_len

        # Dynamic overlap stride.
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
                "hist_mask": hmask,
                "timestep_id_seq": ts_seq,
                "target_value": 0.0,
                "task_type": ex["task_type"],
                "group_id": ex.get("group_id", "unknown"),
                "timestep": int(ex.get("timestep", 0)),
                "chunk_start": int(start),
            })

            if end == seq_len:
                break
            start += chunk_max_len - dyn_stride

    return HFDataset.from_list(chunked)


def custom_data_collator(batch):
    """
    Collate already-padded chunk dicts into tensors.
    """
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)

    ids = torch.full((B, maxlen), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((B, maxlen), dtype=torch.long)

    labels_ce = torch.full((B, maxlen), -100, dtype=torch.long)
    labels_focal = torch.full((B, maxlen), -100, dtype=torch.long)
    focal_mask = torch.zeros((B, maxlen), dtype=torch.float)

    lbl_idx = torch.zeros((B, maxlen), dtype=torch.long)
    hist_mask = torch.zeros((B, maxlen), dtype=torch.float)
    timestep_id_seq = torch.zeros((B, maxlen), dtype=torch.long)

    targets = torch.zeros(B, dtype=torch.float)
    types: List[str] = []

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
        hist_mask[i, :l] = torch.tensor(ex.get("hist_mask", [0] * l), dtype=torch.float)
        timestep_id_seq[i, :l] = torch.tensor(ex.get("timestep_id_seq", [0] * l), dtype=torch.long)

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
        "hist_mask": hist_mask,
        "timestep_id_seq": timestep_id_seq,
        "target_value": targets,
        "task_type": types,
        "group_id": group_ids,
        "timestep": timesteps,
        "chunk_start": chunk_starts,
    }


# =============================================================================
# 8) Temporal embedding + wrapper model
# =============================================================================
class RelativeTemporalEmbedding(nn.Module):
    """
    Per-token embedding module that injects:
      1) absolute token position
      2) global label id
      3) relative position-to-end
      4) timestep id

    These four embeddings are concatenated and projected by a small MLP.
    """
    def __init__(self, max_len: int, dim: int, max_label_vocab: int, timestep_vocab: int):
        super().__init__()
        self.temporal_emb = nn.Embedding(max_len, dim)
        self.label_emb = nn.Embedding(max_label_vocab, dim)
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, dim)
        self.timestep_emb = nn.Embedding(timestep_vocab, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.max_len = max_len
        self.timestep_vocab = timestep_vocab

    def forward(self, attention_mask: torch.Tensor, label_indices: torch.Tensor, timestep_id_seq: torch.Tensor) -> torch.Tensor:
        """
        Build temporal features aligned to every token.
        """
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
        ts = self.timestep_emb(timestep_id_seq.clamp(0, self.timestep_vocab - 1))

        h = torch.cat([tmp, lbl, rpe, ts], dim=-1)
        return self.mlp(h)


@dataclass
class LongitudinalOutput(ModelOutput):
    """
    Wrapper output object.
    """
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    class_logits: Optional[torch.FloatTensor] = None
    hist_class_logits: Optional[torch.FloatTensor] = None
    numeric_preds: Optional[torch.FloatTensor] = None


class LongitudinalMultiTaskModel(nn.Module):
    """
    Wrapper around the causal LM.

    Adds:
      - temporal embedding injection
      - pooled classification head
      - history-only pooled classification head
      - regression head
    """
    def __init__(self, lm, max_timestep_emb: int, temporal_dim: int, num_classes: int):
        super().__init__()
        self.lm = lm
        self.hidden = lm.config.hidden_size

        self.temporal = RelativeTemporalEmbedding(
            max_len=max_timestep_emb,
            dim=temporal_dim,
            max_label_vocab=num_classes,
            timestep_vocab=TIMESTEP_MAX_VOCAB,
        )

        lm_dtype = lm.get_input_embeddings().weight.dtype
        self.temp_proj = nn.Linear(temporal_dim, self.hidden).to(dtype=lm_dtype)

        self.cls_head = nn.Linear(self.hidden, num_classes).to(dtype=lm_dtype)
        self.hist_cls_head = nn.Linear(self.hidden, num_classes).to(dtype=lm_dtype)
        self.reg_head = nn.Linear(self.hidden, 1).to(dtype=lm_dtype)

        self.temporal = self.temporal.to(dtype=lm_dtype)

    def forward(self, input_ids, attention_mask, label_index_seq=None, hist_mask=None, timestep_id_seq=None):
        """
        Forward pass:
          1) token embeddings
          2) temporal features
          3) wrapped LM
          4) pooled auxiliary heads
        """
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

        # Main pooled classification head:
        # pool at last label-stamped token if present, else last valid token.
        last_token_positions = (attention_mask.sum(1) - 1).clamp(min=0)
        mask_lbl = (label_index_seq != 0)
        idx = torch.arange(L, device=device).unsqueeze(0).expand_as(label_index_seq)
        idx_masked = idx * mask_lbl.long()
        last_label_positions = idx_masked.max(dim=1).values
        has_label = mask_lbl.any(dim=1)
        pool_positions = torch.where(has_label, last_label_positions, last_token_positions)

        pooled = hidden[torch.arange(B, device=device), pool_positions].to(dtype=self.cls_head.weight.dtype)
        c_logits = self.cls_head(pooled)

        # History-only classification head:
        # mean-pool hidden states inside <HIST>.
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
# 9) Loss
# =============================================================================
def compute_class_balanced_weights(label_ids: List[int], num_classes: int, beta: float = 0.99) -> torch.Tensor:
    """
    Compute class-balanced weights using effective number of samples.

    Class 0 is ignored and gets weight 0.
    """
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
    """
    Focal LM loss computed only on active output positions.
    """
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
    """
    Focal loss for pooled classification logits.
    """
    ce = F.cross_entropy(class_logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss:
      - prompt CE
      - output focal LM
      - pooled classification focal
      - history-only pooled classification focal
      - optional MSE regression
    """
    def __init__(
        self,
        ce_prompt_weight: float,
        focal_output_weight: float,
        class_focal_weight: float,
        hist_class_weight: float,
        mse_weight: float,
        gamma: float,
        class_weights: Optional[torch.Tensor],
        log_breakdown: bool = True,
    ):
        super().__init__()
        self.ce_prompt_weight = ce_prompt_weight
        self.focal_output_weight = focal_output_weight
        self.class_focal_weight = class_focal_weight
        self.hist_class_weight = hist_class_weight
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
        hist_class_logits: Optional[torch.Tensor],
        class_targets: Optional[torch.Tensor],
        numeric_preds: Optional[torch.Tensor],
        numeric_targets: Optional[torch.Tensor],
        class_mask: torch.Tensor,
        hist_present_mask: torch.Tensor,
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
# 10) Sequence-consistent batching
# =============================================================================
class GroupTimeBatchSampler(Sampler[List[int]]):
    """
    Sort chunks by:
      (group_id, timestep, chunk_start)

    This encourages mini-batches to remain temporally coherent within a group.
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
# 11) Custom Trainer
# =============================================================================
def _count_active_tokens(labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> int:
    """
    Count active supervised token positions for logging/debugging.
    """
    active = (labels != -100)
    if mask is not None:
        active = active & (mask > 0.5)
    return int(active.sum().detach().cpu().item())


class LongitudinalTrainer(Trainer):
    """
    Trainer subclass that:
      - uses the custom temporal/group batch sampler
      - computes the multitask objective from the wrapper outputs
    """
    def __init__(self, loss_fn: MultiTaskLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def get_train_dataloader(self):
        """
        Override HF default dataloader so batches preserve group/time ordering.
        """
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
        """
        Compute total multitask loss from:
          - LM logits
          - pooled class logits
          - history-only pooled class logits
        """
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

        has_label = (lbl_idx != 0).any(dim=1).float().to(device)
        class_mask = has_label

        mask_lbl = (lbl_idx != 0)
        idx = torch.arange(T, device=device).unsqueeze(0).expand_as(lbl_idx)
        idx_masked = idx * mask_lbl.long()
        last_label_positions = idx_masked.max(dim=1).values.clamp(min=0)
        class_targets = lbl_idx[torch.arange(B, device=device), last_label_positions].long()

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
# 12) Eval helper
# =============================================================================
def compute_classification_metrics(preds, labels):
    """
    Minimal accuracy + macro-F1 implementation without sklearn.
    """
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
    """
    Evaluate pooled classification head on the validation set.
    """
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
# 13) LoRA schedule + stage-state helpers
# =============================================================================
STAGE_SCHEDULE = {
    1: {"lora_r": 4,  "lr": 2e-4, "scheduler": "cosine"},
    2: {"lora_r": 8,  "lr": 1e-4, "scheduler": "cosine"},
    3: {"lora_r": 16, "lr": 5e-5, "scheduler": "constant"},
}


def stage_cfg(stage_idx: int) -> Dict[str, Any]:
    """
    Return stage-specific schedule.
    """
    return STAGE_SCHEDULE.get(stage_idx, STAGE_SCHEDULE[max(STAGE_SCHEDULE)])


def build_lora_config_for_stage(stage_idx: int) -> LoraConfig:
    """
    Build LoRA config for the given curriculum stage.
    """
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
    """
    Load the global stage progress state from S3.
    """
    txt = s3_read_text(STAGE_STATE_S3)
    if not txt:
        return {"last_completed_stage": 0}
    try:
        return json.loads(txt)
    except Exception:
        return {"last_completed_stage": 0}


def save_stage_state_s3(last_completed_stage: int, last_stage_s3_dir: Optional[str] = None,
                        last_lora_s3_dir: Optional[str] = None, last_heads_s3_path: Optional[str] = None,
                        last_checkpoint_s3_dir: Optional[str] = None) -> None:
    """
    Persist stage completion / resume metadata to S3.

    last_checkpoint_s3_dir is optional metadata describing the newest known
    checkpoint for the last completed stage.
    """
    payload = {"last_completed_stage": int(last_completed_stage)}
    if last_stage_s3_dir:
        payload["last_stage_s3_dir"] = str(last_stage_s3_dir)
    if last_lora_s3_dir:
        payload["last_lora_s3_dir"] = str(last_lora_s3_dir)
    if last_heads_s3_path:
        payload["last_heads_s3_path"] = str(last_heads_s3_path)
    if last_checkpoint_s3_dir:
        payload["last_checkpoint_s3_dir"] = str(last_checkpoint_s3_dir)
    payload["updated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    s3_write_text_atomic(STAGE_STATE_S3, json.dumps(payload, indent=2))


def _load_adapter_state_dict(adapter_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load adapter state dict from safetensors or .bin.
    """
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if _HAS_SAFETENSORS and os.path.exists(st_path):
        return safetensors_load_file(st_path)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def load_lora_with_rank_resize(peft_model, prev_adapter_dir: str) -> None:
    """
    Load previous-stage LoRA weights into current-stage LoRA tensors even if rank changed.

    Overlapping dimensions are copied; any newly added rank dimensions remain trainable.
    """
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
    """
    Freeze the earlier portion of LoRA tensors when stage rank increases.

    Example:
      stage 1 rank = 4
      stage 2 rank = 8

    The first 4 ranks can be frozen so only the newly added portion learns.
    """
    log_event(f"Freezing old LoRA ranks [0:{old_rank}]")
    for name, param in peft_model.named_parameters():
        if "lora_A" in name and param.ndim == 2 and param.shape[0] >= old_rank:
            param.register_hook(lambda grad, old_rank=old_rank: torch.cat([torch.zeros_like(grad[:old_rank]), grad[old_rank:]], dim=0))
        if "lora_B" in name and param.ndim == 2 and param.shape[1] >= old_rank:
            param.register_hook(lambda grad, old_rank=old_rank: torch.cat([torch.zeros_like(grad[:, :old_rank]), grad[:, old_rank:]], dim=1))


# =============================================================================
# 14) Wrapper heads save/load
# =============================================================================
def atomic_torch_save(obj, final_path: str) -> None:
    """
    Atomic torch.save using temp file + rename.
    """
    ensure_dir(os.path.dirname(final_path) or ".")
    tmp_path = final_path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


def get_wrapper_heads_state(model: nn.Module) -> Dict[str, Any]:
    """
    Extract all sidecar wrapper weights plus metadata.
    """
    return {
        "temporal": model.temporal.state_dict(),
        "temp_proj": model.temp_proj.state_dict(),
        "cls_head": model.cls_head.state_dict(),
        "hist_cls_head": model.hist_cls_head.state_dict(),
        "reg_head": model.reg_head.state_dict(),
        "meta": {
            "num_classes": int(getattr(model.cls_head, "out_features", -1)),
            "temporal_dim": int(getattr(model.temp_proj, "in_features", -1)),
            "hidden": int(getattr(model.temp_proj, "out_features", -1)),
            "temporal_max_len": int(getattr(model.temporal, "max_len", -1)),
            "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
        }
    }


def load_wrapper_heads_state(model: nn.Module, path: str) -> bool:
    """
    Best-effort load of sidecar wrapper weights.
    """
    if not os.path.exists(path):
        return False
    try:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            if "temporal" in sd:
                model.temporal.load_state_dict(sd["temporal"], strict=False)
            if "temp_proj" in sd:
                model.temp_proj.load_state_dict(sd["temp_proj"], strict=False)
            if "cls_head" in sd:
                model.cls_head.load_state_dict(sd["cls_head"], strict=False)
            if "hist_cls_head" in sd and hasattr(model, "hist_cls_head"):
                model.hist_cls_head.load_state_dict(sd["hist_cls_head"], strict=False)
            if "reg_head" in sd:
                model.reg_head.load_state_dict(sd["reg_head"], strict=False)
        return True
    except Exception as e:
        log_event(f"[WARN] Failed to load wrapper heads from {path}: {repr(e)}")
        return False


# =============================================================================
# 14.5) Full merged model export helpers
# =============================================================================
def _unwrap_model(m: nn.Module) -> nn.Module:
    """
    Unwrap DataParallel / DDP-style .module containers if present.
    """
    return m.module if hasattr(m, "module") else m


def _infer_export_dtype():
    """
    Pick export dtype:
      - bf16 if CUDA exists
      - else fp32

    Avoids low-bit export paths.
    """
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def export_full_model_from_adapter_local(
    *,
    base_model_name: str,
    adapter_dir_local: str,
    export_dir_local: str,
    tokenizer,
) -> str:
    """
    SAFE merged full-model export path.

    Strategy:
      1) load fresh base model
      2) resize embeddings for tokenizer
      3) attach saved adapter
      4) merge adapter into base model
      5) save merged model + tokenizer

    This avoids mutating the live training model in memory.
    """
    ensure_dir(export_dir_local)
    log_event(f"Preparing FULL merged export from adapter: {adapter_dir_local} -> {export_dir_local}")

    dtype = _infer_export_dtype()

    fresh_base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    fresh_base.resize_token_embeddings(len(tokenizer))

    peft_loaded = PeftModel.from_pretrained(
        fresh_base,
        adapter_dir_local,
        is_trainable=False,
    )

    merged_model = peft_loaded.merge_and_unload()
    merged_model.save_pretrained(export_dir_local)
    tokenizer.save_pretrained(export_dir_local)

    log_event(f"FULL merged export complete: {export_dir_local}")
    return export_dir_local


def export_full_stage_artifacts_to_s3(
    *,
    base_model_name: str,
    adapter_dir_local: str,
    wrapped_model: nn.Module,
    stage_s3_dir: str,
    out_dir_local: str,
    tokenizer,
) -> Dict[str, str]:
    """
    Export END-OF-STAGE merged full-model package.

    Writes locally:
      out_dir_local/full_export/full_model/
      out_dir_local/full_export/wrapper_heads_full.pt
      out_dir_local/full_export/full_inference_manifest.json

    Uploads to S3:
      {stage_s3_dir}/full_model/
      {stage_s3_dir}/wrapper_heads_full.pt
      {stage_s3_dir}/full_inference_manifest.json
    """
    full_export_local = os.path.join(out_dir_local, "full_export")
    safe_rmtree(full_export_local)
    ensure_dir(full_export_local)

    full_model_local = os.path.join(full_export_local, "full_model")
    ensure_dir(full_model_local)

    export_full_model_from_adapter_local(
        base_model_name=base_model_name,
        adapter_dir_local=adapter_dir_local,
        export_dir_local=full_model_local,
        tokenizer=tokenizer,
    )

    heads_full_local = os.path.join(full_export_local, "wrapper_heads_full.pt")
    atomic_torch_save(get_wrapper_heads_state(wrapped_model), heads_full_local)

    manifest = {
        "format": "merged_full_model_plus_wrapper_heads",
        "base_model_baked_in": True,
        "requires_lora_at_inference": False,
        "full_model_s3_dir": s3_join(stage_s3_dir, "full_model"),
        "wrapper_heads_full_s3_path": s3_join(stage_s3_dir, "wrapper_heads_full.pt"),
        "num_classes": int(CONFIG.num_classes),
        "temporal_dim": int(CONFIG.temporal_embed_dim),
        "temporal_max_len": int(CONFIG.max_timestep_emb),
        "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
        "hist_cls_weight": float(CONFIG.hist_class_weight),
        "special_tokens": SPECIAL_TOKENS,
        "global_label_map_size": int(len(GLOBAL_LABEL_INV)),
        "task_labels": TASK_LABELS,
        "base_model_name_original": base_model_name,
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    full_manifest_local = os.path.join(full_export_local, "full_inference_manifest.json")
    with open(full_manifest_local, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    s3_put_dir(full_model_local, s3_join(stage_s3_dir, "full_model"), recursive=True)
    s3_put_file(heads_full_local, s3_join(stage_s3_dir, "wrapper_heads_full.pt"))
    s3_put_file(full_manifest_local, s3_join(stage_s3_dir, "full_inference_manifest.json"))

    return {
        "full_model_s3_dir": s3_join(stage_s3_dir, "full_model"),
        "wrapper_heads_full_s3_path": s3_join(stage_s3_dir, "wrapper_heads_full.pt"),
        "full_manifest_s3_path": s3_join(stage_s3_dir, "full_inference_manifest.json"),
    }


def export_best_full_model_to_s3(
    *,
    base_model_name: str,
    trainer: Trainer,
    stage_s3_dir: str,
    out_dir_local: str,
    tokenizer,
) -> Dict[str, str]:
    """
    Export the current BEST loaded state as a merged standalone model package.

    Strategy:
      1) save current loaded adapter to temp dir
      2) rebuild merged model from fresh base + temp adapter
      3) save wrapper heads from current trainer model
    """
    best_full_local = os.path.join(out_dir_local, "best_full_export")
    safe_rmtree(best_full_local)
    ensure_dir(best_full_local)

    temp_adapter_local = os.path.join(best_full_local, "adapter_tmp")
    ensure_dir(temp_adapter_local)

    wrapped = _unwrap_model(trainer.model)
    wrapped.lm.save_pretrained(temp_adapter_local)

    full_model_local = os.path.join(best_full_local, "full_model_best")
    ensure_dir(full_model_local)

    export_full_model_from_adapter_local(
        base_model_name=base_model_name,
        adapter_dir_local=temp_adapter_local,
        export_dir_local=full_model_local,
        tokenizer=tokenizer,
    )

    heads_best_local = os.path.join(best_full_local, "wrapper_heads_best_full.pt")
    atomic_torch_save(get_wrapper_heads_state(wrapped), heads_best_local)

    manifest = {
        "format": "merged_full_model_plus_wrapper_heads",
        "base_model_baked_in": True,
        "requires_lora_at_inference": False,
        "full_model_s3_dir_best": s3_join(stage_s3_dir, "full_model_best"),
        "wrapper_heads_s3_path_best_full": s3_join(stage_s3_dir, "wrapper_heads_best_full.pt"),
        "num_classes": int(CONFIG.num_classes),
        "temporal_dim": int(CONFIG.temporal_embed_dim),
        "temporal_max_len": int(CONFIG.max_timestep_emb),
        "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
        "hist_cls_weight": float(CONFIG.hist_class_weight),
        "special_tokens": SPECIAL_TOKENS,
        "global_label_map_size": int(len(GLOBAL_LABEL_INV)),
        "task_labels": TASK_LABELS,
        "base_model_name_original": base_model_name,
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    manifest_local = os.path.join(best_full_local, "full_inference_manifest_best.json")
    with open(manifest_local, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    s3_put_dir(full_model_local, s3_join(stage_s3_dir, "full_model_best"), recursive=True)
    s3_put_file(heads_best_local, s3_join(stage_s3_dir, "wrapper_heads_best_full.pt"))
    s3_put_file(manifest_local, s3_join(stage_s3_dir, "full_inference_manifest_best.json"))

    return {
        "full_model_s3_dir_best": s3_join(stage_s3_dir, "full_model_best"),
        "wrapper_heads_s3_path_best_full": s3_join(stage_s3_dir, "wrapper_heads_best_full.pt"),
        "full_manifest_s3_path_best": s3_join(stage_s3_dir, "full_inference_manifest_best.json"),
    }


# =============================================================================
# 14.6) Final continuous model export helpers
# =============================================================================
def export_final_continuous_model(
    *,
    final_stage_s3_dir: str,
    final_lora_local: str,
    final_heads_local: str,
    wrapped_model: nn.Module,
    tokenizer,
) -> Dict[str, Any]:
    """
    Export the final continuously trained model into a canonical Final_Model folder.

    "Continuously trained" means the final stage state after:
      stage 1 -> stage 2 -> stage 3

    Exports:
      - Final_Model/lora_model/
      - Final_Model/wrapper_heads.pt
      - Final_Model/full_model/                  (if SAVE_FULL_MODEL_EXPORTS=1)
      - Final_Model/wrapper_heads_full.pt        (if SAVE_FULL_MODEL_EXPORTS=1)
      - Final_Model/final_inference_manifest.json
      - Final_Model/LATEST_FINAL.json            (written by caller)
    """
    safe_rmtree(FINAL_MODEL_LOCAL_DIR)
    ensure_dir(FINAL_MODEL_LOCAL_DIR)

    # Copy final-stage adapter package into canonical final folder.
    final_lora_out = os.path.join(FINAL_MODEL_LOCAL_DIR, "lora_model")
    shutil.copytree(final_lora_local, final_lora_out)

    final_heads_out = os.path.join(FINAL_MODEL_LOCAL_DIR, "wrapper_heads.pt")
    shutil.copy2(final_heads_local, final_heads_out)

    s3_put_dir(final_lora_out, s3_join(FINAL_MODEL_S3_DIR, "lora_model"), recursive=True)
    s3_put_file(final_heads_out, s3_join(FINAL_MODEL_S3_DIR, "wrapper_heads.pt"))

    manifest = {
        "format": "final_continuous_model",
        "source": "continuous_training_stage1_stage2_stage3",
        "final_stage_s3_dir": final_stage_s3_dir,
        "lora_s3_dir": s3_join(FINAL_MODEL_S3_DIR, "lora_model"),
        "wrapper_heads_s3_path": s3_join(FINAL_MODEL_S3_DIR, "wrapper_heads.pt"),
        "num_classes": int(CONFIG.num_classes),
        "temporal_dim": int(CONFIG.temporal_embed_dim),
        "temporal_max_len": int(CONFIG.max_timestep_emb),
        "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
        "hist_cls_weight": float(CONFIG.hist_class_weight),
        "special_tokens": SPECIAL_TOKENS,
        "task_labels": TASK_LABELS,
        "base_model_name_original": CONFIG.model_name,
        "save_full_model_exports": bool(SAVE_FULL_MODEL_EXPORTS),
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    if SAVE_FULL_MODEL_EXPORTS:
        final_full_local = os.path.join(FINAL_MODEL_LOCAL_DIR, "full_model")
        export_full_model_from_adapter_local(
            base_model_name=CONFIG.model_name,
            adapter_dir_local=final_lora_local,
            export_dir_local=final_full_local,
            tokenizer=tokenizer,
        )

        final_heads_full_local = os.path.join(FINAL_MODEL_LOCAL_DIR, "wrapper_heads_full.pt")
        atomic_torch_save(get_wrapper_heads_state(wrapped_model), final_heads_full_local)

        s3_put_dir(final_full_local, s3_join(FINAL_MODEL_S3_DIR, "full_model"), recursive=True)
        s3_put_file(final_heads_full_local, s3_join(FINAL_MODEL_S3_DIR, "wrapper_heads_full.pt"))

        manifest["full_model_s3_dir"] = s3_join(FINAL_MODEL_S3_DIR, "full_model")
        manifest["wrapper_heads_full_s3_path"] = s3_join(FINAL_MODEL_S3_DIR, "wrapper_heads_full.pt")

    manifest_local = os.path.join(FINAL_MODEL_LOCAL_DIR, "final_inference_manifest.json")
    with open(manifest_local, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    s3_put_file(manifest_local, s3_join(FINAL_MODEL_S3_DIR, "final_inference_manifest.json"))
    return manifest


# =============================================================================
# 15) Checkpoint resume helpers
# =============================================================================
def list_s3_checkpoints(stage_s3_dir: str) -> List[Tuple[int, str]]:
    """
    List checkpoint folders already uploaded to S3 for a stage.

    Returns:
      [(global_step, s3_checkpoint_path), ...] sorted ascending by step
    """
    ckpt_root = s3_join(stage_s3_dir, "checkpoints")
    if not s3_exists(ckpt_root):
        return []

    out = []
    try:
        entries = fs.ls(ckpt_root, detail=False)
    except Exception as e:
        log_event(f"[WARN] Could not list S3 checkpoints at {ckpt_root}: {repr(e)}")
        return []

    for p in entries:
        name = os.path.basename(str(p).rstrip("/"))
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                out.append((step, p))
            except Exception:
                continue

    out.sort(key=lambda x: x[0])
    return out


def get_latest_s3_checkpoint(stage_s3_dir: str) -> Optional[Tuple[int, str]]:
    """
    Return the latest checkpoint on S3 for this stage, if any.
    """
    ckpts = list_s3_checkpoints(stage_s3_dir)
    if not ckpts:
        return None
    return ckpts[-1]


def download_latest_stage_checkpoint_if_any(stage_s3_dir: str, out_dir_local: str) -> Optional[str]:
    """
    Download the newest checkpoint for this stage from S3 into the local output dir.

    Returns:
      local checkpoint path, or None if no checkpoint exists.
    """
    latest = get_latest_s3_checkpoint(stage_s3_dir)
    if latest is None:
        return None

    step, s3_ckpt_path = latest
    local_ckpt_path = os.path.join(out_dir_local, f"checkpoint-{step}")

    # If it already exists locally, reuse it.
    if os.path.isdir(local_ckpt_path):
        log_event(f"Latest local checkpoint already present: {local_ckpt_path}")
        return local_ckpt_path

    # Remove stale local checkpoint dirs to reduce disk usage.
    try:
        for nm in os.listdir(out_dir_local):
            if nm.startswith("checkpoint-"):
                safe_rmtree(os.path.join(out_dir_local, nm))
    except Exception:
        pass

    log_event(f"Downloading latest stage checkpoint from S3: {s3_ckpt_path} -> {local_ckpt_path}")
    s3_get_dir(s3_ckpt_path, local_ckpt_path)
    return local_ckpt_path


def should_resume_incomplete_stage(stage_idx: int, last_completed_stage: int) -> bool:
    """
    True if this stage is the first not-yet-completed stage and may therefore
    have a partially completed checkpoint to resume.
    """
    return stage_idx == (last_completed_stage + 1)


def safe_train_with_resume(trainer: Trainer, resume_ckpt_local: Optional[str] = None):
    """
    Resume from checkpoint if available.

    If checkpoint restore fails for any reason, log a warning and fall back
    to a fresh stage start rather than crashing immediately.
    """
    if resume_ckpt_local is None:
        trainer.train()
        return

    try:
        log_event(f"Resuming trainer from checkpoint: {resume_ckpt_local}")
        trainer.train(resume_from_checkpoint=resume_ckpt_local)
    except Exception as e:
        log_event(f"[WARN] Resume from checkpoint failed: {repr(e)}")
        log_event("[WARN] Falling back to fresh start for this stage.")
        trainer.train()


# =============================================================================
# 16) S3 checkpoint callback
# =============================================================================
class S3CheckpointCallback(TrainerCallback):
    """
    Upload Trainer checkpoint directories to S3 on each save.

    Important:
      - These checkpoints now include FULL trainer state because
        TrainingArguments.save_only_model=False
      - That makes mid-stage resume reliable.

    After upload, local checkpoint directories are pruned.
    """
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

            # Optional pointer for easier external inspection.
            latest_payload = {
                "checkpoint_name": ckpt_name,
                "checkpoint_s3_dir": s3_ckpt,
                "global_step": int(state.global_step),
                "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
            }
            s3_write_text_atomic(
                s3_join(self.stage_s3_dir, "checkpoints", "LATEST_CHECKPOINT.json"),
                json.dumps(latest_payload, indent=2),
            )

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
# 17) Tokenizer upload + latest manifest
# =============================================================================
def upload_tokenizer_to_s3_once() -> None:
    """
    Upload tokenizer to S3 only once.
    """
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
    """
    Update the root LATEST.json pointer to the newest stage manifest.
    """
    payload = dict(info)
    payload["updated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    s3_write_text_atomic(LATEST_MANIFEST_S3, json.dumps(payload, indent=2))
    log_event(f"Wrote latest manifest: {LATEST_MANIFEST_S3}")


# =============================================================================
# 18) Builder sanity check
# =============================================================================
def sanity_check_builder_signature() -> None:
    """
    Validate that trainer-side assumptions match builder outputs.

    Checks:
      - curriculum prefix consistency
      - existence of expected stage files
    """
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
# 19) Training report helpers
# =============================================================================
def _load_training_report() -> Dict[str, Any]:
    """
    Load cumulative training report from S3.
    """
    txt = s3_read_text(TRAINING_REPORT_S3)
    if not txt:
        return {"runs": [], "latest": None}
    try:
        return json.loads(txt)
    except Exception:
        return {"runs": [], "latest": None}


def _save_training_report(report: Dict[str, Any]) -> None:
    """
    Save cumulative training report to S3.
    """
    s3_write_text_atomic(TRAINING_REPORT_S3, json.dumps(report, indent=2))


def _update_training_report(stage_entry: Dict[str, Any]) -> None:
    """
    Insert or replace a stage entry in the global training report.
    """
    report = _load_training_report()
    runs = report.get("runs", [])
    runs = [
        r for r in runs
        if not (r.get("stage_idx") == stage_entry.get("stage_idx") and r.get("stage_s3_dir") == stage_entry.get("stage_s3_dir"))
    ]
    runs.append(stage_entry)
    runs.sort(key=lambda x: (int(x.get("stage_idx", 0)), str(x.get("stage_s3_dir", ""))))
    report["runs"] = runs
    report["latest"] = stage_entry
    report["updated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    _save_training_report(report)


# =============================================================================
# 20) Curriculum summary
# =============================================================================
def _write_curriculum_summary(
    stage_s3_dir: str,
    builder_items: List[Dict[str, Any]],
    balanced_items: List[Dict[str, Any]],
    chunks: HFDataset,
) -> None:
    """
    Save a compact summary of raw data, balanced data, and chunk stats.
    """
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
        "exports": {
            "save_full_model_exports": bool(SAVE_FULL_MODEL_EXPORTS),
        },
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    s3_write_text_atomic(s3_join(stage_s3_dir, "curriculum_summary.json"), json.dumps(summary, indent=2))


# =============================================================================
# 21) Best pointer + log history + best exports
# =============================================================================
def _save_log_history_to_s3(trainer: Trainer, stage_s3_dir: str, out_dir_local: str) -> None:
    """
    Save Trainer log_history to S3 for debugging / later inspection.
    """
    hist = getattr(trainer.state, "log_history", None)
    if not hist:
        return
    local_path = os.path.join(out_dir_local, "log_history.json")
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    s3_put_file(local_path, s3_join(stage_s3_dir, "log_history.json"))


def _write_best_pointer(stage_s3_dir: str, trainer: Trainer) -> Dict[str, Any]:
    """
    Write BEST.json containing the trainer's best checkpoint metadata.
    """
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
    """
    Export the currently loaded BEST state into:
      best_export/lora_model/
      best_export/wrapper_heads.pt
    """
    best_export_local = os.path.join(out_dir_local, "best_export")
    safe_rmtree(best_export_local)
    ensure_dir(best_export_local)

    lora_local = os.path.join(best_export_local, "lora_model")
    ensure_dir(lora_local)

    wrapped = _unwrap_model(trainer.model)
    wrapped.lm.save_pretrained(lora_local)
    s3_put_dir(lora_local, s3_join(stage_s3_dir, "best_export", "lora_model"), recursive=True)

    heads_local = os.path.join(best_export_local, "wrapper_heads.pt")
    atomic_torch_save(get_wrapper_heads_state(wrapped), heads_local)
    s3_put_file(heads_local, s3_join(stage_s3_dir, "best_export", "wrapper_heads.pt"))


def _export_best_weights_canonical_to_s3(trainer: Trainer, stage_s3_dir: str, out_dir_local: str) -> None:
    """
    Export the currently loaded BEST state into canonical paths:
      - lora_model_best/
      - wrapper_heads_best.pt
    """
    best_canon_local = os.path.join(out_dir_local, "best_canonical")
    safe_rmtree(best_canon_local)
    ensure_dir(best_canon_local)

    wrapped = _unwrap_model(trainer.model)

    lora_best_local = os.path.join(best_canon_local, "lora_model_best")
    ensure_dir(lora_best_local)
    wrapped.lm.save_pretrained(lora_best_local)
    s3_put_dir(lora_best_local, s3_join(stage_s3_dir, "lora_model_best"), recursive=True)

    heads_best_local = os.path.join(best_canon_local, "wrapper_heads_best.pt")
    atomic_torch_save(get_wrapper_heads_state(wrapped), heads_best_local)
    s3_put_file(heads_best_local, s3_join(stage_s3_dir, "wrapper_heads_best.pt"))


# =============================================================================
# 22) Resume safety helpers
# =============================================================================
def _assert_no_bitsandbytes(model) -> None:
    """
    Forbid bitsandbytes / low-bit modules in this pipeline.

    This helps keep checkpoint/export behavior predictable.
    """
    for _, m in model.named_modules():
        mod_name = m.__class__.__name__.lower()
        mod_pkg = m.__class__.__module__.lower()
        if "bitsandbytes" in mod_pkg or "8bit" in mod_name or "4bit" in mod_name:
            raise RuntimeError("Detected bitsandbytes module. This pipeline forbids bitsandbytes.")


def rehydrate_previous_artifacts_from_state(state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Re-download previous completed stage's adapter + wrapper heads from S3.

    Used when continuing to the next stage after a process/container restart.
    """
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


# =============================================================================
# 23) Main train loop
# =============================================================================
def train_curriculum():
    """
    Main curriculum loop across stage 1, stage 2, and stage 3.

    Restart behavior:
      - completed stages are skipped via curriculum_state.json
      - the first incomplete stage attempts checkpoint resume from S3
    """
    sanity_check_builder_signature()
    upload_tokenizer_to_s3_once()

    state = load_stage_state_s3()
    last_done = int(state.get("last_completed_stage", 0))
    prev_lora_local, prev_heads_local = rehydrate_previous_artifacts_from_state(state)

    # Final continuously trained model trackers.
    final_stage_s3_dir = None
    final_lora_local = None
    final_heads_local = None
    final_wrapped_model = None

    for idx, s3_jfile in enumerate(CONFIG.json_sources_s3):
        stage_idx = idx + 1
        stage_tag = os.path.basename(s3_jfile).replace(".jsonl", "").replace(".json", "")
        stage_s3_dir = s3_join(OUTPUT_S3, f"stage_{stage_idx}_{stage_tag}")
        out_dir_local = os.path.join(CONFIG.output_base_dir_local, f"stage_{stage_idx}_{stage_tag}")
        ensure_dir(out_dir_local)

        # If this is the first incomplete stage, it may have a checkpoint to resume.
        resume_ckpt_local = None
        if should_resume_incomplete_stage(stage_idx, last_done):
            resume_ckpt_local = download_latest_stage_checkpoint_if_any(
                stage_s3_dir=stage_s3_dir,
                out_dir_local=out_dir_local,
            )
            if resume_ckpt_local:
                log_event(f"Will resume stage {stage_idx} from checkpoint: {resume_ckpt_local}")

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

        # ---------------------------------------------------------------------
        # 1) Balance and standardize raw builder items.
        # ---------------------------------------------------------------------
        rng = random.Random(10_000 + stage_idx)
        balanced = balance_dataset(builder_items, rng=rng, kshot_bins=list(range(0, 15)))
        raw_data = [standardize_item(x) for x in balanced]

        label_ids_for_weights = [ex["label_id"] for ex in raw_data if ex["label_id"] != 0]
        class_weights = compute_class_balanced_weights(label_ids_for_weights, CONFIG.num_classes) if label_ids_for_weights else None

        # ---------------------------------------------------------------------
        # 2) Load base causal LM and attach stage-specific LoRA.
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 3) If available, initialize from previous completed stage.
        #
        # Important:
        #   Even if we later resume from a mid-stage checkpoint, this initialization
        #   is still okay because Trainer resume will overwrite weights/states with
        #   the checkpoint contents.
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 4) Tokenize and chunk examples.
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 5) Split into train / validation if possible.
        # ---------------------------------------------------------------------
        if len(chunks) > 1 and CONFIG.val_fraction > 0:
            split = chunks.train_test_split(CONFIG.val_fraction, seed=42)
            train_ds, eval_ds = split["train"], split["test"]
            log_event(f"Train={len(train_ds)}, Eval={len(eval_ds)}")
        else:
            train_ds, eval_ds = chunks, None
            log_event("No validation split.")

        # ---------------------------------------------------------------------
        # 6) Build multitask loss.
        # ---------------------------------------------------------------------
        mt_loss = MultiTaskLoss(
            ce_prompt_weight=CONFIG.ce_prompt_weight,
            focal_output_weight=CONFIG.focal_output_weight,
            class_focal_weight=CONFIG.class_focal_weight,
            hist_class_weight=CONFIG.hist_class_weight,
            mse_weight=CONFIG.mse_weight,
            gamma=CONFIG.focal_gamma,
            class_weights=class_weights,
            log_breakdown=True,
        )

        # ---------------------------------------------------------------------
        # 7) Trainer arguments.
        #
        # Critical for resume:
        #   save_only_model MUST be False so optimizer/scheduler/trainer state
        #   are included inside checkpoints.
        # ---------------------------------------------------------------------
        save_steps = int(os.environ.get("SAVE_STEPS", "5"))
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

            # IMPORTANT:
            # Must be False to allow full-state checkpoint resume.
            save_only_model=False,
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

        # ---------------------------------------------------------------------
        # 8) Train, resuming from the latest checkpoint if available.
        # ---------------------------------------------------------------------
        log_event(f"Training stage {stage_idx}... (LOSS_LOG_EVERY={os.environ.get('LOSS_LOG_EVERY', '5')})")
        safe_train_with_resume(trainer, resume_ckpt_local)

        # Save log history.
        _save_log_history_to_s3(trainer, stage_s3_dir=stage_s3_dir, out_dir_local=out_dir_local)

        # Best checkpoint metadata.
        best_payload = _write_best_pointer(stage_s3_dir, trainer)

        # ---------------------------------------------------------------------
        # 9) Export best weights if validation is enabled.
        # ---------------------------------------------------------------------
        if eval_ds is not None and args.load_best_model_at_end:
            log_event("Exporting BEST weights (LoRA + canonical + optional full merged model) ...")
            _export_best_weights_to_s3(
                trainer,
                stage_s3_dir=stage_s3_dir,
                out_dir_local=out_dir_local,
            )
            _export_best_weights_canonical_to_s3(
                trainer,
                stage_s3_dir=stage_s3_dir,
                out_dir_local=out_dir_local,
            )

            if SAVE_FULL_MODEL_EXPORTS:
                export_best_full_model_to_s3(
                    base_model_name=CONFIG.model_name,
                    trainer=trainer,
                    stage_s3_dir=stage_s3_dir,
                    out_dir_local=out_dir_local,
                    tokenizer=tokenizer,
                )

        # ---------------------------------------------------------------------
        # 10) Final validation loss.
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 11) Validation metrics for classification.
        # ---------------------------------------------------------------------
        metrics = {}
        if eval_ds is not None:
            log_event("Evaluating classification metrics...")
            metrics = evaluate_stage_model(model, eval_ds)
            local_metrics = os.path.join(out_dir_local, "eval_metrics.json")
            with open(local_metrics, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            s3_put_file(local_metrics, s3_join(stage_s3_dir, "eval_metrics.json"))

        # ---------------------------------------------------------------------
        # 12) Save END-OF-STAGE LoRA + wrapper heads.
        # ---------------------------------------------------------------------
        wrapped_model = _unwrap_model(trainer.model)

        lora_local = os.path.join(out_dir_local, "lora_model")
        ensure_dir(lora_local)
        wrapped_model.lm.save_pretrained(lora_local)
        s3_put_dir(lora_local, s3_join(stage_s3_dir, "lora_model"), recursive=True)

        heads_local = os.path.join(out_dir_local, "wrapper_heads.pt")
        atomic_torch_save(get_wrapper_heads_state(wrapped_model), heads_local)
        s3_put_file(heads_local, s3_join(stage_s3_dir, "wrapper_heads.pt"))

        # ---------------------------------------------------------------------
        # 13) Save END-OF-STAGE full merged model package.
        # ---------------------------------------------------------------------
        if SAVE_FULL_MODEL_EXPORTS:
            export_full_stage_artifacts_to_s3(
                base_model_name=CONFIG.model_name,
                adapter_dir_local=lora_local,
                wrapped_model=wrapped_model,
                stage_s3_dir=stage_s3_dir,
                out_dir_local=out_dir_local,
                tokenizer=tokenizer,
            )

        logs_local = os.path.join(out_dir_local, "logs")
        if os.path.isdir(logs_local):
            s3_put_dir(logs_local, s3_join(stage_s3_dir, "logs"), recursive=True)

        # ---------------------------------------------------------------------
        # 14) Write stage inference manifest.
        # ---------------------------------------------------------------------
        manifest = {
            "base_model_name": CONFIG.model_name,
            "tokenizer_s3_dir": TOKENIZER_S3_DIR,
            "stage_s3_dir": stage_s3_dir,

            # End-of-stage LoRA package
            "lora_s3_dir": s3_join(stage_s3_dir, "lora_model"),
            "wrapper_heads_s3_path": s3_join(stage_s3_dir, "wrapper_heads.pt"),

            # End-of-stage full merged package
            "full_model_s3_dir": s3_join(stage_s3_dir, "full_model"),
            "wrapper_heads_full_s3_path": s3_join(stage_s3_dir, "wrapper_heads_full.pt"),

            "best_pointer_s3": s3_join(stage_s3_dir, "BEST.json"),
            "best_model_checkpoint_local": best_payload.get("best_model_checkpoint"),
            "best_metric": best_payload.get("best_metric"),
            "best_global_step": best_payload.get("best_global_step"),

            # Best LoRA package
            "best_export_s3_dir": s3_join(stage_s3_dir, "best_export"),
            "lora_s3_dir_best": s3_join(stage_s3_dir, "lora_model_best"),
            "wrapper_heads_s3_path_best": s3_join(stage_s3_dir, "wrapper_heads_best.pt"),

            # Best full merged package
            "full_model_s3_dir_best": s3_join(stage_s3_dir, "full_model_best"),
            "wrapper_heads_s3_path_best_full": s3_join(stage_s3_dir, "wrapper_heads_best_full.pt"),

            "curriculum_summary_s3": s3_join(stage_s3_dir, "curriculum_summary.json"),

            "num_classes": int(CONFIG.num_classes),
            "temporal_dim": int(CONFIG.temporal_embed_dim),
            "temporal_max_len": int(CONFIG.max_timestep_emb),
            "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
            "hist_cls_weight": float(CONFIG.hist_class_weight),
            "sequence_consistent_batching": True,
            "save_full_model_exports": bool(SAVE_FULL_MODEL_EXPORTS),

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
            "latest_checkpoint_pointer_s3": s3_join(stage_s3_dir, "checkpoints", "LATEST_CHECKPOINT.json"),
        }

        s3_write_text_atomic(s3_join(stage_s3_dir, "inference_manifest.json"), json.dumps(manifest, indent=2))
        write_latest_manifest(stage_s3_dir=stage_s3_dir, info=manifest)

        # ---------------------------------------------------------------------
        # 15) Update cumulative training report.
        # ---------------------------------------------------------------------
        stage_entry = {
            "stage_idx": int(stage_idx),
            "stage_file_s3": s3_jfile,
            "stage_s3_dir": stage_s3_dir,
            "curriculum_prefix_s3": CURRICULUM_PREFIX_S3,
            "best_pointer_s3": s3_join(stage_s3_dir, "BEST.json"),
            "best_model_checkpoint_local": best_payload.get("best_model_checkpoint"),
            "best_metric": best_payload.get("best_metric"),
            "best_global_step": best_payload.get("best_global_step"),

            # Best LoRA
            "lora_s3_dir_best": s3_join(stage_s3_dir, "lora_model_best"),
            "wrapper_heads_s3_path_best": s3_join(stage_s3_dir, "wrapper_heads_best.pt"),

            # Best full merged
            "full_model_s3_dir_best": s3_join(stage_s3_dir, "full_model_best"),
            "wrapper_heads_s3_path_best_full": s3_join(stage_s3_dir, "wrapper_heads_best_full.pt"),

            # End LoRA
            "lora_s3_dir_end": s3_join(stage_s3_dir, "lora_model"),
            "wrapper_heads_s3_path_end": s3_join(stage_s3_dir, "wrapper_heads.pt"),

            # End full merged
            "full_model_s3_dir_end": s3_join(stage_s3_dir, "full_model"),
            "wrapper_heads_s3_path_end_full": s3_join(stage_s3_dir, "wrapper_heads_full.pt"),

            "eval_loss": eval_loss_payload.get("eval_loss", None),
            "metrics": metrics,
            "curriculum_summary_s3": s3_join(stage_s3_dir, "curriculum_summary.json"),
            "longitudinal": {
                "timestep_max_vocab": int(TIMESTEP_MAX_VOCAB),
                "hist_cls_weight": float(CONFIG.hist_class_weight),
                "sequence_consistent_batching": True,
            },
            "exports": {
                "save_full_model_exports": bool(SAVE_FULL_MODEL_EXPORTS),
            },
            "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        _update_training_report(stage_entry)

        # ---------------------------------------------------------------------
        # 16) Carry forward end-of-stage outputs for next stage initialization.
        # ---------------------------------------------------------------------
        prev_lora_local = lora_local
        prev_heads_local = heads_local

        # Track final continuously trained state.
        final_stage_s3_dir = stage_s3_dir
        final_lora_local = lora_local
        final_heads_local = heads_local
        final_wrapped_model = wrapped_model

        # Discover latest checkpoint path for metadata.
        latest_ckpt = get_latest_s3_checkpoint(stage_s3_dir)
        latest_ckpt_s3 = latest_ckpt[1] if latest_ckpt is not None else None

        save_stage_state_s3(
            last_completed_stage=stage_idx,
            last_stage_s3_dir=stage_s3_dir,
            last_lora_s3_dir=s3_join(stage_s3_dir, "lora_model"),
            last_heads_s3_path=s3_join(stage_s3_dir, "wrapper_heads.pt"),
            last_checkpoint_s3_dir=latest_ckpt_s3,
        )
        log_event(f"Completed stage {stage_idx}; updated S3 state: {STAGE_STATE_S3}")

        # ---------------------------------------------------------------------
        # 17) Clean local workspace but keep end-of-stage adapter + heads.
        # ---------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 18) Export final continuous model after all curriculum stages finish.
    # -------------------------------------------------------------------------
    if final_stage_s3_dir and final_lora_local and final_heads_local and final_wrapped_model is not None:
        log_event("Exporting FINAL continuous model (stage1 -> stage2 -> stage3) to Final_Model ...")
        final_manifest = export_final_continuous_model(
            final_stage_s3_dir=final_stage_s3_dir,
            final_lora_local=final_lora_local,
            final_heads_local=final_heads_local,
            wrapped_model=final_wrapped_model,
            tokenizer=tokenizer,
        )
        s3_write_text_atomic(
            s3_join(FINAL_MODEL_S3_DIR, "LATEST_FINAL.json"),
            json.dumps(final_manifest, indent=2),
        )
        log_event(f"FINAL continuous model exported to: {FINAL_MODEL_S3_DIR}")

    log_event("Curriculum training complete.")


# =============================================================================
# 24) Main
# =============================================================================
def main():
    """
    Entrypoint.
    """
    log_event("START: iFT TRAINER ONLY (S3-first) — LONGITUDINAL UPDATED + FULL EXPORTS + FINAL CONTINUOUS EXPORT + INTERRUPTION-SAFE RESUME")
    log_event(f"CURRICULUM_PREFIX_S3={CURRICULUM_PREFIX_S3}")
    log_event(f"Stage files: {CONFIG.json_sources_s3}")
    train_curriculum()
    log_event(f"Wrote training report: {TRAINING_REPORT_S3}")
    log_event("DONE.")


if __name__ == "__main__":
    main()
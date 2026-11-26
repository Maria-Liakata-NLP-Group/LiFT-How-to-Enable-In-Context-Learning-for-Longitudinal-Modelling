
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
IFT BUILDER (S3-first) — Curriculum JSONL Generator (BUILD ONLY)
================================================================================

This script BUILDS the curriculum that the trainer will later consume.
It DOES NOT train any model. Its only job is to:

  1) Load 3 datasets (ANNoMI, LRS, TalkLife/MOOD) from local paths or S3.
  2) Convert each row into a longitudinal prompt with optional history (<HIST>).
  3) Add true in-group longitudinal few-shot demonstrations (not fake k flags).
  4) Enforce strict token budgets by trimming history and dropping demos if needed.
  5) Write exactly THREE JSONL stage files to S3 with a FIXED mapping:
        Stage 1 -> ANNoMI (target k=1)
        Stage 2 -> LRS    (target k=2)
        Stage 3 -> MOOD   (target k=3)
     No Stage 4 output.
     No Reddit TIMELINE dataset.

--------------------------------------------------------------------------------
OUTPUTS (WRITTEN TO S3)
--------------------------------------------------------------------------------
The builder writes to:
  CURRICULUM_PREFIX_S3 = env CURRICULUM_PREFIX_S3
                         default: {IFT_NEWIFTDATA_S3}/ICL_curriculum_IFT_llama8bnew

And produces EXACTLY:
  {CURRICULUM_PREFIX_S3}_stage1.jsonl   # ONLY task_type="annomi"
  {CURRICULUM_PREFIX_S3}_stage2.jsonl   # ONLY task_type="lrs"
  {CURRICULUM_PREFIX_S3}_stage3.jsonl   # ONLY task_type="mood"

It also writes build statistics to:
  {IFT_OUTPUT_S3}/builder_stats.json

--------------------------------------------------------------------------------
WHAT EACH JSONL LINE LOOKS LIKE (TRAINING EXAMPLE FORMAT)
--------------------------------------------------------------------------------
Each JSONL line is one training example with these important keys:

{
  "input":  "<instruction> ... </instruction>\n"
            "<few-shot> ... </few-shot>\n"
            "<query> ... </query>\n"
            "<output>\n",
  "output": "gold_label_text",     # the correct label string the model must generate
  "task_type": "annomi|lrs|mood",
  "raw_label_id": int,             # label id within that task (task-local)
  "k_shot_used": int,              # number of true demos included (0..k)
  "stage": int,                    # 1, 2, or 3
  "meta": {...},                   # original row metadata (for auditing)
  "example_id": "16-char-hash"     # stable ID for dedup + traceability
}

The "input" always ends with "<output>\n" so the trainer learns to generate the
label text into that region.

--------------------------------------------------------------------------------
LONGITUDINAL HISTORY BEHAVIOR (<HIST>)
--------------------------------------------------------------------------------
For each task, examples are created in temporal order within a "group":

- ANNoMI groups by:  topic
  History lines look like:  "client: ..." or "therapist: ..."

- MOOD groups by:    user_id
  History lines look like:  "[datetime] post text"

- LRS groups by:     timeline_id
  History lines look like:  "[datetime] post text"

Each training example includes:
  header + optional history block + current post/utterance + label space

History is wrapped in:
  <HIST> ... </HIST>

IMPORTANT: History is TRIMMED from the OLDEST side until the total token count
fits within MAX_TOKENS_PER_INPUT. This ensures examples stay within the model’s
context window.

--------------------------------------------------------------------------------
FEW-SHOT DEMONSTRATIONS (TRUE IN-GROUP, LONGITUDINAL)
--------------------------------------------------------------------------------
For each query item, builder selects demonstrations ONLY from the SAME group
(topic / user_id / timeline_id) and only from PREVIOUS time steps.

- Demos DO NOT include history.
- Each demo is formatted as:
    ### Demonstration Example N
    Input:
      Current post ... + Label space ...
    Output: GOLD_LABEL

Target k per stage:
  Stage 1: k=1
  Stage 2: k=2
  Stage 3: k=3

If (prompt + demos + output) exceeds MAX_CONTEXT_FOR_BUILDER:
  - drop demos one-by-one until it fits
  - if still too long: drop the example entirely (counted in stats)

"k_shot_used" reports the TRUE number of demos included after fitting.

--------------------------------------------------------------------------------
TOKEN BUDGETS (IMPORTANT KNOBS)
--------------------------------------------------------------------------------
The builder uses a tokenizer of BASE_MODEL_NAME to estimate token counts.

- LM_CONTEXT_WINDOW comes from model config max_position_embeddings
- BUILDER_CAP (env BUILDER_CAP) caps maximum size builder will attempt
- MAX_TOKENS_PER_INPUT (env MAX_TOKENS_PER_INPUT) caps history trimming size

Effective cap used:
  MAX_CONTEXT_FOR_BUILDER = min(LM_CONTEXT_WINDOW - 32, BUILDER_CAP)
  MAX_TOKENS_PER_INPUT    = min(MAX_TOKENS_PER_INPUT_ENV, MAX_CONTEXT_FOR_BUILDER)

--------------------------------------------------------------------------------
DEDUPE / STAGING
--------------------------------------------------------------------------------
- Each example gets a stable example_id = sha256(task + meta + input_repr prefix)
- Example IDs are tracked across all stages to prevent duplicates.
- If STRICT_DEDUP=1, builder hard-fails if any duplicate example_id appears
  across stage files.

Staging is FIXED by task_type (not round-robin):
  annomi -> stage1
  lrs    -> stage2
  mood   -> stage3

--------------------------------------------------------------------------------
REQUIRED / OPTIONAL ENV VARS
--------------------------------------------------------------------------------
Required:
  IFT_OUTPUT_S3
  IFT_NEWIFTDATA_S3
  FSSPEC_S3_ENDPOINT_URL
  FSSPEC_S3_KEY
  FSSPEC_S3_SECRET

Optional:
  CURRICULUM_PREFIX_S3
  IFT_DATASET_S3              (if you want default dataset files from S3)
  PATH_ANNOMI / PATH_LRS / PATH_MOOD (explicit override paths)
  BASE_MODEL_NAME
  BUILDER_CAP
  MAX_TOKENS_PER_INPUT
  STRICT_DEDUP
  HF_TOKEN (auto forwarded to HUGGINGFACE_HUB_TOKEN)

--------------------------------------------------------------------------------
WHAT THIS BUILDER ENABLES FOR THE TRAINER
--------------------------------------------------------------------------------
The downstream trainer can now assume:
  - Stage 1 contains ONLY ANNoMI with ~1-shot in-group demos
  - Stage 2 contains ONLY LRS with ~2-shot in-group demos
  - Stage 3 contains ONLY MOOD with ~3-shot in-group demos
  - No stage 4 exists
  - No reddit timeline dataset exists
  - Adds explicit timestep markers and history-length markers into the prompt header.
- Prefixes each history line with explicit relative time index: t-3, t-2, t-1.
- Stores meta fields:
    group_id, timestep, group_size, history_len_used
  so trainer can do sequence-consistent batching and timestep embedding.
This makes curriculum scheduling deterministic and avoids task mixing.

================================================================================
"""

import os
import json
import random
import datetime
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from ift_s3_utils import (
    make_s3_fs,
    log_event,
    s3_join,
    s3_write_text_atomic,
)

# =============================================================================
# Repro
# =============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# =============================================================================
# ENV
# =============================================================================
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

OUTPUT_S3 = os.environ["IFT_OUTPUT_S3"].rstrip("/")
CURRICULUM_S3 = os.environ["IFT_NEWIFTDATA_S3"].rstrip("/")

CURRICULUM_PREFIX_S3 = os.environ.get(
    "CURRICULUM_PREFIX_S3",
    s3_join(CURRICULUM_S3, "ICL_curriculum_IFT_llama8b"),
).rstrip("/")

STRICT_DEDUP = os.environ.get("STRICT_DEDUP", "0") == "1"
BUILDER_CAP = int(os.environ.get("BUILDER_CAP", "4096"))
MAX_TOKENS_PER_INPUT_ENV = int(os.environ.get("MAX_TOKENS_PER_INPUT", "4096"))

if "HF_TOKEN" in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

# =============================================================================
# S3 FS
# =============================================================================
fs = make_s3_fs()

# =============================================================================
# Tokenizer
# =============================================================================
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

MAX_CONTEXT_FOR_BUILDER = min(LM_CONTEXT_WINDOW - 32, BUILDER_CAP)
MAX_TOKENS_PER_INPUT = min(MAX_TOKENS_PER_INPUT_ENV, MAX_CONTEXT_FOR_BUILDER)

# =============================================================================
# Dataset paths (local OR S3)
# =============================================================================
IFT_DATASET_S3 = os.environ.get("IFT_DATASET_S3", "").rstrip("/")

def resolve_dataset_path(local_env_name: str, default_name: str) -> str:
    lp = os.environ.get(local_env_name, "")
    if lp:
        return lp
    if IFT_DATASET_S3:
        return s3_join(IFT_DATASET_S3, default_name)
    return ""


PATH_ANNOMI = resolve_dataset_path("PATH_ANNOMI", "annomi_train.csv")
PATH_MOOD   = resolve_dataset_path("PATH_MOOD", "Talklife_filtered_IS_IE_O_train.csv")
PATH_LRS    = resolve_dataset_path("PATH_LRS", "lrs17_train_train.csv")

def file_exists_any(path: str) -> bool:
    if not path:
        return False
    if path.startswith("s3://"):
        return bool(fs.exists(path))
    return os.path.exists(path)

def read_csv_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        with fs.open(path, "rb") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)

def assert_inputs_exist():
    required = {
        "ANNoMI": PATH_ANNOMI,
        "TALKLIFE(MOOD)": PATH_MOOD,
        "LRS": PATH_LRS,
    }
    log_event("Builder: checking dataset paths (hard-fail if missing):")
    missing = []
    for name, p in required.items():
        ok = file_exists_any(p)
        log_event(f"  - {name}: {'FOUND' if ok else 'MISSING'} -> {p}")
        if not ok:
            missing.append((name, p))
    if missing:
        lines = "\n".join([f"{n}: {p}" for n, p in missing])
        raise RuntimeError(f"Missing dataset files:\n{lines}")

def stable_example_id(task_type: str, meta: Any, input_repr: str) -> str:
    h = hashlib.sha256()
    h.update(task_type.encode("utf-8"))
    try:
        h.update(json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    except Exception:
        h.update(str(meta).encode("utf-8"))
    h.update(input_repr[:2000].encode("utf-8"))
    return h.hexdigest()[:16]

# =============================================================================
# Instruction pools
# =============================================================================
INSTRUCTION_CURRICULUM: Dict[str, Dict[str, List[str]]] = {
    "annomi": {
        "simple": [
            "Classify the client utterance as neutral, sustain, or change.",
            "Label this client statement: change, sustain, or neutral.",
            "Identify the motivational interviewing talk type here.",
        ],
        "medium": [
            "Based on motivational interviewing conventions, decide whether the utterance expresses change, sustain talk, or neutrality.",
            "Determine the client's motivational direction: change, sustain, or neutral.",
            "Analyze the client’s language to identify whether it moves toward or away from change.",
        ],
        "complex": [
            "Infer the underlying motivational dynamics expressed in the client's utterance and categorize its MI talk type.",
            "Consider the semantic, affective, and motivational cues in this utterance and determine the corresponding MI label.",
            "Evaluate linguistic markers to determine whether the client is articulating readiness, ambivalence, resistance, or neutrality.",
        ],
    },
    "mood": {
        "simple": [
            "Label the post as 0 (stable), IS (switch), or IE (escalation).",
            "Classify the mood: 0, IS, or IE.",
            "Does this message show stability, switch, or escalation?",
        ],
        "medium": [
            "Determine whether this post reflects emotional stability, a mood shift, or an escalation in emotional intensity.",
            "Analyze the user’s mood trajectory and classify the current post accordingly.",
            "Given the emotional tone of the message, identify if it indicates a shift or escalation.",
        ],
        "complex": [
            "Infer the latent emotional trajectory expressed in the post and map it onto the categories of stability, mood switching, or escalation.",
            "Consider the affective and contextual cues to determine whether the emotional state diverges from baseline.",
            "Assess the user’s evolving sentiment profile and categorize this post within the mood-change continuum.",
        ],
    },
    "lrs": {
        "simple": [
            "Label this post as stable (0) or switch (1).",
            "Does the user's stance stay the same or change?",
            "Classify stance as stable (0) or switched (1).",
        ],
        "medium": [
            "Determine whether this post reflects continuity or a stance shift relative to earlier posts.",
            "Analyze the stance trajectory and decide whether a switch occurs here.",
            "Identify whether this message deviates from prior stance behavior in the timeline.",
        ],
        "complex": [
            "Infer whether this post indicates a latent stance shift given longitudinal context.",
            "Evaluate longitudinal stance drift and classify whether a meaningful switch occurs.",
            "Assess contextual stance evolution and decide if the stance remains stable or switches.",
        ],
    },
}

def get_instruction(task_type: str, stage: int) -> str:
    level = "simple" if stage == 1 else ("medium" if stage == 2 else "complex")
    pools = INSTRUCTION_CURRICULUM.get(task_type, {})
    candidates = pools.get(level, [])
    return random.choice(candidates) if candidates else "Analyze the input and provide the appropriate label."

# =============================================================================
# History trimming (<HIST>) + explicit time markers
# =============================================================================
def _format_history_with_time_indices(history_lines: List[str]) -> List[str]:
    """
    Convert raw history list [oldest..newest] into:
      t-(K): ...
      ...
      t-2: ...
      t-1: ...
    so the model sees explicit recency.
    """
    K = len(history_lines)
    out = []
    for i, line in enumerate(history_lines):
        rel = K - i
        out.append(f"t-{rel}: {line}")
    return out

def trim_history_to_fit(
    header: str,
    history_lines: List[str],
    current_block: str,
    label_space_line: str,
    max_tokens: int
) -> Tuple[str, int]:
    """
    Returns:
      (text, history_len_used)
    history is trimmed from OLDEST side until the token budget fits.
    """
    hist = list(history_lines)
    while True:
        hist_marked = _format_history_with_time_indices(hist)

        parts = [header.strip(), ""]
        if hist_marked:
            parts.append("History (from oldest to most recent):")
            parts.append("<HIST>")
            parts.extend(hist_marked)
            parts.append("</HIST>")
            parts.append("")
        parts.append(current_block.strip())
        if label_space_line:
            parts.append(label_space_line.strip())

        text = "\n".join(parts).strip() + "\n"
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens <= max_tokens or not hist:
            return text, len(hist)
        hist.pop(0)

# =============================================================================
# Few-shot demo formatting (NO history in demos)
# =============================================================================
def build_demo_block(demos: List[Dict[str, Any]]) -> str:
    if not demos:
        return ""
    out = []
    used = 0
    for d in demos:
        demo_repr = str(d.get("demo_repr", "")).strip()
        if not demo_repr:
            continue
        out.append(f"### Demonstration Example {used + 1}")
        out.append(f"Input:\n{demo_repr}")
        out.append(f"Output: {d['gold_output']}\n")
        used += 1
    return "\n".join(out).strip()

# =============================================================================
# k-shot builder
# =============================================================================
def make_k_shot(
    dataset: List[Dict[str, Any]],
    query_item: Dict[str, Any],
    k: int,
    stage: int
) -> Optional[Dict[str, Any]]:
    task = query_item["task_type"]
    demos: List[Dict[str, Any]] = []

    # Longitudinal demos from same group
    if task in ("annomi", "mood", "lrs"):
        def _get_group_key(it):
            m = it.get("meta", {}) or {}
            if task == "annomi":
                return m.get("topic", m.get("_dummy_topic", "all"))
            if task == "mood":
                return m.get("user_id", m.get("_dummy_uid", "all"))
            return m.get("timeline_id", m.get("_dummy_tid", "all"))

        group_key = _get_group_key(query_item)
        same_group = [d for d in dataset if d["task_type"] == task and _get_group_key(d) == group_key]

        qid = query_item.get("example_id")
        query_index = None
        for idx, d in enumerate(same_group):
            if d.get("example_id") == qid:
                query_index = idx
                break

        previous_items = same_group[:query_index] if (query_index is not None and query_index > 0) else []
        demos = previous_items[-k:] if (k > 0 and previous_items) else []

    demo_block = build_demo_block(demos)
    instruction = get_instruction(task, stage)

    def build_full_input(demo_block_str: str) -> str:
        parts = [
            "<instruction>", instruction.strip(), "</instruction>", "",
            "<few-shot>", (demo_block_str.strip() if demo_block_str else ""), "</few-shot>", "",
            "<query>", query_item["input_repr"].strip(), "</query>", "",
            "<output>",
        ]
        return "\n".join([p for p in parts if p is not None]).strip() + "\n"

    full_input = build_full_input(demo_block)
    gold_output = str(query_item["gold_output"])
    combined_text = full_input + gold_output
    n_tokens = len(tokenizer(combined_text, add_special_tokens=False)["input_ids"])

    # Drop demos until fits
    if n_tokens > MAX_CONTEXT_FOR_BUILDER:
        original_k_req = k
        while len(demos) > 0 and n_tokens > MAX_CONTEXT_FOR_BUILDER:
            demos = demos[:-1]
            demo_block = build_demo_block(demos)
            full_input = build_full_input(demo_block)
            combined_text = full_input + gold_output
            n_tokens = len(tokenizer(combined_text, add_special_tokens=False)["input_ids"])
        if n_tokens > MAX_CONTEXT_FOR_BUILDER:
            log_event(
                f"[WARN] Dropping example task={task} tokens={n_tokens} > {MAX_CONTEXT_FOR_BUILDER} "
                f"(requested_k={original_k_req})"
            )
            return None

    # carry through longitudinal metadata for trainer
    meta = query_item.get("meta", None) or {}
    return {
        "input": full_input,
        "output": query_item["gold_output"],
        "task_type": query_item["task_type"],
        "raw_label_id": int(query_item["label_id"]),
        "k_shot_used": int(len(demos) if demos else 0),
        "stage": int(stage),
        "meta": meta,
        "example_id": query_item.get("example_id"),
        "group_id": query_item.get("group_id"),
        "timestep": int(query_item.get("timestep", 0)),
        "group_size": int(query_item.get("group_size", 0)),
        "history_len_used": int(query_item.get("history_len_used", 0)),
    }

# =============================================================================
# Loaders (now add timestep/group_id/group_size/history_len_used)
# =============================================================================
def load_annomi(path: str) -> List[Dict[str, Any]]:
    if not file_exists_any(path):
        log_event(f"[WARN] ANNoMI not found: {path}")
        return []

    df = read_csv_any(path)
    items: List[Dict[str, Any]] = []
    label_map = {"neutral": 0, "sustain": 1, "change": 2}

    if "topic" not in df.columns:
        df["_dummy_topic"] = "all"
        topic_col = "_dummy_topic"
        log_event("[WARN] ANNoMI missing 'topic' column; using dummy topic.")
    else:
        topic_col = "topic"

    sort_cols = []
    if "utterance_id" in df.columns:
        sort_cols.append("utterance_id")
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    if not sort_cols:
        df = df.reset_index()
        sort_cols = ["index"]

    for topic, g in df.groupby(topic_col):
        g_sorted = g.sort_values(by=sort_cols)
        rows = list(g_sorted.iterrows())
        group_size = len(rows)

        history_lines: List[str] = []
        for t_idx, (_, row) in enumerate(rows):
            utt = str(row.get("utterance_text", ""))
            speaker = str(row.get("interlocutor", "client")).strip()
            label_raw = str(row.get("client_talk_type", "")).strip().lower()

            group_id = f"annomi::{topic}"

            header = (
                "Task: Classify the client's talk type in motivational interviewing.\n"
                f"Group: {group_id}\n"
                f"Timestep: {t_idx+1}/{group_size}\n"
                f"History_len: {len(history_lines)}\n"
                f"Topic: {topic}\n"
            )
            current_block = f"Current utterance (client): {utt}"
            label_space = "Label space: neutral, sustain, change."
            demo_repr = f"{current_block}\n{label_space}"

            if speaker.lower() == "client" and label_raw in label_map:
                input_repr, hist_used = trim_history_to_fit(
                    header=header,
                    history_lines=history_lines,
                    current_block=current_block,
                    label_space_line=label_space,
                    max_tokens=MAX_TOKENS_PER_INPUT,
                )
                meta = dict(row)
                meta["group_id"] = group_id
                meta["timestep"] = t_idx
                meta["group_size"] = group_size
                meta["history_len_used"] = hist_used

                ex_id = stable_example_id("annomi", meta, input_repr)

                items.append({
                    "task_type": "annomi",
                    "input_repr": input_repr,
                    "demo_repr": demo_repr,
                    "gold_output": label_raw,
                    "label_id": label_map[label_raw],
                    "example_id": ex_id,
                    "meta": meta,
                    "group_id": group_id,
                    "timestep": t_idx,
                    "group_size": group_size,
                    "history_len_used": hist_used,
                })

            history_lines.append(f"{speaker}: {utt}")

    return items

def load_mood(path: str) -> List[Dict[str, Any]]:
    if not file_exists_any(path):
        log_event(f"[WARN] Mood not found: {path}")
        return []

    df = read_csv_any(path)
    items: List[Dict[str, Any]] = []
    label_map = {"0": 0, "IS": 1, "IE": 2}

    if "user_id" not in df.columns:
        df["_dummy_uid"] = df.index
        user_col = "_dummy_uid"
        log_event("[WARN] TalkLife missing 'user_id'; using dummy ids.")
    else:
        user_col = "user_id"

    time_col = "time_epoch_days" if "time_epoch_days" in df.columns else ("datetime" if "datetime" in df.columns else None)
    df = df.sort_values(by=[user_col, time_col] if time_col else [user_col])

    for uid, g in df.groupby(user_col):
        rows = list(g.iterrows())
        group_size = len(rows)

        history_lines: List[str] = []
        for t_idx, (_, row) in enumerate(rows):
            text = str(row.get("content", ""))
            label_raw = str(row.get("label", "0")).strip().upper()
            label_id = label_map.get(label_raw, 0)
            dt = str(row.get("datetime", row.get("time_epoch_days", "")))

            group_id = f"mood::{uid}"

            header = (
                "Task: Detect mood change over time in a user's social media timeline.\n"
                f"Group: {group_id}\n"
                f"Timestep: {t_idx+1}/{group_size}\n"
                f"History_len: {len(history_lines)}\n"
                f"User ID: {uid}\n"
            )
            current_block = f"Current post [{dt}]:\n{text}"
            label_space = "Label space: 0 (stable), IS (switch), IE (escalation)."
            demo_repr = f"{current_block}\n{label_space}"

            input_repr, hist_used = trim_history_to_fit(
                header=header,
                history_lines=history_lines,
                current_block=current_block,
                label_space_line=label_space,
                max_tokens=MAX_TOKENS_PER_INPUT,
            )

            meta = dict(row)
            meta["group_id"] = group_id
            meta["timestep"] = t_idx
            meta["group_size"] = group_size
            meta["history_len_used"] = hist_used

            ex_id = stable_example_id("mood", meta, input_repr)

            items.append({
                "task_type": "mood",
                "input_repr": input_repr,
                "demo_repr": demo_repr,
                "gold_output": label_raw,
                "label_id": label_id,
                "meta": meta,
                "example_id": ex_id,
                "group_id": group_id,
                "timestep": t_idx,
                "group_size": group_size,
                "history_len_used": hist_used,
            })

            history_lines.append(f"[{dt}] {text}")

    return items

def load_lrs(path: str) -> List[Dict[str, Any]]:
    if not file_exists_any(path):
        log_event(f"[WARN] LRS not found: {path}")
        return []

    df = read_csv_any(path)
    required = {"timeline_id", "text", "datetime", "label_binary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] LRS missing columns: {missing}")

    df = df.sort_values(by=["timeline_id", "datetime"])
    items: List[Dict[str, Any]] = []

    for tid, g in df.groupby("timeline_id"):
        rows = list(g.iterrows())
        group_size = len(rows)

        history_lines: List[str] = []
        for t_idx, (_, row) in enumerate(rows):
            text = str(row.get("text", ""))
            dt = str(row.get("datetime", ""))

            label_bin = row.get("label_binary", 0)
            label_id = 1 if int(label_bin) == 1 else 0
            label_raw = "1" if label_id == 1 else "0"

            group_id = f"lrs::{tid}"

            header = (
                "Task: Detect stance switches over time in a rumor discussion.\n"
                f"Group: {group_id}\n"
                f"Timestep: {t_idx+1}/{group_size}\n"
                f"History_len: {len(history_lines)}\n"
                f"Timeline ID: {tid}\n"
            )
            current_block = f"Current post [{dt}]:\n{text}"
            label_space = "Label space: 0 (stable), 1 (switch)."
            demo_repr = f"{current_block}\n{label_space}"

            input_repr, hist_used = trim_history_to_fit(
                header=header,
                history_lines=history_lines,
                current_block=current_block,
                label_space_line=label_space,
                max_tokens=MAX_TOKENS_PER_INPUT,
            )

            meta = dict(row)
            meta["group_id"] = group_id
            meta["timestep"] = t_idx
            meta["group_size"] = group_size
            meta["history_len_used"] = hist_used

            ex_id = stable_example_id("lrs", meta, input_repr)

            items.append({
                "task_type": "lrs",
                "input_repr": input_repr,
                "demo_repr": demo_repr,
                "gold_output": label_raw,
                "label_id": label_id,
                "meta": meta,
                "example_id": ex_id,
                "group_id": group_id,
                "timestep": t_idx,
                "group_size": group_size,
                "history_len_used": hist_used,
            })

            history_lines.append(f"[{dt}] {text}")

    return items

# =============================================================================
# Staging
# =============================================================================
def _validate_no_duplicates_across_stages(stages: List[List[Dict[str, Any]]]) -> None:
    seen = {}
    dups = []
    for si, st in enumerate(stages, start=1):
        for ex in st:
            eid = ex.get("example_id")
            if not eid:
                continue
            if eid in seen:
                dups.append((eid, seen[eid], si))
            else:
                seen[eid] = si
    if dups:
        msg = "\n".join([f"example_id={eid} stage_first={a} stage_dup={b}" for (eid, a, b) in dups[:50]])
        raise RuntimeError(f"Duplicate example_id across stages detected:\n{msg}")

def build_curriculum_stages_fixed_tasks(unified_items: List[Dict[str, Any]]):
    """
    Fixed mapping:
      Stage 1: ANNoMI (target k=1)
      Stage 2: LRS    (target k=2)
      Stage 3: MOOD   (target k=3)
    """
    stage1, stage2, stage3 = [], [], []
    dropped = 0

    task_to_stage = {"annomi": 1, "lrs": 2, "mood": 3}
    stage_to_target_k = {1: 1, 2: 2, 3: 3}

    seen_example_ids = set()

    def emit(stage_idx: int, ex: Dict[str, Any]):
        if stage_idx == 1:
            stage1.append(ex)
        elif stage_idx == 2:
            stage2.append(ex)
        else:
            stage3.append(ex)

    # Keep items ordered by (task, group_id, timestep) to preserve clean demo selection
    def _sort_key(it):
        return (it.get("task_type", ""), it.get("group_id", ""), int(it.get("timestep", 0)))

    for item in sorted(unified_items, key=_sort_key):
        t = item.get("task_type")
        if t not in task_to_stage:
            continue
        stage_idx = task_to_stage[t]
        requested_k = stage_to_target_k[stage_idx]

        ex_id = item.get("example_id")
        if ex_id and ex_id in seen_example_ids:
            continue

        ex = make_k_shot(unified_items, item, requested_k, stage_idx)
        if ex is None:
            dropped += 1
            continue

        if ex_id:
            seen_example_ids.add(ex_id)
        emit(stage_idx, ex)

    if STRICT_DEDUP:
        _validate_no_duplicates_across_stages([stage1, stage2, stage3])

    return stage1, stage2, stage3, dropped

def stage_stats(stage_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    n = len(stage_items)
    if n == 0:
        return {"n": 0, "tasks": {}, "k_shot_used": {}, "stages": {}}
    tasks = Counter([x.get("task_type", "unknown") for x in stage_items])
    ks = Counter([int(x.get("k_shot_used", 0)) for x in stage_items])
    st = Counter([int(x.get("stage", 0)) for x in stage_items])
    return {"n": int(n), "tasks": dict(tasks), "k_shot_used": dict(ks), "stages": dict(st)}

def save_stage_to_jsonl_s3(stage_items: List[Dict[str, Any]], s3_path: str) -> None:
    tmp = s3_path + ".tmp"
    with fs.open(tmp, "w") as f:
        for ex in stage_items:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write("\n")
    try:
        fs.copy(tmp, s3_path)
        fs.rm(tmp)
    except Exception:
        with fs.open(s3_path, "w") as f:
            for ex in stage_items:
                f.write(json.dumps(ex, ensure_ascii=False))
                f.write("\n")
        try:
            fs.rm(tmp)
        except Exception:
            pass
    log_event(f"Wrote {len(stage_items)} examples -> {s3_path}")

# =============================================================================
# Main
# =============================================================================
def run_builder() -> None:
    log_event("Builder start")
    log_event(f"BASE_MODEL_NAME={BASE_MODEL_NAME}")
    log_event(f"CURRICULUM_PREFIX_S3={CURRICULUM_PREFIX_S3}")
    log_event(f"MAX_CONTEXT_FOR_BUILDER={MAX_CONTEXT_FOR_BUILDER} | MAX_TOKENS_PER_INPUT={MAX_TOKENS_PER_INPUT}")

    assert_inputs_exist()

    annomi_items = load_annomi(PATH_ANNOMI)
    mood_items   = load_mood(PATH_MOOD)
    lrs_items    = load_lrs(PATH_LRS)

    log_event(f"ANNoMI items:   {len(annomi_items)}")
    log_event(f"TalkLife items: {len(mood_items)}")
    log_event(f"LRS items:      {len(lrs_items)}")

    unified = annomi_items + lrs_items + mood_items
    if not unified:
        raise RuntimeError("Builder loaded zero items. Check paths/formats.")

    log_event(f"Total unified items: {len(unified)}")

    stage1, stage2, stage3, dropped = build_curriculum_stages_fixed_tasks(unified)

    save_stage_to_jsonl_s3(stage1, f"{CURRICULUM_PREFIX_S3}_stage1.jsonl")
    save_stage_to_jsonl_s3(stage2, f"{CURRICULUM_PREFIX_S3}_stage2.jsonl")
    save_stage_to_jsonl_s3(stage3, f"{CURRICULUM_PREFIX_S3}_stage3.jsonl")

    stats = {
        "curriculum_prefix_s3": CURRICULUM_PREFIX_S3,
        "dropped_examples_due_to_context_cap": int(dropped),
        "stage1": stage_stats(stage1),
        "stage2": stage_stats(stage2),
        "stage3": stage_stats(stage3),
        "builder_cap_tokens": int(MAX_CONTEXT_FOR_BUILDER),
        "max_tokens_per_input": int(MAX_TOKENS_PER_INPUT),
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "builder_mode": "fixed_tasks_stage1_annomi_k1_stage2_lrs_k2_stage3_mood_k3_no_stage4",
        "max_k": 3,
        "grouping": {"annomi": "topic", "mood": "user_id", "lrs": "timeline_id"},
        "strict_dedup": bool(STRICT_DEDUP),
        "outputs": {
            "stage1": f"{CURRICULUM_PREFIX_S3}_stage1.jsonl",
            "stage2": f"{CURRICULUM_PREFIX_S3}_stage2.jsonl",
            "stage3": f"{CURRICULUM_PREFIX_S3}_stage3.jsonl",
        },
        "no_stage4": True,
        "no_timeline_dataset": True,
        "longitudinal_upgrades": {
            "history_time_indices": True,
            "header_has_timestep_and_history_len": True,
            "meta_has_group_id_timestep_group_size": True,
        }
    }

    s3_write_text_atomic(fs, s3_join(OUTPUT_S3, "builder_stats.json"), json.dumps(stats, indent=2))
    log_event("Builder finished")

def main():
    run_builder()

if __name__ == "__main__":
    main()


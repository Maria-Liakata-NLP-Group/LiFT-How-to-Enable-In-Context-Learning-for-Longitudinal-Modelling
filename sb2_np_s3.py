import os
import sys
import subprocess
import shutil
import traceback

def sh(cmd):
    print(f"🧪 RUN: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def env_required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise ValueError(f"Missing required env var: {name}")
    return v

print("=" * 60)
print("🛠️  iFT S3 WRAPPER START")
print("=" * 60)

WORKDIR = os.environ.get("IFT_WORKDIR", "/tmp/ift_run")
ensure_dir(WORKDIR)

VENV_PATH = os.path.join(WORKDIR, "venv")
PYTHON_VENV = os.path.join(VENV_PATH, "bin", "python")

if not os.path.exists(PYTHON_VENV):
    print(f"🐍 Creating venv at {VENV_PATH}", flush=True)
    sh([sys.executable, "-m", "venv", VENV_PATH])

print(f"📌 WORKDIR = {WORKDIR}")
print(f"📌 VENV    = {VENV_PATH}")
print(f"📌 PYTHON  = {PYTHON_VENV}")

print("📦 Upgrading pip/setuptools/wheel...", flush=True)
sh([
    PYTHON_VENV, "-m", "pip", "install",
    "--upgrade", "--no-cache-dir",
    "pip>=24.0", "setuptools>=68.0", "wheel"
])

print("📦 Installing required packages...", flush=True)

pkgs = [
    "numpy<2.0",
    "pyarrow==15.0.2",

    "transformers==4.45.2",
    "accelerate==0.34.2",
    "datasets==2.20.0",
    "peft==0.13.2",
    "safetensors",

    "pandas",
    "fsspec",
    "s3fs",
    "boto3",
    "tensorboard",
    "scikit-learn",
]

sh([
    PYTHON_VENV, "-m", "pip", "install",
    "--upgrade",
    "--no-cache-dir",
    "--prefer-binary",
] + pkgs)

print("📦 Versions:", flush=True)
sh([
    PYTHON_VENV,
    "-c",
    "import transformers, accelerate, datasets, pyarrow; "
    "print('transformers', transformers.__version__); "
    "print('accelerate', accelerate.__version__); "
    "print('datasets', datasets.__version__); "
    "print('pyarrow', pyarrow.__version__)"
])

env_required("FSSPEC_S3_ENDPOINT_URL")
env_required("FSSPEC_S3_KEY")
env_required("FSSPEC_S3_SECRET")

DATASET_S3 = os.environ.get("IFT_DATASET_S3", "").strip()
OUTPUT_S3 = env_required("IFT_OUTPUT_S3")
NEWIFTDATA_S3 = env_required("IFT_NEWIFTDATA_S3")

print(f"📌 IFT_DATASET_S3     = {DATASET_S3}")
print(f"📌 IFT_OUTPUT_S3      = {OUTPUT_S3}")
print(f"📌 IFT_NEWIFTDATA_S3  = {NEWIFTDATA_S3}")

if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

JOB_TYPE = os.environ.get("IFT_JOB_TYPE", "both").strip().lower()
if JOB_TYPE not in ("build", "train", "both"):
    raise ValueError("IFT_JOB_TYPE must be one of: build, train, both")

SCRIPT_DIR = os.environ.get(
    "IFT_SCRIPT_DIR",
    "/home/jovyan/qp24875-shared-fs/LIFT_CODE_ALL/Driver_script_ift_llama8b"
)

BUILDER_SCRIPT = os.path.join(SCRIPT_DIR, "ift_builder.py")
TRAINER_SCRIPT = os.path.join(SCRIPT_DIR, "ift_trainer_fullmodelcopy.py")

print(f"📌 SCRIPT_DIR     = {SCRIPT_DIR}")
print(f"📌 BUILDER_SCRIPT = {BUILDER_SCRIPT}")
print(f"📌 TRAINER_SCRIPT = {TRAINER_SCRIPT}")

ok = False

try:
    if JOB_TYPE in ("build", "both"):
        if not DATASET_S3:
            raise ValueError("IFT_DATASET_S3 is required when IFT_JOB_TYPE is build or both.")
        if not os.path.exists(BUILDER_SCRIPT):
            raise FileNotFoundError(f"Missing builder script at {BUILDER_SCRIPT}")

        print("🏗️  Running BUILDER...", flush=True)
        sh([PYTHON_VENV, BUILDER_SCRIPT])

    if JOB_TYPE in ("train", "both"):
        if not os.path.exists(TRAINER_SCRIPT):
            raise FileNotFoundError(f"Missing trainer script at {TRAINER_SCRIPT}")

        print("🎓 Running TRAINER...", flush=True)
        sh([PYTHON_VENV, TRAINER_SCRIPT])

    ok = True

except Exception as e:
    print("❌ Job failed.", flush=True)
    print("Exception:", repr(e), flush=True)
    traceback.print_exc()
    raise

finally:
    force_cleanup = os.environ.get("IFT_FORCE_CLEANUP", "0") == "1"

    if ok or force_cleanup:
        print(f"🧹 Cleaning WORKDIR: {WORKDIR}", flush=True)
        shutil.rmtree(WORKDIR, ignore_errors=True)
    else:
        print(f"🧾 Keeping WORKDIR for debugging: {WORKDIR}", flush=True)

print("🎉 DONE.")
print("=" * 60)
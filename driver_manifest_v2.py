import json
import os
from kubernetes import client, config

SECRET_FILE = "/home/jovyan/qp24875-shared-fs/LIFT_CODE_ALL/Driver_script_ift_llama8b/s3_credentials (1).json"
if not os.path.exists(SECRET_FILE):
    raise FileNotFoundError(f"Missing {SECRET_FILE}")

with open(SECRET_FILE, "r") as f:
    s3_creds = json.load(f)

JOB_NAME = "ift-llama8b-curriculum-h100"
NAMESPACE = "kf-qp24875"

IMAGE_NAME = "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime"
SCRIPT_PATH = "/home/jovyan/qp24875-shared-fs/LIFT_CODE_ALL/Driver_script_ift_llama8b/sb2_np_s3.py"

DATASET_S3 = "s3://nlp-01/ia180/trainingdataICL/"
OUTPUT_S3 = "s3://nlp-01/ia180/IFT_OUTPUTCuricullum/llama8bnew/"
NEWIFTDATA_S3 = "s3://nlp-01/ia180/NEWIFTDATACurriculum/llama8bnew"
CURRICULUM_PREFIX_S3 = "s3://nlp-01/ia180/NEWIFTDATACurriculum/llama8bnew/ICL_curriculum_IFT_llama8b"

IFT_JOB_TYPE = "train"
HF_TOKEN_VALUE = os.environ["HF_TOKEN"]

MOUNTPATH = "/home/jovyan/qp24875-shared-fs"
CLAIMNAME = "qp24875-shared-fs"
IFT_WORKDIR = "/tmp/ift_run"

access_key = s3_creds["FSSPEC_S3_KEY"]
secret_key = s3_creds["FSSPEC_S3_SECRET"]
endpoint = s3_creds["FSSPEC_S3_ENDPOINT_URL"]

manifest = {
    "apiVersion": "kubeflow.org/v1",
    "kind": "PyTorchJob",
    "metadata": {
        "name": JOB_NAME,
        "namespace": NAMESPACE,
        "labels": {"kueue.x-k8s.io/queue-name": "nlp"},
    },
    "spec": {
        "pytorchReplicaSpecs": {
            "Master": {
                "replicas": 1,
                "restartPolicy": "OnFailure",
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "pytorch",
                                "image": IMAGE_NAME,
                                "command": [
                                    "bash",
                                    "-c",
                                    f"""
set -e

echo "Installing system dependencies..."
apt-get update
apt-get install -y wget bzip2 ca-certificates build-essential gcc g++ curl pkg-config git

echo "Installing Miniconda Python 3.10..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /tmp/conda310

export PATH=/tmp/conda310/bin:$PATH

echo "Python:"
python --version

python -m pip install --upgrade pip setuptools wheel

echo "Installing Rust..."
curl https://sh.rustup.rs -sSf | bash -s -- -y
source $HOME/.cargo/env

rustc --version
cargo --version

echo "Running wrapper..."
python {SCRIPT_PATH}
"""
                                ],
                                "env": [
                                    {"name": "FSSPEC_S3_ENDPOINT_URL", "value": endpoint},
                                    {"name": "FSSPEC_S3_KEY", "value": access_key},
                                    {"name": "FSSPEC_S3_SECRET", "value": secret_key},
                                    {"name": "HF_TOKEN", "value": HF_TOKEN_VALUE},
                                    {"name": "HUGGINGFACE_HUB_TOKEN", "value": HF_TOKEN_VALUE},
                                    {"name": "IFT_DATASET_S3", "value": DATASET_S3},
                                    {"name": "IFT_OUTPUT_S3", "value": OUTPUT_S3},
                                    {"name": "IFT_NEWIFTDATA_S3", "value": NEWIFTDATA_S3},
                                    {"name": "CURRICULUM_PREFIX_S3", "value": CURRICULUM_PREFIX_S3},
                                    {"name": "IFT_JOB_TYPE", "value": IFT_JOB_TYPE},
                                    {"name": "IFT_WORKDIR", "value": IFT_WORKDIR},
                                    {"name": "BASE_MODEL_NAME", "value": "meta-llama/Llama-3.1-8B-Instruct"},
                                    {"name": "BUILDER_CAP", "value": "4096"},
                                    {"name": "MAX_TOKENS_PER_INPUT", "value": "4096"},
                                    {"name": "STRICT_DEDUP", "value": "1"},
                                ],
                                "resources": {
                                    "limits": {
                                        "nvidia.com/h100-96g": "1",
                                        "cpu": "32",
                                        "memory": "64Gi",
                                    }
                                },
                                "volumeMounts": [
                                    {"name": "workspace", "mountPath": MOUNTPATH}
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "workspace",
                                "persistentVolumeClaim": {"claimName": CLAIMNAME},
                            }
                        ],
                    }
                },
            }
        }
    },
}

if __name__ == "__main__":
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    print(f"🚀 Submitting job {JOB_NAME}")

    api.create_namespaced_custom_object(
        group="kubeflow.org",
        version="v1",
        namespace=NAMESPACE,
        plural="pytorchjobs",
        body=manifest,
    )
# ift_s3_utils.py
import os
import datetime
import fsspec


def env_required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise ValueError(f"Missing required env var: {name}")
    return v


def log_event(message: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def s3_join(prefix: str, *parts: str) -> str:
    prefix = prefix.rstrip("/")
    rest = "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None and p != ""])
    return f"{prefix}/{rest}" if rest else prefix


def make_s3_fs():
    storage_options = {
        "key": env_required("FSSPEC_S3_KEY"),
        "secret": env_required("FSSPEC_S3_SECRET"),
        "client_kwargs": {"endpoint_url": env_required("FSSPEC_S3_ENDPOINT_URL")},
    }
    return fsspec.filesystem("s3", **storage_options)


def s3_put_dir(fs, local_dir: str, s3_dir: str, recursive: bool = True) -> None:
    if not os.path.exists(local_dir):
        return
    log_event(f"📤 S3 UPLOAD DIR: {local_dir} -> {s3_dir}")
    fs.put(local_dir, s3_dir, recursive=recursive)


def s3_put_file(fs, local_file: str, s3_file: str) -> None:
    if not os.path.exists(local_file):
        return
    log_event(f"📤 S3 UPLOAD FILE: {local_file} -> {s3_file}")
    fs.put(local_file, s3_file)


def s3_get_dir(fs, s3_dir: str, local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    log_event(f"📥 S3 DOWNLOAD DIR: {s3_dir} -> {local_dir}")
    fs.get(s3_dir, local_dir, recursive=True)


def s3_get_file(fs, s3_file: str, local_file: str) -> None:
    os.makedirs(os.path.dirname(local_file) or ".", exist_ok=True)
    log_event(f"📥 S3 DOWNLOAD FILE: {s3_file} -> {local_file}")
    fs.get(s3_file, local_file)


def s3_write_text(fs, s3_path: str, text: str) -> None:
    with fs.open(s3_path, "w") as f:
        f.write(text)


def s3_read_text(fs, s3_path: str):
    if not fs.exists(s3_path):
        return None
    with fs.open(s3_path, "r") as f:
        return f.read()


def s3_exists(fs, s3_path: str) -> bool:
    return bool(fs.exists(s3_path))


def s3_write_text_atomic(fs, s3_path: str, text: str) -> None:
    tmp = s3_path + ".tmp"
    s3_write_text(fs, tmp, text)
    try:
        fs.copy(tmp, s3_path)
        fs.rm(tmp)
    except Exception:
        s3_write_text(fs, s3_path, text)
        try:
            fs.rm(tmp)
        except Exception:
            pass

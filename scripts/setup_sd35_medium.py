from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

LOCAL_DIR = ROOT_DIR / "models" / "sd35_medium"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

SD35_LARGE_DIR = ROOT_DIR / "models" / "sd35_large"
REPO_ID = "stabilityai/stable-diffusion-3.5-medium"
LOG_FILE = ROOT_DIR / "sd35m_run.log"

SHARED_DIRS = [
    "tokenizer",
    "tokenizer_2",
    "tokenizer_3",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "vae",
    "scheduler",
]

FILES_TO_DOWNLOAD = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
]


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def setup_sd35_medium():
    log("=" * 80)
    log("SETTING UP STABLE DIFFUSION 3.5 MEDIUM")
    log("=" * 80)

    # 1. Copy shared text encoders, tokenizers, vae from local sd35_large
    log("\n--- [Step 1] Linking / Copying Shared Components from Local sd35_large ---")
    for comp in SHARED_DIRS:
        src = SD35_LARGE_DIR / comp
        dst = LOCAL_DIR / comp
        if src.exists() and not dst.exists():
            log(f"[+] Copying {comp} ...")
            shutil.copytree(src, dst)
        elif dst.exists():
            log(f"[OK] {comp} already present.")

    # 2. Download Medium-specific files (transformer & config)
    log("\n--- [Step 2] Downloading Medium-Specific Transformer & Configs ---")
    import huggingface_hub
    hf_token = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    auth_headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    for filename in FILES_TO_DOWNLOAD:
        dest = LOCAL_DIR / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = LOCAL_DIR / f"{filename}.part"

        url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
        r_head = requests.head(url, headers=auth_headers, allow_redirects=True, timeout=30)
        r_head.raise_for_status()
        total_size = int(r_head.headers.get("content-length", 0))

        if dest.exists() and dest.stat().st_size == total_size:
            log(f"[OK] Already complete: {filename} ({total_size / (1024**2):.1f} MB)")
            continue

        retries = 0
        max_retries = 100

        while retries < max_retries:
            initial_bytes = 0
            if temp_dest.exists():
                initial_bytes = temp_dest.stat().st_size
                if initial_bytes > total_size:
                    temp_dest.unlink()
                    initial_bytes = 0
                elif initial_bytes == total_size:
                    temp_dest.replace(dest)
                    log(f"[OK] Completed: {filename} ({total_size / (1024**2):.1f} MB)")
                    break

            headers = dict(auth_headers)
            if initial_bytes > 0:
                log(f"[->] Resuming {filename} from {initial_bytes / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB ({initial_bytes/total_size*100:.1f}%)")
                headers["Range"] = f"bytes={initial_bytes}-"
            else:
                log(f"[..] Downloading {filename} (Total: {total_size / (1024**2):.1f} MB)...")

            try:
                r = requests.get(url, headers=headers, stream=True, timeout=30)
                r.raise_for_status()

                mode = "ab" if initial_bytes > 0 else "wb"
                downloaded = initial_bytes
                t_start = time.time()
                t_last_log = t_start

                with open(temp_dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - t_last_log >= 15.0 or downloaded == total_size:
                                elapsed = now - t_start
                                speed = (downloaded - initial_bytes) / elapsed / (1024 * 1024) if elapsed > 0 else 0
                                pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                                log(f"    [{filename}]: {downloaded/(1024**2):.1f}/{total_size/(1024**2):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s")
                                t_last_log = now

                if temp_dest.exists() and temp_dest.stat().st_size == total_size:
                    temp_dest.replace(dest)
                    log(f"[+] Saved {filename} ({total_size / (1024**2):.1f} MB)")
                    break
            except Exception as exc:
                retries += 1
                log(f"[!] Network interrupted ({exc}), retrying in 3s... (attempt {retries}/{max_retries})")
                time.sleep(3)

    log("\n[+] SD 3.5 Medium Setup Complete!")


if __name__ == "__main__":
    setup_sd35_medium()

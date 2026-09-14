#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
from pathlib import Path

# --- Sabitler ---
QUEUE_FILE = '/tf/start_training/queue/queue.json'
LOG_DIR = Path('/tf/start_training/queue')
ERROR_LOG = LOG_DIR / 'error.log'
DONE_LOG  = LOG_DIR / 'done.log'
CHECK_INTERVAL = 10  # saniye

print("Tracker started. Watching queue.json...", flush=True)

# --- Yardımcılar ---

def read_queue():
    with open(QUEUE_FILE, 'r') as f:
        return json.load(f)

def write_queue(queue):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)

def log_to_file(log_path, message):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(f"[{ts}] {message}\n")

def sh(cmd, check=True, capture_output=True, text=True, cwd=None):
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text, cwd=cwd)

# --- Branch -> Model eşlemesi ---

BRANCH_MODEL_MAP = {
    r'\byamnet\b':       'yamnet',
    r'\btimesnet\b':     'timesnet',
    r'\bcrnn\b':         'crnn',
    r'\brelation[_-]?net\b': 'relation_net',
    r'\bcnn\b':          'cnn',
}

def detect_branch(item: dict) -> str:
    # Öncelik: CI’dan gelen git_branch
    if item.get('git_branch'):
        return str(item['git_branch'])
    # Alternatif: repo_dir varsa git’ten oku (opsiyonel)
    repo = item.get('repo_dir')
    if repo and Path(repo).exists():
        out = sh(['git', '-C', repo, 'rev-parse', '--abbrev-ref', 'HEAD'])
        return out.stdout.strip()
    raise ValueError("Branch bilgisi yok. CI 'git_branch' alanını item’a eklemeli veya 'repo_dir' sağlamalısın.")

def branch_to_model_type(branch: str) -> str:
    low = branch.lower()
    for pat, mtype in BRANCH_MODEL_MAP.items():
        if re.search(pat, low):
            return mtype
    raise ValueError(f"Branch adından model tipi çıkarılamadı: '{branch}'")

# --- Komut inşası ---

def build_command_from_type(model_type: str, config: dict):
    params_key = f"{model_type}_parameters"
    params = config.get(params_key, {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Config içinde '{params_key}' yok ya da boş.")

    if model_type == "yamnet":
        return [
            "python", "/tf/start_training/YAMNET/train.py",
            "--data-csv", params["data_path"],
            "--dataset-cache-dir", params["dataset_cache_dir"],
            "--epochs", str(params.get("epochs", 500)),
            "--output-dir", params.get("output_dir", "./output"),
            "--model-name", "YAMNET " + str(params.get("batch_size", 16)),
            "--class-names", str(params["class_names"]),
            "--batch-size", str(params.get("batch_size", 8)),
            "--early-stopping-patience", str(params.get("early_stopping_patience", 50)),
            "--normalization-method", str(params.get("normalization_method")),
            "--num-of-groups", str(params.get("num_of_groups")),
        ]

    if model_type == "timesnet":
        cmd = [
            "python", "/tf/start_training/TIMESNET_V2/train_timesnet.py",
            "--csv_path", params["csv_path"],
            "--output_dir", params["output_dir"],
            "--batch", str(params.get("batch_size", 128)),
            "--label_smoothing", str(params.get("label_smoothing", 0.0)),
            "--epoch", str(params.get("epochs", 200)),
            "--run_name", params.get("run_name", "TimesNet_Run_Auto"),
            "--patience", str(params.get("patience", 12)),
            "--lr", str(params.get("lr", 0.001)),
            "--augmentation_type", str(params.get("augmentation_type", "onthefly")),
            "--loss", "focal" if params.get("loss", "ce") == "focal" else "ce",
        ]
        if params.get("cmvn", False):
            cmd.append("--cmvn")
        if params.get("specaug", False):
            cmd.append("--specaug")
        return cmd

    if model_type == "crnn":
        return [
            "python", "/tf/start_training/CRNN/train.py",
            "--batch", str(params["batch_size"]),
            "--epoch", str(params["epochs"]),
        ]

    if model_type == "relation_net":
        return [
            "python", "/tf/start_training/RELATIONNET/train.py",
            "--data_folder", params["data_folder"],
            "--batch", str(params["batch"]),
            "--epoch", str(params["epoch"]),
            "--patience", str(params["patience"]),
            "--lr", str(params["lr"]),
            "--n_way", str(params["n_way"]),
            "--k_shot", str(params["k_shot"]),
            "--q_query", str(params["q_query"]),
            "--model_name", params["model_name"],
        ]

    if model_type == "cnn":
        return [
            "python", "/tf/start_training/CNN/train.py",
            "--data_folder", params["data_folder"],
            "--batch", str(params["batch"]),
            "--epoch", str(params["epoch"]),
            "--patience", str(params["patience"]),
            "--output_shape", str(params["output_shape"]),
            "--classes", str(params["classes"]),
            "--thresholds", str(params["thresholds"]),
            "--model_name", params["model_name"],
        ]

    raise ValueError(f"Bilinmeyen model tipi: {model_type}")

# --- Çalıştırıcı ---

def run_training(command):
    print(f"Training started with command: {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
        return "done"
    except subprocess.CalledProcessError as e:
        print(f"Training process failed (returncode={e.returncode})", flush=True)
        return "error"

# --- Main loop ---

def main():
    print("Tracker started main loop...", flush=True)
    while True:
        updated = False
        queue = read_queue()
        print(f"Queue state: {[item.get('status') for item in queue]}", flush=True)

        for idx, item in enumerate(queue):
            if item.get("status") == "pending":
                print(f"Processing item {idx}...", flush=True)

                queue[idx]["status"] = "running"
                write_queue(queue)
                time.sleep(1)

                branch = None
                model_type = None
                try:
                    branch = detect_branch(queue[idx])
                    model_type = branch_to_model_type(branch)
                    print(f"[branch={branch}] -> model_type={model_type}", flush=True)

                    cmd = build_command_from_type(model_type, queue[idx])
                    result = run_training(cmd)

                except Exception as e:
                    msg = f"Hata: {e} | item={json.dumps(queue[idx])}"
                    print(msg, flush=True)
                    log_to_file(ERROR_LOG, msg)
                    result = "error"

                updated_queue = read_queue()
                updated_queue[idx]["status"] = result
                if branch:      updated_queue[idx]["_resolved_branch"] = branch
                if model_type:  updated_queue[idx]["_resolved_model_type"] = model_type
                write_queue(updated_queue)

                if result == "done":
                    log_to_file(DONE_LOG, f"Training finished for idx={idx} [branch={updated_queue[idx].get('_resolved_branch')}]")

                updated = True
                break

        if not updated:
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

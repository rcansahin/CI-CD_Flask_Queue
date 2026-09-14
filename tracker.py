#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
from pathlib import Path

# ------------------ Sabitler ------------------
QUEUE_FILE = '/tf/start_training/queue/queue.json'
LOG_DIR = Path('/tf/start_training/queue')
ERROR_LOG = LOG_DIR / 'error.log'
DONE_LOG  = LOG_DIR / 'done.log'
CHECK_INTERVAL = 10           # saniye
MAX_CONCURRENT = 2            # aynı anda en fazla kaç eğitim
BACKOFF_SECONDS = 300         # OOM varsayımıyla yeniden deneme gecikmesi 5 dk

print("Tracker started. Watching queue.json...", flush=True)

# Çalışan işler: {queue_index: Popen}
RUNNING = {}

# ------------------ Yardımcılar ------------------

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

def now() -> float:
    return time.time()

# ------------------ Branch -> Model eşlemesi ------------------

BRANCH_MODEL_MAP = {
    r'\byamnet\b':           'yamnet',
    r'\btimesnet\b':         'timesnet',
    r'\bcrnn\b':             'crnn',
    r'\brelation[_-]?net\b': 'relation_net',
    r'\bcnn\b':              'cnn',
}

def detect_branch(item: dict) -> str:
    if item.get('git_branch'):
        return str(item['git_branch'])
    repo = item.get('repo_dir')
    if repo and Path(repo).exists():
        out = sh(['git', '-C', repo, 'rev-parse', '--abbrev-ref', 'HEAD'])
        return out.stdout.strip()
    raise ValueError("Branch bilgisi yok. CI 'git_branch' eklemeli veya 'repo_dir' verilmelidir.")

def branch_to_model_type(branch: str) -> str:
    low = branch.lower()
    for pat, mtype in BRANCH_MODEL_MAP.items():
        if re.search(pat, low):
            return mtype
    raise ValueError(f"Branch adından model tipi çıkarılamadı: '{branch}'")

# ------------------ Priority & deneme zamanı ------------------

def _to_int_priority(v):
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return int(float(s))  # "2", "2.0" gibi durumlar
        except ValueError:
            return None
    return None

def extract_priority(item: dict, model_type: str | None) -> int:
    # Küçük sayı = yüksek öncelik. Ayarlanmamışsa düşük öncelik olsun.
    DEFAULT_LOW = 10**9

    v = _to_int_priority(item.get('priority'))
    if v is not None:
        return v

    if model_type:
        pk = f"{model_type}_parameters"
        params = item.get(pk, {})
        if isinstance(params, dict):
            v = _to_int_priority(params.get('priority'))
            if v is not None:
                return v

    for k, vdict in item.items():
        if isinstance(vdict, dict) and k.endswith('_parameters'):
            v = _to_int_priority(vdict.get('priority'))
            if v is not None:
                return v

    return DEFAULT_LOW


def should_try(item: dict) -> bool:
    if item.get('status') != 'pending':
        return False
    dt = item.get('deferred_until')
    if isinstance(dt, (int, float)) and now() < float(dt):
        return False
    return True

def set_backoff(queue: list, idx: int, reason: str):
    queue[idx]['status'] = 'pending'
    queue[idx]['deferred_until'] = now() + BACKOFF_SECONDS
    queue[idx]['retry_count'] = int(queue[idx].get('retry_count', 0)) + 1
    write_queue(queue)
    log_to_file(ERROR_LOG, f"idx={idx} backoff {BACKOFF_SECONDS}s ({reason}). retry_count={queue[idx]['retry_count']}")

# ------------------ Komut inşası ------------------

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
        if params.get("cmvn", False):    cmd.append("--cmvn")
        if params.get("specaug", False): cmd.append("--specaug")
        if "seed" in params:             cmd += ["--seed", str(params["seed"])]
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

# ------------------ Başlat / İzle ------------------

def launch_job(queue: list, idx: int):
    item = queue[idx]
    branch = detect_branch(item)
    model_type = branch_to_model_type(branch)
    cmd = build_command_from_type(model_type, item)

    print(f"[launch] idx={idx} branch={branch} model={model_type} cmd={' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd)  # asenkron
    queue[idx]['status'] = 'running'
    queue[idx]['pid'] = proc.pid
    queue[idx]['start_ts'] = int(now())
    write_queue(queue)

    RUNNING[idx] = proc

def reap_finished():
    finished = []
    for idx, proc in list(RUNNING.items()):
        rc = proc.poll()
        if rc is None:
            continue
        finished.append((idx, rc))
        del RUNNING[idx]

    if not finished:
        return

    queue = read_queue()
    print(f"Queue state: {[item.get('status') for item in queue]}", flush=True)
    for idx, rc in finished:
        if rc == 0:
            queue[idx]['status'] = 'done'
            write_queue(queue)
            log_to_file(DONE_LOG, f"Training finished idx={idx} pid={queue[idx].get('pid')}")
        else:
            if len(RUNNING) > 0:
                set_backoff(queue, idx, reason=f"returncode={rc}, concurrent_jobs={len(RUNNING)}")
            else:
                queue[idx]['status'] = 'error'
                write_queue(queue)
                log_to_file(ERROR_LOG, f"Training failed idx={idx} rc={rc} no other concurrent jobs")

def start_new_jobs_if_possible():
    slots = MAX_CONCURRENT - len(RUNNING)
    if slots <= 0:
        return

    queue = read_queue()

    candidate_indices = [i for i, it in enumerate(queue) if should_try(it)]
    print(f"[scheduler] slots={slots} running={len(RUNNING)} candidates={candidate_indices}", flush=True)
    scored = []
    for i in candidate_indices:
        try:
            b = detect_branch(queue[i])
            mt = branch_to_model_type(b)
            pr = extract_priority(queue[i], mt)
        except Exception:
            pr = 10**9  # en düşük öncelik
        scored.append((pr, i))

    scored.sort(key=lambda x: (x[0], x[1]))
    dbg_post = ", ".join([f"(idx={i}, pr={p})" for p, i in scored])
    print(f"[scheduler] post-sort: {dbg_post}", flush=True)
    started = 0
    for _, idx in scored:
        if started >= slots:
            break 
        try:
            launch_job(queue, idx)
            started += 1
        except Exception as e:
            queue = read_queue()
            if len(RUNNING) > 0:
                set_backoff(queue, idx, reason=f"launch_error: {e}")
            else:
                queue[idx]['status'] = 'error'
                write_queue(queue)
                log_to_file(ERROR_LOG, f"launch_error idx={idx}: {e}")

# ------------------ Main loop ------------------

def main():
    print("Tracker started main loop...", flush=True)
    while True:
        try:
            reap_finished()
            start_new_jobs_if_possible()
        except Exception as e:
            log_to_file(ERROR_LOG, f"loop_error: {e}")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

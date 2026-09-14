#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
from pathlib import Path

QUEUE_FILE = '/tf/start_training/queue/queue.json'
LOG_DIR = Path('/tf/start_training/queue')
ERROR_LOG = LOG_DIR / 'error.log'
DONE_LOG  = LOG_DIR / 'done.log'
CHECK_INTERVAL = 5          # saniye
RETRY_AFTER_SEC = 300       # 5 dk
MAX_CONCURRENT = 2          # aynı anda en fazla N eğitim

BRANCH_MODEL_MAP = {
    r'\byamnet\b': 'yamnet',
    r'\btimesnet\b': 'timesnet',
    r'\bcrnn\b': 'crnn',
    r'\brelation[_-]?net\b': 'relation_net',
    r'\bcnn\b': 'cnn',
}

def log_to_file(log_path, message):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(f"[{ts}] {message}\n")

def read_queue():
    with open(QUEUE_FILE, 'r') as f:
        return json.load(f)

def write_queue(queue):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)

def sh(cmd, check=True, capture_output=True, text=True, cwd=None):
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text, cwd=cwd)

def detect_branch(item: dict) -> str:
    if item.get('git_branch'):
        return str(item['git_branch'])
    repo = item.get('repo_dir')
    if repo and Path(repo).exists():
        out = sh(['git', '-C', repo, 'rev-parse', '--abbrev-ref', 'HEAD'])
        return out.stdout.strip()
    raise ValueError("Branch bilgisi yok. CI 'git_branch' eklemeli veya 'repo_dir' sağlamalısın.")

def branch_to_model_type(branch: str) -> str:
    low = branch.lower()
    for pat, mtype in BRANCH_MODEL_MAP.items():
        if re.search(pat, low):
            return mtype
    raise ValueError(f"Branch adından model tipi çıkarılamadı: '{branch}'")

def extract_params_block(config: dict) -> dict:
    # config içinde ilk *_parameters bloğunu yakala (tek model varsayımıyla sade)
    for k, v in config.items():
        if isinstance(k, str) and k.endswith("_parameters") and isinstance(v, dict):
            return v
    return {}

def get_priority(item: dict) -> int:
    # Öncelik: top-level "priority" → yoksa params içindeki "priority" → yoksa 100
    if "priority" in item and isinstance(item["priority"], int):
        return item["priority"]
    params = extract_params_block(item)
    p = params.get("priority")
    try:
        return int(p)
    except Exception:
        return 100

def pending_eligible(item: dict, now_ts: float) -> bool:
    if item.get("status") != "pending":
        return False
    nt = item.get("next_try_at")
    if nt is None:
        return True
    try:
        return float(nt) <= now_ts
    except Exception:
        return True

def build_command_from_type(model_type: str, config: dict):
    # İlgili param bloğunu bul
    params_key = f"{model_type}_parameters"
    params = config.get(params_key)
    if not isinstance(params, dict) or not params:
        # fallback: tek blok varsa onu kullan
        params = extract_params_block(config)
    if not isinstance(params, dict) or not params:
        raise ValueError("Config içinde parametre bloğu yok veya boş.")

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
            "--patience", str(params.get("patience", 20)),
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

def run_training(command):
    print(f"Training: {' '.join(command)}", flush=True)
    try:
        # Not: blocking çalışıyor; istersen nohup+Popen ile tam paralellik yaparsın.
        subprocess.run(command, check=True)
        return "done"
    except subprocess.CalledProcessError as e:
        print(f"Training error (rc={e.returncode})", flush=True)
        return "error"
    except Exception as e:
        print(f"Training exception: {e}", flush=True)
        return "error"

def main():
    print("Tracker started main loop...", flush=True)

    while True:
        queue = read_queue()
        now_ts = time.time()

        # Kaç running var?
        running_idx = [i for i, it in enumerate(queue) if it.get("status") == "running"]
        running_cnt = len(running_idx)

        # Başlatılabilir pending’leri priority’ye göre sırala
        pending_idxs = [i for i, it in enumerate(queue) if pending_eligible(it, now_ts)]
        pending_sorted = sorted(
            pending_idxs,
            key=lambda i: (get_priority(queue[i]), i)  # düşük öncelik değeri önce
        )

        started_any = False

        # Boş slot kadar job başlatmayı dene
        for i in pending_sorted:
            if running_cnt >= MAX_CONCURRENT:
                break

            # item’i running olarak işaretle
            queue[i]["status"] = "running"
            # Bu turda çözümlenen bilgileri kaydedelim
            try:
                branch = detect_branch(queue[i])
                mtype  = branch_to_model_type(branch)
                queue[i]["_resolved_branch"] = branch
                queue[i]["_resolved_model_type"] = mtype
                write_queue(queue)  # ara durum
                time.sleep(0.5)

                cmd = build_command_from_type(mtype, queue[i])
                # Çalıştır
                result = run_training(cmd)

                # Sonucu yaz
                queue = read_queue()  # taze oku
                if result == "done":
                    queue[i]["status"] = "done"
                    log_to_file(DONE_LOG, f"idx={i} done [branch={branch}]")
                    running_cnt += 0  # bitirdi
                else:
                    # Hata: Eğer başka running süreç(ler) varsa bu muhtemelen OOM olabilir.
                    # Tekrar pending yap ve 5 dk sonra dene.
                    if any(it.get("status") == "running" for it in queue if it is not queue[i]):
                        queue[i]["status"] = "pending"
                        queue[i]["next_try_at"] = time.time() + RETRY_AFTER_SEC
                        log_to_file(ERROR_LOG, f"idx={i} error; reschedule after {RETRY_AFTER_SEC}s (OOM muhtemel)")
                    else:
                        # Running yoksa da error olduysa pending’e al ama hemen tekrar denemeyelim.
                        queue[i]["status"] = "pending"
                        queue[i]["next_try_at"] = time.time() + RETRY_AFTER_SEC
                        log_to_file(ERROR_LOG, f"idx={i} error; no concurrent jobs; reschedule after {RETRY_AFTER_SEC}s")
                write_queue(queue)

                # running_cnt’yi gerçek duruma göre güncelle
                running_cnt = sum(1 for it in queue if it.get("status") == "running")
                started_any = True

            except Exception as e:
                # Komuta bile gelemediysek: pending’e geri al ve gecikmeli dene
                queue = read_queue()
                msg = f"prepare/dispatch error idx={i}: {e}"
                log_to_file(ERROR_LOG, msg)
                try:
                    queue[i]["status"] = "pending"
                    queue[i]["next_try_at"] = time.time() + RETRY_AFTER_SEC
                except Exception:
                    pass
                write_queue(queue)

        if not started_any:
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

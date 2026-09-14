import json
import os
import subprocess
import time
from pathlib import Path

QUEUE_FILE = '/tf/start_training/queue/queue.json'
LOG_DIR = Path('/tf/start_training/queue')
ERROR_LOG = LOG_DIR / 'error.log'
DONE_LOG = LOG_DIR / 'done.log'
CHECK_INTERVAL = 10  # saniye

print("Tracker started. Watching queue.json...", flush=True)

def read_queue():
    with open(QUEUE_FILE, 'r') as f:
        return json.load(f)

def write_queue(queue):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)

def log_to_file(log_path, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def build_command(config):
    if "MODEL_TYPE" not in config:
        raise ValueError("MODEL_TYPE alanı eksik veya yanlış formatta")

    model_type = config.get("MODEL_TYPE")
    
    """if model_type == "cnn":
        params = config.get("cnn_parameters", {})
        if "MODE" not in params:
            raise ValueError("MODE alanı eksik veya yanlış formatta")
            
        mode = params.get("MODE")

        if mode == "retrain":
            return [
                "python", params["model_path"], "retrain",
                "--data_path", params["data_path"],
                "--model_path", params["retrain_model_path"]
            ]
        elif mode == "new_model":
            return [
                "python", params["model_path"], "new_model",
                "--model_path", params["model_path"]
            ]
        elif mode == "new_data":
            return [
                "python", params["model_path"], "new_data_train",
                "--new_data_path", params["data_path"]
            ]
        raise ValueError(f"Invalid CNN mode: {mode}")"""
    
    if model_type == "yamnet":
        params = config.get("yamnet_parameters", {})

        data_path = params["data_path"]
        class_names = params["class_names"]
        dataset_cache_dir = params["dataset_cache_dir"]
        model_name = "YAMNET " + str(params.get("batch_size", 16))
        confidence_dict = params.get("time_series_confidence", {})
        
        return [
            "python", "/tf/start_training/YAMNET/train.py",
            "--data-csv", data_path,
            "--dataset-cache-dir", dataset_cache_dir,
            "--epochs", str(params.get("epochs", 500)),
            "--output-dir", params.get("output_dir", "./output"),
            "--model-name", model_name,
            "--class-names", str(class_names),
            "--batch-size", str(params.get("batch_size", 8)),
            "--early-stopping-patience", str(params.get("early_stopping_patience", 50)),
            "--normalization-method", str(params.get("normalization_method")),
            "--num-of-groups", str(params.get("num_of_groups"))
        ]
    
    elif model_type == "timesnet":
        params = config.get("timesnet_parameters", {})
        # required = ["csv_path", "output_dir", "batch_size", "epochs"]
        # missing = [k for k in required if k not in params]
        # if missing:
        #     raise ValueError(f"TimesNet parametreleri eksik: {missing}")
        cmd = [
        "python", "/tf/start_training/TIMESNET_V2/train_timesnet.py",
        "--csv_path", params["csv_path"],
        "--output_dir", params["output_dir"],
        "--batch", str(params.get("batch_size", 128)),
        "--label_smoothing", str(params.get("label_smoothing", 0.4)),    
        "--epoch", str(params.get("epochs", 200)),
        "--run_name", params.get("run_name", "TimesNet_Run_Auto"),
        "--patience", str(params.get("patience", 20)),
        "--lr", str(params.get("lr", 0.001)),
        "--augmentation_type", str(params.get("augmentation_type", "onthefly")),
        "--loss", "focal" if params.get("loss", "ce") == "focal" else "ce",
        ]
        
        if params.get("cmvn", False):
            cmd.append("--cmvn")
        if params.get("specaug", False):
            cmd.append("--specaug")
        
        return cmd

    elif model_type == "crnn":
        params = config.get("crnn_parameters", {})
        return [
            "python", "/tf/start_training/CRNN/train.py",
            "--batch", str(params["batch_size"]),
            "--epoch", str(params["epochs"]),
        ]
        
    elif model_type == "relation_net":
        params = config.get("relation_net_parameters", {})
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
            "--model_name", params["model_name"]
        ]

    elif model_type == "cnn":
        params = config.get("cnn_parameters", {})
        return [
            "python", "/tf/start_training/CNN/train.py",
            "--data_folder", params["data_folder"],
            "--batch", str(params["batch"]),
            "--epoch", str(params["epoch"]),
            "--patience", str(params["patience"]),
            "--output_shape", str(params["output_shape"]),
            "--classes", str(params["classes"]),
            "--thresholds", str(params["thresholds"]),
            "--model_name", params["model_name"]
        ]

    raise ValueError(f"Unknown model type: {model_type}")
    
def run_training(command):
    print(f"Training started with command: {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
        return "done"
    except subprocess.CalledProcessError as e:
        print("Training process failed", flush=True)
        return "error"

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

                try:
                    command = build_command(queue[idx])
                    result = run_training(command)
                except Exception as e:
                    error_msg = f"Hata: {e} | item: {json.dumps(queue[idx])}"
                    print(error_msg, flush=True)
                    log_to_file(ERROR_LOG, error_msg)
                    result = "error"

                updated_queue = read_queue()
                updated_queue[idx]["status"] = result
                write_queue(updated_queue)

                if result == "done":
                    done_msg = f"Training finished for item: {json.dumps(updated_queue[idx])}"
                    log_to_file(DONE_LOG, done_msg)

                updated = True
                break

        if not updated:
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

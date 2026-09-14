import json
import os
import subprocess
import time

QUEUE_FILE = '/tf/start_training/queue/queue.json'
CHECK_INTERVAL = 10  # saniye
print("Tracker started. Watching queue.json...", flush=True)

def read_queue():
    with open(QUEUE_FILE, 'r') as f:
        return json.load(f)

def write_queue(queue):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)

def build_command(item):
    mode = item.get("MODE")
    
    if mode == "retrain":
        return [
            "python", "/tf/start_training/train.py", "new_retrain",
            "--retrain_data_path", item["DATA_PATH"],
            "--retrain_model_path", item["RETRAIN_MODEL_PATH"]
        ]
    elif mode == "new_model":
        return [
            "python", "/tf/start_training/train.py", "new_model_train",
            "--new_model_path", item["MODEL_PATH"]
        ]
    elif mode == "new_data":
        return [
            "python", "/tf/start_training/train.py", "new_data_train",
            "--new_data_path", item["DATA_PATH"]
        ]
    else:
        raise ValueError(f"Geçersiz MODE: {mode}")

def run_training(command):
    print(f"Training started with command: {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
        return "done"
    except subprocess.CalledProcessError:
        print("Training process failed", flush=True)
        return "error"

def main():
    print("Tracker started...", flush=True)
    while True:
        queue = read_queue()
        updated = False
        print(f"Queue state: {[item.get('status') for item in queue]}", flush=True)
        for item in queue:
            if item.get("status", "pending") == "pending":
                item["status"] = "running"
                write_queue(queue)
                time.sleep(1)

                try:
                    command = build_command(item)
                    result = run_training(command)
                except Exception as e:
                    print(f"Hata: {e}", flush=True)
                    result = "error"

                item["status"] = result
                write_queue(queue)
                updated = True
                break  # sıradakine geçmek için döngüden çık

        if not updated:
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
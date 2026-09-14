# CI-CD Flask Queue Training Tracker

This repository provides a lightweight queue-based training launcher for AI model training jobs.

The idea is simple:

1. A Flask HTTP service accepts a training request and writes it into a JSON queue file.
2. A background tracker process reads that queue file, detects the requested model type from the Git branch, sorts jobs by priority, and launches the matching training command.
3. When a training process completes, the tracker marks the job as `done` or `error` and writes logs.
4. If the project is deployed through Docker, the same service can be started in a container with `supervisord` so both the Flask API and tracker process remain active.

In other words, once you pull this project and start the Docker tracker, you can push a training job to the queue and the tracker automatically detects it, places it at the end of the queue if needed, and starts the corresponding training when a slot is available.

---

## Project Structure

```text
app.py                Flask API that receives training requests and stores them in queue.json
tracker.py            Scheduler and launcher that reads the queue and starts model training
queue.json            Queue storage file for pending, running, done, and error jobs
requirements.txt      Python dependencies for the API
tracker_docker_files/
  Dockerfile          Docker image definition
  docker-compose.yml  Example compose layout for running the tracker container
  supervisord.conf    Starts Flask app and tracker together
logs/                 Runtime logs and tracker outputs
```

---

## How the Flow Works

### 1. Push a job

The Flask app exposes the route:

```http
POST /push-queue
```

The route receives JSON and appends it to `queue.json`.

### 2. Read and sort the queue

The tracker reads the queue file and checks items from `pending` status. It looks at the `git_branch` field or the `repo_dir` location to determine the branch. Then it maps the branch name to a supported model.

Supported branch-to-model mappings:

```text
"yamnet"       -> branch contains yamnet
"timesnet"     -> branch contains timesnet
"crnn"         -> branch contains crnn
"relation_net" -> branch contains relation_net or relation-net
"cnn"          -> branch contains cnn
```

### 3. Build a training command

The tracker converts the selected model type into a command list. Examples:

- `yamnet` -> `python /tf/start_training/YAMNET/train.py ...`
- `timesnet` -> `python /tf/start_training/TIMESNET_V2/train_timesnet.py ...`
- `crnn` -> `python /tf/start_training/CRNN/train.py ...`
- `relation_net` -> `python /tf/start_training/RELATIONNET/train.py ...`
- `cnn` -> `python /tf/start_training/CNN/train.py ...`

### 4. Launch the training process

The scheduler launches the generated command with `subprocess.Popen`. It stores the process id and marks the queue item as `running`.

### 5. Job completion / state update

After launch, the tracker continuously checks the queue and active processes. It updates the item status:

```text
pending    -> job is waiting for execution
running    -> job is actively launching/training
done       -> the training finished successfully
error      -> the training failed
```

The tracker writes logs into:

```text
error.log
```

and

```text
done.log
```

---

## Example Queue Payload

A typical payload sent to `/push-queue` looks like this:

```json
{
  "git_branch": "timesnet",
  "priority": 5,
  "timesnet_parameters": {
    "csv_path": "/tf/start_training/TIMESNET_V2/preproc_5cls_balanced_20k_winsize",
    "epochs": 5,
    "output_dir": "/tf/start_training/TIMESNET_V2/timesnet_outputs/balanced_data_250ep",
    "batch_size": 64,
    "augmentation_type": "onthefly",
    "patience": 12,
    "lr": 0.001,
    "cmvn": true,
    "run_name": "TimesNet_Run_002",
    "seed": 42,
    "label_smoothing": 0,
    "loss": "focal"
  }
}
```

The `priority` value controls the order. Lower values run first.

---

## Local Usage

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Start the Flask queue API

```bash
python app.py
```

The Flask app listens on port 3000:

```text
http://0.0.0.0:3000
```

### Start the tracker scheduler

Open another terminal and run:

```bash
python tracker.py
```

The tracker watches the queue file at:

```text
/tf/start_training/queue/queue.json
```

and will start jobs automatically whenever there is an available launch slot.

---

## Push a Job Example

You can push a training job to the queue with curl:

```bash
curl -X POST http://localhost:3000/push-queue \
  -H "Content-Type: application/json" \
  -d '{
    "git_branch": "timesnet",
    "priority": 5,
    "timesnet_parameters": {
      "csv_path": "/tf/start_training/TIMESNET_V2/preproc_5cls_balanced_20k_winsize",
      "epochs": 5,
      "output_dir": "/tf/start_training/TIMESNET_V2/timesnet_outputs/balanced_data_250ep",
      "batch_size": 64,
      "augmentation_type": "onthefly",
      "patience": 12,
      "lr": 0.001,
      "cmvn": true,
      "run_name": "TimesNet_Run_002",
      "seed": 42,
      "label_smoothing": 0,
      "loss": "focal"
    }
  }'
```

The service responds with:

```json
{
  "message": "Job added to queue",
  "newData": {...}
}
```

---

## Docker Deployment

The repository also contains a container setup for running both the Flask queue API and the tracker in the same deployment environment.

### Docker compose

The compose definition in `tracker_docker_files/docker-compose.yml` configures the container as:

```yaml
services:
  tracker-server-1:
    image: tracker-service:v2
    container_name: tracker-server-1
    build:
      context: /home/fotas/tracker
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - /home/fotas:/home/fotas
      - /home/fotas/fotas-ai/aiworks/ailab:/tf
    working_dir: /tf
    command: ["supervisord", "-c", "/etc/supervisord.conf"]
    restart: unless-stopped
```

The `supervisord.conf` file starts two background programs:

```text
[program:flask_app]
command=python3 /tf/start_training/queue/app.py

[program:tracker]
command=python3 /tf/start_training/queue/tracker.py
```

This means that when the container is started, the queue API and tracker are both launched together. New jobs pushed through the Flask API are persisted in the queue file and then processed by the tracker.

---

## Dockerfile Note

The included `Dockerfile` is based on:

```text
nvidia/cuda:12.5.0-devel-ubuntu22.04
```

It installs Python 3.11 and necessary dependencies such as TensorFlow, MLflow, scikit-learn, pandas, numpy, PyWavelets, seaborn, and Flask. The container is built for GPU-enabled training workloads.

---

## Queue Scheduling Behavior

The tracker has a built-in scheduler with:

```python
MAX_CONCURRENT = 2
CHECK_INTERVAL = 10
BACKOFF_SECONDS = 300
```

This means:

- at most two training jobs can run at the same time;
- the scheduler checks the queue every 10 seconds;
- if a job fails due to an execution or environment error, the queue item may be retried after a backoff delay.

---

## Recommended Workflow

1. Pull or clone this repository.
2. Install the Python dependencies from `requirements.txt`.
3. Build or run the Docker tracker container.
4. Start the Flask queue endpoint and tracker.
5. Send a valid training payload with `git_branch` and model parameters.
6. The tracker will place the job in the queue and automatically launch it when a free execution slot is available.

---

## Important Notes

- The queue file path is hardcoded in the project as `/tf/start_training/queue/queue.json`.
- The tracker determines the model type from the Git branch name.
- If the branch name is not recognized, the job is not launched.
- The training command must match the expected directory structure inside the container or host environment.

---

## Example Project Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start flask app
python app.py

# Start tracker
python tracker.py

# Push queue item
curl -X POST http://localhost:3000/push-queue \
  -H "Content-Type: application/json" \
  -d '{"git_branch":"timesnet","priority":5,"timesnet_parameters":{"epochs":5}}'
```

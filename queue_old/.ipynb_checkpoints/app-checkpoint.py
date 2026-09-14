from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)

DATA_FILE = '/tf/start_training/queue/queue.json'

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump([], f)

@app.route('/push-queue', methods=['POST'])
def push_queue():
    new_data = request.get_json()

    try:
        with open(DATA_FILE, 'r') as f:
            queue = json.load(f)
    except (ValueError, json.JSONDecodeError):
        queue = []

    queue.append(new_data)

    with open(DATA_FILE, 'w') as f:
        json.dump(queue, f, indent=2)


    return jsonify({"message": "Veri basariyla kuyruga eklendi", "newData": new_data}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

initialized = False

@app.route("/reset", methods=["POST"])
def reset():
    global initialized
    initialized = True
    return jsonify({"success": True})

@app.route("/step", methods=["POST"])
def step():
    if not initialized:
        return jsonify({"error": "reset first"}), 400

    action = request.json.get("action", "")
    reward = len(action) * 0.1

    return jsonify({
        "observation": {"echoed_message": action},
        "reward": reward,
        "done": False
    })

@app.route("/")
def home():
    return "Running"
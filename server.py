from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

initialized = False


# =========================
# HEALTH CHECK (IMPORTANT)
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# =========================
# RESET ENDPOINT
# =========================
@app.route("/reset", methods=["POST"])
def reset():
    global initialized
    initialized = True
    return jsonify({"success": True}), 200


# =========================
# STEP ENDPOINT
# =========================
@app.route("/step", methods=["POST"])
def step():
    global initialized

    if not initialized:
        return jsonify({"error": "reset first"}), 400

    try:
        data = request.get_json(force=True)

        if not data or "action" not in data:
            return jsonify({"error": "missing 'action' field"}), 400

        action = str(data.get("action", ""))

        # Reward logic (same as env)
        reward = len(action) * 0.1

        return jsonify({
            "observation": {"echoed_message": action},
            "reward": reward,
            "done": False
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# ROOT (PING)
# =========================
@app.route("/", methods=["GET"])
def home():
    return "Running", 200


# =========================
# RUN SERVER (HF SPACES)
# =========================
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
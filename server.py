"""
╔══════════════════════════════════════════════════════════════════╗
║        DEEPFAKE DETECTOR — Flask + ML Backend                    ║
║        Run: python server.py                                     ║
║        Open: http://127.0.0.1:5000                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import pipeline
from PIL import Image
import io
import os
import time

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Models ────────────────────────────────────────────────────────
MODELS = {
    "ai-detector": {
        "name":        "AI Image Detector",
        "model_id":    "umm-maybe/AI-image-detector",
        "fake_labels": ["artificial", "fake", "ai"],
        "real_labels": ["real", "human", "photo"],
        "weight":      0.5,
    },
    "deepfake-detector": {
        "name":        "EfficientNet-B4 DeepFake",
        "model_id":    "dima806/deepfake_vs_real_image_detection",
        "fake_labels": ["fake"],
        "real_labels": ["real"],
        "weight":      0.3,
    },
    "sdxl-detector": {
        "name":        "SDXL Detector",
        "model_id":    "Organika/sdxl-detector",
        "fake_labels": ["sdxl", "artificial"],
        "real_labels": ["not sdxl", "real", "photo"],
        "weight":      0.2,
    },
}

# Cache loaded pipelines so they don't reload every request
_pipelines = {}

def get_pipeline(model_key):
    if model_key not in _pipelines:
        cfg = MODELS[model_key]
        print(f"  Loading model: {cfg['model_id']} ...")
        _pipelines[model_key] = pipeline(
            task="image-classification",
            model=cfg["model_id"],
            device=0 if torch.cuda.is_available() else -1,
        )
        print(f"  ✓ Loaded: {cfg['name']}")
    return _pipelines[model_key]


def run_model(model_key, image):
    cfg        = MODELS[model_key]
    classifier = get_pipeline(model_key)

    start   = time.time()
    results = classifier(image)
    elapsed = round(time.time() - start, 2)

    fake_score = 0.0
    real_score = 0.0

    for r in results:
        label = r["label"].lower()
        score = r["score"]
        if any(fl in label for fl in cfg["fake_labels"]):
            fake_score += score
        elif any(rl in label for rl in cfg["real_labels"]):
            real_score += score

    if fake_score == 0 and real_score == 0:
        fake_score = real_score = 0.5
    elif fake_score == 0:
        fake_score = 1.0 - real_score
    elif real_score == 0:
        real_score = 1.0 - fake_score

    return {
        "model_name":  cfg["name"],
        "fake_score":  round(fake_score, 4),
        "real_score":  round(real_score, 4),
        "raw":         results,
        "elapsed":     elapsed,
    }


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file  = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Resize for faster inference
    max_px = 512
    w, h   = image.size
    if max(w, h) > max_px:
        scale  = max_px / max(w, h)
        image  = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    model_results = []
    weighted_fake = 0.0
    total_weight  = 0.0

    for key in MODELS:
        try:
            result = run_model(key, image)
            weight = MODELS[key]["weight"]
            weighted_fake += result["fake_score"] * weight
            total_weight  += weight
            model_results.append({
                "name":       result["model_name"],
                "fake_score": result["fake_score"],
                "real_score": result["real_score"],
                "elapsed":    result["elapsed"],
                "weight":     weight,
            })
        except Exception as e:
            model_results.append({
                "name":  MODELS[key]["name"],
                "error": str(e),
            })

    if total_weight == 0:
        return jsonify({"error": "All models failed"}), 500

    final_fake = weighted_fake / total_weight
    final_real = 1.0 - final_fake

    return jsonify({
        "fake_score":     round(final_fake, 4),
        "real_score":     round(final_real, 4),
        "verdict":        "FAKE" if final_fake >= 0.5 else "REAL",
        "model_results":  model_results,
        "device":         "GPU" if torch.cuda.is_available() else "CPU",
    })


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║     DEEPFAKE DETECTOR — Starting server...          ║")
    print("║     Open http://127.0.0.1:5000 in your browser      ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(f"  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("  Models will load on first analysis request...\n")
    app.run(debug=False, port=5000)
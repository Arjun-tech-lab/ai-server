from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
from flask_cors import CORS  # 👈 import this

# -----------------------------
# 1️⃣ Setup Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)  # 👈 allow requests from any frontend

# -----------------------------
# 2️⃣ Load Keras model
# -----------------------------
MODEL_PATH = "pothole_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found!")

model = load_model(MODEL_PATH)

# Classes
classes = ["minor_pothole", "moderate_pothole", "major_pothole"]

# -----------------------------
# 3️⃣ Preprocess image
# -----------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# -----------------------------
# 4️⃣ Prediction endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    img_tensor = preprocess_image(file.read())

    preds = model.predict(img_tensor)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # ✅ Changed key "class" → "clazz" to match Java model
    return jsonify({
        "clazz": classes[class_idx],
        "confidence": confidence
    })

@app.route("/")
def home():
    return "AI server is running!"

# -----------------------------
# 5️⃣ Run server
# -----------------------------
if __name__ == "__main__":
    print("✅ Starting AI server on http://localhost:5002 ...")
    app.run(host="0.0.0.0", port=5002, debug=True)

from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO



BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "classify" / "train" / "weights" / "best.pt"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

DISPLAY_NAMES = {
    "cane": "Dog",
    "cavallo": "Horse",
    "elefante": "Elephant",
    "farfalla": "Butterfly",
    "gallina": "Chicken",
    "gatto": "Cat",
    "mucca": "Cow",
    "pecora": "Sheep",
    "ragno": "Spider",
    "scoiattolo": "Squirrel",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please choose an animal image first."
        elif not allowed_file(uploaded_file.filename):
            error = "Use a JPG, JPEG, PNG, or WEBP image."
        else:
            filename = secure_filename(uploaded_file.filename)
            image_path = UPLOAD_FOLDER / filename
            uploaded_file.save(image_path)

            prediction = model(str(image_path), verbose=False)[0]
            class_id = prediction.probs.top1
            class_name = prediction.names[class_id]
            confidence = float(prediction.probs.top1conf)
            result = {
                "image_url": url_for("static", filename=f"uploads/{filename}"),
                "label": DISPLAY_NAMES.get(class_name, class_name.title()),
                "raw_label": class_name,
                "confidence": confidence,
            }

    return render_template("index.html", result=result, error=error)


@app.errorhandler(413)
def request_entity_too_large(_error):
    return render_template(
        "index.html", result=None, error="That file is too large. Please use an image under 10 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
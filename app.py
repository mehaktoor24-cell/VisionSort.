from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np


app = Flask(__name__)


model = tf.keras.models.load_model("Model/visionsort.h5")


classes = [
    "Metal",
    "Organic",
    "Plastic"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    img = Image.open(image)

    img = img.resize((150,150))

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img)

    result = classes[np.argmax(prediction)]


    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)

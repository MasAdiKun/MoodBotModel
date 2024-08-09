from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import nltk
from nltk.data import find
import json
import random
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
CORS(app)
emo_model = load_model("emorecog.h5")
# Daftar emosi
emotions = ["angry", "happy", "sad"]


def download_nltk_data():
    nltk.data.path.append("/app/nltk_data")
    try:
        find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir="/app/nltk_data")


download_nltk_data()


@app.route("/")
def index():
    return render_template("index.html")


# Preprocessing function
def preprocess_image(image):
    # Mengubah ukuran gambar ke ukuran yang digunakan saat pelatihan
    img_shape = 224
    image = cv2.resize(image, (img_shape, img_shape))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image


@app.route("/api/emopredict", methods=["POST"])
def predict():
    # Periksa apakah file gambar dikirim dalam permintaan
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    # Baca gambar dan konversi menjadi array numpy
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Preprocessing gambar
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    # Prediksi menggunakan model
    prediction = emo_model.predict(img)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    # Tentukan threshold untuk confidence
    threshold = 0.5
    # Mendapatkan emosi yang diprediksi
    emotion = emotions[predicted_index]
    # Menyiapkan respons
    if confidence >= threshold:
        response = {"prediction": emotion, "status": 200}
    else:
        response = {"prediction": emotion, "status": 100}

    return jsonify(response)


# Load trained chatbot model
chatbot_model = load_model("chatbot_v4.5.h5")

# Load intents from JSON file
data_file = open("intent_ver4.json").read()
intents = json.loads(data_file)

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load words and classes from pickle files
words = pickle.load(open("texts.pkl", "rb"))
classes = pickle.load(open("labels.pkl", "rb"))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    if len(return_list) == 0:
        return_list.append({"intent": "default", "probability": "1.0"})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = i["responses"]
            break
    return random.choice(result), tag


@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    message = data["text"]
    ints = predict_class(message, chatbot_model)
    response, tag = get_response(ints, intents)
    return jsonify({"response": response, "tag": tag})


if __name__ == "__main__":
    app.run()

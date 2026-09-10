import os
import string
import sys
import joblib
import nltk

from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "IMDB REVIEWS sentiment_analysis_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "IMDB REVIEWS sentiment_vectorizer.pkl"
)
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
nltk.data.path.insert(0, NLTK_DATA_DIR)

# --------------------------------------------------
# NLTK
# --------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

print("Loading sentiment model...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Flask
# --------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # 16 KB


# --------------------------------------------------
# CORS
# --------------------------------------------------

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://127.0.0.1:5500"
)

CORS(
    app,
    resources={
        r"/analyze_sentiment": {
            "origins": [FRONTEND_ORIGIN]
        }
    }
)


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def remove_punctuation(text):
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )


def remove_stopwords(text):
    words = word_tokenize(text)

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


def lemmatize_text(text):
    words = word_tokenize(text)

    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(lemmatized_words)


def preprocess_text(text):

    text = text.lower()

    text = remove_punctuation(text)

    text = remove_stopwords(text)

    text = lemmatize_text(text)

    return text


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "Sentiment Analysis API"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    })


# --------------------------------------------------
# Sentiment API
# --------------------------------------------------

@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():

    try:

        if not request.is_json:
            return jsonify({
                "error": "Request must contain JSON"
            }), 400


        data = request.get_json()

        comment = data.get("comment")


        if not comment or not isinstance(comment, str):
            return jsonify({
                "error": "Invalid or missing comment"
            }), 400


        comment = comment.strip()


        if len(comment) > 5000:
            return jsonify({
                "error": "Comment is too long. Maximum 5000 characters."
            }), 400


        # Preprocessing

        processed_text = preprocess_text(comment)


        # Vectorization

        features = vectorizer.transform(
            [processed_text]
        )


        # Prediction

        prediction = model.predict(features)[0]


        sentiment = str(prediction).capitalize()
        

        prediction = model.predict(features)[0]
        decision = model.decision_function(features)[0]




        return jsonify({
            "status": "success",
            "sentiment": sentiment,
            "comment_length": len(comment)
        })


    except Exception as e:

        print(
            f"Prediction error: {e}",
            file=sys.stderr
        )

        return jsonify({
            "error": "Internal server error"
        }), 500


# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )

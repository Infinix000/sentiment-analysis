from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import CORS  # Import CORS for Chrome Extension compatibility

# Load the vectorizer and the trained model
with open('vectoriser-ngram-(1,2).pickle', 'rb') as file:
    vectoriser = pickle.load(file)

with open('Sentiment-BNB.pickle', 'rb') as file:
    LRmodel = pickle.load(file)

def predict_sentiment(texts, vectoriser, model):
    # Vectorize the input list of texts
    textdata = vectoriser.transform(texts)
    sentiments = model.predict(textdata)
    
    # Map predictions to labels
    labels = ["Positive" if s == 1 else "Negative" for s in sentiments]
    return labels

app = Flask(__name__)
CORS(app)  # Allow CORS requests

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        comments = data.get('comments', [])
        
        if not comments or not isinstance(comments, list):
            return jsonify({"error": "Invalid input. 'comments' must be a list."}), 400

        predictions = predict_sentiment(comments, vectoriser, LRmodel)
        
        result = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_web():
    text = request.form['Text']
    sentiment = predict_sentiment([text], vectoriser, LRmodel)[0]
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)

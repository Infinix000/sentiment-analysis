from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the vectorizer and the trained model
with open('vectoriser-ngram-(1,2).pickle', 'rb') as file:
    vectoriser = pickle.load(file)

with open('Sentiment-BNB.pickle', 'rb') as file:
    LRmodel = pickle.load(file)

def predict_sentiment(text, vectoriser, model):
    # Vectorize the input text
    textdata = vectoriser.transform([text])  # Input should be in list format for transformation
    # Predict sentiment
    sentiment = model.predict(textdata)
    
    # Convert prediction to human-readable format
    if sentiment[0] == 1:
        return "Positive"
    else:
        return "Negative"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract text data from form
    text = request.form['Text']
    
    # Make prediction
    sentiment = predict_sentiment(text, vectoriser, LRmodel)

    # Render the result on the same page
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)

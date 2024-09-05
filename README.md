Sentiment Analysis with TF-IDF and Machine Learning
Overview
This project implements a sentiment analysis pipeline using various machine learning models and TF-IDF for text feature extraction. The dataset used for training and testing consists of tweets, and the goal is to classify the sentiment of each tweet as either positive or negative.

Features
Data Preprocessing: Handles URL removal, emoji replacement, user mention replacement, and text normalization.
Feature Extraction: Uses TF-IDF with n-grams for transforming text data into numerical features.
Model Training: Implements and evaluates three different models:
Bernoulli Naive Bayes
Linear Support Vector Classification (SVC)
Logistic Regression
Model Evaluation: Provides performance metrics and confusion matrices for the models.
Model Saving and Loading: Save and load models for later use.
Prediction: Predicts sentiment for new text inputs.
Getting Started
Prerequisites
Ensure you have Python 3.x installed along with the following libraries:

numpy
pandas
matplotlib
seaborn
wordcloud
nltk
scikit-learn
pickle
You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn wordcloud nltk scikit-learn
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/your-repo.git
Navigate to the project directory:

bash
Copy code
cd your-repo
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Data Preprocessing

The script preprocesses the data by:

Lowercasing the text
Replacing URLs with the word 'URL'
Replacing emojis with their meanings
Removing non-alphanumeric characters
Reducing consecutive characters
python
Copy code
from preprocessing import preprocess
processedtext = preprocess(text)
Training and Evaluating Models

Models are trained using TF-IDF features and evaluated using performance metrics.

python
Copy code
from model import train_and_evaluate_models

train_and_evaluate_models(X_train, y_train, X_test, y_test)
Saving and Loading Models

Save the trained models and vectorizer for future use:

python
Copy code
from utils import save_models

save_models(vectoriser, LRmodel, BNBmodel)
Load the saved models and vectorizer:

python
Copy code
from utils import load_models

vectoriser, LRmodel = load_models()
Predicting Sentiments

Predict sentiments for new text inputs:

python
Copy code
from utils import predict

text = ["I hate twitter", "May the Force be with you.", "Mr. Stark, I don't feel so good"]
predictions = predict(vectoriser, LRmodel, text)
print(predictions)
Data
The dataset used in this project is a large collection of processed tweets:

File: training.1600000.processed.noemoticon.csv
Columns: sentiment, ids, date, flag, user, text
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Thanks to the scikit-learn library for providing powerful machine learning tools.
Inspiration from various sentiment analysis and text processing tutorials.

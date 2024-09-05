# Sentiment Analysis on Tweets

This project performs sentiment analysis on a large dataset of tweets, using various machine learning algorithms. The goal is to classify the sentiment of a tweet as either positive or negative. This project implements data preprocessing, feature extraction, and builds models using Logistic Regression, Bernoulli Naive Bayes, and Linear SVC.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
- [Results](#results)


## Dataset

The dataset used for this project is the **Sentiment140 dataset** (1.6 million tweets) where:
- 0 = Negative Sentiment
- 1 = Positive Sentiment

It can be found [here](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Requirements

The following Python libraries are used in this project:
- numpy
- pandas
- matplotlib
- seaborn
- wordcloud
- nltk
- sklearn

Install the required packages using the following command:


## Data Preprocessing
### The preprocessing steps include:

- Removing URLs, user mentions, and non-alphabetical characters.
- Replacing emojis with their meaning using a pre-defined dictionary.
- Lemmatization using NLTK’s WordNet Lemmatizer.
- Removal of stopwords.
- The processed data is then split into training and test sets.

## Models
### Three models are used to classify the sentiment:

- Bernoulli Naive Bayes (BNB)
- Linear Support Vector Classifier (Linear SVC)
- Logistic Regression (LR)
- Feature Extraction
The text data is transformed using TF-IDF vectorization, with n-grams ranging from 1 to 2 and a maximum of 500,000 features.

## Results
The performance of each model is evaluated using metrics like precision, recall, and F1-score. Below are the accuracies for the models:

- Bernoulli Naive Bayes: 80%
- Linear SVC: 82%
- Logistic Regression: 83%


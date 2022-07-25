import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier  # RidgeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier  # SelfTrainingClassifier

df = pd.read_csv(
    r'tweet_emotions.csv')
# Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    df.content, df.sentiment, test_size=0.0007000000000000001, random_state=25, shuffle=True)
# NLP system entegration to Data
X_CountVectorizer = CountVectorizer(stop_words='english')

X_train_counts = X_CountVectorizer.fit_transform(X_train)

X_TfidfTransformer = TfidfTransformer()

X_train_tfidf = X_TfidfTransformer.fit_transform(X_train_counts)

# Model Creating
model_semi = SelfTrainingClassifier(RidgeClassifier())

model_semi.fit(X_train_tfidf, y_train)

# Data of Prediction
text = str(input("Enter the sentence : "))

text = [text]

text_counts = X_CountVectorizer.transform(text)

# Prediction Processing
prediction = model_semi.predict(text_counts)

print(prediction[0])
f"Prediction is {prediction[0]}"

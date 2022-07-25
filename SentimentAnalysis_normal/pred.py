import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string


text = open("emotions.txt", encoding="utf-8").read()

lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
tokenized_words = cleaned_text.split()
stop_words = set(stopwords.words('english'))

final_words = []
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)


dict = {}
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(
            ",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        dict[word.strip()] = emotion.strip()


txt = str(input("Enter text: "))

for k, v in dict.items():
    if txt == k:
        print(dict[k])

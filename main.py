
from processingutils import processing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd
import pickle
import numpy as np

TrueTitleText = pd.read_csv('audaxlabs_project_1/True.csv')["title"]
FakeTitleText = pd.read_csv('audaxlabs_project_1/Fake.csv')["title"]

# First vectorize text
categories = [TrueTitleText, FakeTitleText]
text_data = []
x=0
texts=[]
for category in categories:
    for text in category:
        processed = processing(text)
        texts.append(processed)
        print(processed)

tfid =TfidfVectorizer()
vectors = tfid.fit_transform(texts)
labels = []
for x in range(len(TrueTitleText)):
    labels.append(0)
for x in range(len(FakeTitleText)):
    labels.append(1)
x_train,x_test,y_train,y_test = train_test_split(vectors,labels,random_state=42,test_size=20)
neighbors = KNeighborsClassifier(n_neighbors=3)
neighbors.fit(x_train,y_train)
print(accuracy_score(y_test,neighbors.predict(x_test)))
pickle.dump(neighbors,open('model.pickle','wb'))
pickle.dump((x_train,y_train,x_test,y_test),open('data.pickle','wb'))




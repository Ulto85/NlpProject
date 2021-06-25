from processingutils import processing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd
import pickle
import numpy as np
tfid = pickle.load(open('audaxlabs_project_1/vectorizer.pickle','rb'))
texts = [processing('I want 5 burgers')]
vectors= tfid.transform(texts)

kneighbors = pickle.load(open('audaxlabs_project_1/model.pickle','rb'))
_,_,x_test,y_test = pickle.load(open('audaxlabs_project_1/data.pickle','rb'))
print(x_test)
pred  = kneighbors.predict(x_test)
print(pred)
print(y_test)
print(kneighbors.predict(vectors))

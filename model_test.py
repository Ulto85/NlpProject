
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd
import pickle
import numpy as np

kneighbors = pickle.load(open('audaxlabs_project_1/model.pickle'))
_,_,x_test,y_test = pickle.load(open('audaxlabs_project_1/data.pickle'))
print(accuracy_score(y_test,kneighbors.predict(x_test)))
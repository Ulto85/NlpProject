import pandas as pd
from nltk import *
from sklearn.feature_extraction.text import TfidfVectorizer

TrueNews = pd.read_csv('audaxlabs_project_1/True.csv')
titlelist = TrueNews["title"]


def processing(text):
    
    porter_stem = PorterStemmer()
    words = word_tokenize(text)
    stopwords = corpus.stopwords.words('english')
    
    for word in stopwords:
        if word in words:
            words.remove(word)
    for word in words:
        words[words.index(word)] = porter_stem.stem(word)
    
    
    return ' '.join(words)
print(titlelist[0])
print(processing(titlelist[0]))

    

  
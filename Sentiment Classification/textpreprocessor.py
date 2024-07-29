import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import re

class TextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, model="en_core_web_sm"):
        self.model = model
    
    def fit(self, data, y=None):
        return self
    
    def transform(self, data):
        nlp = spacy.load(self.model)
        X = self.preprocess_text(data)
        docs = [nlp(doc) for doc in X]
        return [self._lemmatize(doc) for doc in docs]

    def preprocess_text(self, data):
        corpus = []
        for r in data:
            review = r.lower()
            cleaned = [re.sub("[\W\d]+", '', word) for word in review.split() if not re.search("(.)\1{3,}", word)]
            words = [w for w in cleaned if w.strip()]
            corpus.append(' '.join(words))
        return corpus


    def _lemmatize(self, doc):
        lemma_doc = []
        for token in doc:
            if not token.is_stop and token.lemma_ != '-PRON-':
                new = token.lemma_.lower().strip()
                if new.isalpha():
                    lemma_doc.append(new)
        return ' '.join(lemma_doc)
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

import csv
import pandas as pd
import numpy as np
import re
from data_cleaning import *
from model_selection import *
from TextPreprocessor import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
from matplotlib import pyplot as plt

df1 = pd.read_csv("yelp_review.csv")
df2 = pd.read_csv("yelp_business.csv")

sub_df1 = df1[['review_id', 'business_id', 'stars', 'text']]
sub_df2 = df2[df2.loc[:, 'categories'].str.contains('food|restaruants', flags=re.IGNORECASE, na=False, regex=True)]
restaurant_df = sub_df1[sub_df1.business_id.isin(sub_df2.business_id.unique().tolist())]

text_tuple = wrap_text(restaurant_df.text, restaurant_df.stars)
train_df = convert_to_csv(text_tuple, columns=['review','rating'])

def update_label(data):
    new = []
    for y in data:
        if y == 3:
            new.append('neu')
        elif y < 3:
            new.append('neg')
        else:
            new.append('pos')
    return new   
  
train_df['label'] = pd.Series(update_label(train_df.rating))
X = [[r] for r in train_df.review]
y = [[l] for l in train_df.label]
indices = train_df.index
sample = 10000
x_sample, y_sample = x[:sample], y[:sample]

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size = 0.2, random_state=0, shuffle=True)
models = models = [
    ('SGD Classifer', SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000)),
    ('Linear SVC', LinearSVC(C=0.1, max_iter=1000, dual=False)),
    ('Random Forest', RandomForestClassifier())
]
results = find_best_estimator(models, x_sample, y_sample)

model = SGDClassifier(alpha=1e-4, penalty='elasticnet', fit_intercept=True)
param_grid = {
    'classifier__loss': ['log_loss', 'hinge'],
    'classifier__l1_ratio': stats.uniform(0, 1),
    'classifier__alpha': stats.loguniform(1e-4, 1e0),
    'classifier__max_iter': np.arange(1000, len(X), 100)
}

random_search = model_hyperparamters(model, param_grid, x_sample, y_sample)
best_estimator = random_search.best_estimator_
print(f'best parameters: {random_search.best_params_},  highest accuracy: {random_search.best_score_}')

clf = Pipeline(steps=[
    ('normalize', TextPreprocessor()),
    ('features', TfidfVectorizer(ngram_range=(1,3))),
    ('classifier', best_estimator)
    ])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy: {accuracy}')
plot_predictions(y_pred, 'SGD Classifier')

save_model(clf, 'sentiment_classification_sgd.joblib')




                                               

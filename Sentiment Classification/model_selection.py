from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve
import joblib
import TextPreprocessor


def find_best_estimator(potential_models: List, x_sample, y_sample) -> Dict:
  best_estimator = tuple()
  results = {}
  for i, (model_name, model) in enumerate(potential_models):
      pipe = Pipeline(steps=[
          ('normalize', TextPreprocessor()),
          ('features', TfidfVectorizer(sublinear_tf=True)),
          ('classifier', model)
      ])
      pipe.fit(sample_X_train, sample_y_train)
      y_pred = model.predict(sample_X_test)
      accuracy = accuracy_score(sample_y_test, y_pred)
      results[model_name] = accuracy
  return results

def model_hyperparameters(param_graid: Dict, model, x_sample, y_sample):
  rcv = Pipeline(steps=[
        ('preprocessor', TextPreprocessor()),
        ('features', TfidfVectorizer(lowercase=False, sublinear_tf=True)),
        ('classifier', model)
])
  random_search = RandomizedSearchCV(rcv, param_grid, verbose=3, random_state=0)
  random_search.fit(X_train, y_train)
  return random_search

def plot_predictions(predictions, model_name):
  fig, ax = plt.subplots(figsize=(10, 5))
  ConfusionMatrixDisplay.from_predictions([val[0] for val in y_test], y_pred, ax=ax)
  ax.set_title(f'Confusion Matrix for {model_name}')

def save_trained_model(trained_model, model_name):
  joblib.dump(model, f'model_name.joblib')
  
    
      
      

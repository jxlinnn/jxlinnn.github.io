from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
from matplotlib import pyplot as plt


def find_best_estimator(potential_models: List) -> best_model: Tuple:
  sample_X_train, sample_X_test, sample_y_train, sample_y_test = train_test_split(X_sample, y_sample, test_size = 0.2, random_state=0, shuffle=True)
  
  models = [
      
      ('SGD Classifer', SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000)),
      ('Linear SVC', LinearSVC(C=0.1, max_iter=1000, dual=False)),
      ('Random Forest', RandomForestClassifier()),
      ('KNN', KNeighborsClassifier(n_neighbors=1000))
  ]
  
  fig, axs = plt.subplots()
  
  for i, (model_name, model) in enumerate(models):
      pipe = Pipeline(steps=[
          ('normalize', TextPreprocessor()),
          ('features', TfidfVectorizer(sublinear_tf=True)),
          ('classifier', model)
          ])
  #     tp = TextPreprocessor()
  #     X_tp = tp.fit_transform(sample_X_train)
  #     vect = TfidfVectorizer(sublinear_tf=True)
  #     X_vect = vect.fit_transform(X_tp)
  #     model.fit(X_vect, sample_y_train)
      pipe.fit(sample_X_train, sample_y_train)
      y_pred = model.predict(sample_X_test)
      fpr, tpr, _ = roc_curve(sample_y_test, y_pred, pos_label=4)
      axs.plot(fpr, tpr, label=model_name)
      accuracy = accuracy_score(sample_y_test, y_pred)
      print(f'{model_name} accuracy: {accuracy}')
      
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.legend()

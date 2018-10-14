import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from pdb import set_trace as st

class LRTrainer():
    def train(self, X, y):
        if y.shape[1] != 1:
            y = np.argmax(y, axis = 1)
        print('Grid search cross validation...')
        parameters = {'C':[2**x for x in range(-15,5)]}
        # lr = LogisticRegression(multi_class='multinomial', penalty='l2', 
        #     tol=0.000001, max_iter=1000, solver='lbfgs')
        lr = LogisticRegression(penalty='l2', tol=0.000001, max_iter=1000, solver='lbfgs')
        cv_clf = GridSearchCV(lr, param_grid = parameters)
        cv_clf.fit(X, y)
        best_C = cv_clf.best_params_['C']
        
        print('Running logistic regression...')
        self.clf = LogisticRegression(C=best_C,penalty='l2', tol=0.000001, max_iter=1000, solver='lbfgs')
        self.clf.fit(X, y)

    def predict(self, X_test):
        return self.clf.predict_proba(X_test)

    def prediction_accuracy(self, X_test, y_test):
        prediction = self.clf.predict_proba(X_test)
        accuracy = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(prediction, axis = 1))

        return accuracy
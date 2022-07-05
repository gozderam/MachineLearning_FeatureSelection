from xgboost import XGBClassifier
import numpy as np

class XgBoostWrapper(XGBClassifier):

    def __init__(self, **kwargs):
        XGBClassifier.__init__(self, **kwargs)

    def fit(self, X, y):
        y_tr = np.array([yi for yi in y])  
        y_tr[ y_tr == -1 ] = 0
        XGBClassifier.fit(self, X, y_tr)
        return self

    def predict(self, X):
        yhat = XGBClassifier.predict(self, X)
        yhat[yhat == 0] = -1
        return yhat


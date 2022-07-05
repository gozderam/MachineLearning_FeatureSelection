from typing import Tuple, Callable, Dict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from factories.abstract_factory import AbstractStepFactory
from classification_wrappers.XGB_wrapper import XgBoostWrapper

class ClassifiersFactory(AbstractStepFactory):
    components = {
        # doc: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        'GB': {
            'full_name': 'Gradient Boosting',
            'type': GradientBoostingClassifier(),
            'hyperparams': {
                'n_estimators': [20, 50, 100, 200, 250, 300],
                'learning_rate': [0.001, 0.01, 0.1, 1],
                'max_depth': [2, 3, 4, 5, 6, 7, 8]
            }
        },
        # doc: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        'XGB': {
            'full_name': 'eXtreme Gradient Boosting',
            'type': XgBoostWrapper(),
            'hyperparams': {
                'n_estimators': [20, 50, 100],
                'max_depth': [5, 7, 10, 15, 30]
            }
        },
        # doc: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        'LGBM': {
            'full_name': 'LightGBM classifier',
            'type': LGBMClassifier(),
            'hyperparams': {
                'n_estimators': [40, 70, 100, 200, 250, 300, 500],
                'max_depth': [-1, 5, 7, 10, 25],
                'boosting_type': ['gbdt', 'goss'],
                'learning_rate': [0.1, 0.5]
            }
        },
        # doc: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'RF': {
            'full_name': 'Random Forest',
            'type': RandomForestClassifier(),
            'hyperparams': {
                'n_estimators': [50, 60, 70, 100, 150, 200],
                'max_depth': [None, 10, 20, 25]
            }
        }
    }

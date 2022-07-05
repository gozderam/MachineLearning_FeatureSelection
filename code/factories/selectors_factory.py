from sklearn.linear_model import LogisticRegression
from classification_wrappers.XGB_wrapper import XgBoostWrapper
from boruta import BorutaPy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, chi2, mutual_info_classif
from factories.abstract_factory import AbstractStepFactory
import CONFIG



class SelectorsFactory(AbstractStepFactory):
    components = {
        # doc: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
        f'kbest_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'K Best',
            'type': SelectKBest(),
            'hyperparams': {
                'k': [5, 7, 10, 12],
            }
        },
        f'kbest_chi2_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'K Best chi2',
            'type': SelectKBest(),
            'hyperparams': {
                'k': [5, 7, 10, 12],
                'score_func': [chi2]
            }
        },
        f'variance_thr_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'Variance Threshold',
            'type': VarianceThreshold(),
            'hyperparams': {
                'threshold': [0],
            }
        },
        # doc: https://github.com/scikit-learn-contrib/boruta_py#parameters
        f'boruta_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'Boruta',
            'type': BorutaPy(estimator=None),
            'hyperparams': {
                'estimator': [ RandomForestClassifier() ],
                'n_estimators': ['auto']
            },
        },
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn-ensemble-randomforestclassifier
        f'rf_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'Random Forest Selector',
            'type': SelectFromModel(estimator=None, max_features=55),
            'hyperparams': {
                'estimator': [
                    RandomForestClassifier(n_estimators=250, max_depth=5),
                    RandomForestClassifier(n_estimators=200, max_depth=5),
                    RandomForestClassifier(n_estimators=200, max_depth=7),
                    RandomForestClassifier(n_estimators=150, max_depth=15),
                    RandomForestClassifier(n_estimators=150, max_depth=30),
                    RandomForestClassifier(n_estimators=100, max_depth=5),
                    RandomForestClassifier(n_estimators=50, max_depth=5),
                ],
                'max_features': [5, 7, 10, 12]
            }
        },
        f'lgbm_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'eXtreme Gradient Boost Selector',
            'type': SelectFromModel(estimator=None,  max_features=55),
            'hyperparams': {
                'estimator': [
                    LGBMClassifier(),
                    LGBMClassifier(n_estimators=200, max_depth=20),
                    LGBMClassifier(n_estimators=200, max_depth=10),
                    LGBMClassifier(n_estimators=200, max_depth=7),
                    LGBMClassifier(n_estimators=200, max_depth=5),
                    LGBMClassifier(n_estimators=200, max_depth=15),
                    LGBMClassifier(n_estimators=100, max_depth=20),
                    LGBMClassifier(n_estimators=100, max_depth=7),
                    LGBMClassifier(n_estimators=100, max_depth=3),
                    LGBMClassifier(n_estimators=100, max_depth=4),
                    LGBMClassifier(n_estimators=100, max_depth=5),
                    LGBMClassifier(n_estimators=50, max_depth=30),
                ]
            },
        },
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn-linear-model-logisticregression
        f'lasso_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'Lasso (L1) Logistic Regression Selector',
            'type': SelectFromModel(estimator=None, max_features=55),
            'hyperparams': {
                'estimator': [
                    LogisticRegression(C=0.01, penalty="l1"),
                    LogisticRegression(C=0.1, penalty="l1"),
                    LogisticRegression(C=1, penalty="l1"),
                    LogisticRegression(C=2, penalty="l1"),
                    LogisticRegression(C=3, penalty="l1"),
                    LogisticRegression(C=0.5, penalty="l1"),
                    LogisticRegression(C=7, penalty="l1"),
                    LogisticRegression(C=10, penalty="l1"),
                ]
            }
        },
        f'XGB_{CONFIG.SELECTOR_SUFFIX}': {
            'full_name': 'Xg boost feature selection',
            'type': SelectFromModel(estimator=None),
            'hyperparams': {
                'estimator': [
                    XgBoostWrapper(n_estimators = 10, max_depth = 10),
                    XgBoostWrapper(n_estimators = 20, max_depth = 5),
                    XgBoostWrapper(n_estimators = 50, max_depth = 7),
                    XgBoostWrapper(n_estimators = 100, max_depth = 5),
                    XgBoostWrapper(n_estimators = 100, max_depth = 10),
                    XgBoostWrapper(n_estimators = 100, max_depth = 15),
                    XgBoostWrapper(n_estimators = 100, max_depth = 30),
                    XgBoostWrapper(n_estimators = 150, max_depth = 10),
                    XgBoostWrapper(n_estimators = 200, max_depth = 10),
                    XgBoostWrapper(n_estimators = 250, max_depth = 10),
                    XgBoostWrapper(n_estimators = 250, max_depth = 15),
                    XgBoostWrapper(n_estimators = 100, max_depth = 3),
                    XgBoostWrapper(n_estimators = 100, max_depth = 5),
                ],
                'max_features': [5, 7, 10, 12]
            }
        },
    }

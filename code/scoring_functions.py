import numpy as np
from sklearn.pipeline import Pipeline
import CONFIG

def __balanced_acc(y, y_pred):

    P = len(y[y==1])
    N = len(y[y==-1])
    TP = len(y[(y == 1) & (y_pred == 1)])
    TN = len(y[(y == -1) & (y_pred == -1)])
    
    BA = 0.5 * (TP/P + TN/N)

    return BA


def get_number_of_selected_features(estimator: Pipeline):

    step_names = [step[0] for step in estimator.steps]

    last_feature_selector_name = None
    for step_name in step_names:
        if(step_name[-2:] == CONFIG.SELECTOR_SUFFIX):
            last_feature_selector_name = step_name
        
    # no feature selectors
    if last_feature_selector_name is None:
        print("no feature selection in pipeline")
        return estimator.steps[-1][1].n_features_in_

    last_feature_selector = estimator.steps[step_names.index(last_feature_selector_name)][1]

    if last_feature_selector_name == f"boruta_{CONFIG.SELECTOR_SUFFIX}":
        return last_feature_selector.n_features_

    is_feature_selected =  last_feature_selector.get_support().astype(int)
    m = np.sum(is_feature_selected)

    return m


def __custom_scorer(estimator: Pipeline, X: np.array, y: np.array, p, q):

    y_pred = estimator.predict(X)
    BA = __balanced_acc(y, y_pred)

    m = get_number_of_selected_features(estimator)

    score_m = ( p*m - q ) if ( p*m - q ) >=0 else 0
    score = BA - 0.01 * score_m

    return score


def artificial_scorer(estimator: Pipeline, X, y):
    return __custom_scorer(estimator, X, y, 1/5, 1)


def digits_scorer(estimator: Pipeline, X, y):
    return __custom_scorer(estimator, X, y, 1/200, 0.25)

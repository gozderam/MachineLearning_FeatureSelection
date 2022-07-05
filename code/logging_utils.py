from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import os
import CONFIG
import re
from sklearn.pipeline import Pipeline
import numpy as np
from os import listdir
from os.path import isfile, join
from scoring_functions import get_number_of_selected_features


def get_model_name(model: RandomizedSearchCV, dataset: str) -> str:
    estiamtor = model.best_estimator_

    step_names = [step[0] for step in estiamtor.steps]
    return dataset + "---" + '-'.join(step_names)


def serialize_model(model: RandomizedSearchCV, dataset: str):
    if not os.path.exists(CONFIG.MODELS_BASE_PATH):
        os.makedirs(CONFIG.MODELS_BASE_PATH)
    dump(model, os.path.join(CONFIG.MODELS_BASE_PATH, get_model_name(model, dataset)))


def deserialize_model(model_name: str) -> RandomizedSearchCV:
    return load(os.path.join(CONFIG.MODELS_BASE_PATH, model_name))


def read_results(result_names: list, datasets=["artificial", "digits"]):
    return __read_from_cv_results(result_names, lambda model, result_name: model.cv_results_[result_name], datasets)


def read_best_scores(datasets=["artificial", "digits"]):
    df = __read_from_cv_results(["best_score"], lambda model, _: (model.best_score_, get_number_of_selected_features(model.best_estimator_)), datasets)
    df2 = pd.DataFrame(df['best_score'].tolist(), index=df.index, columns=["best_score", "no of features"])
    df2['dataset'] = df['dataset']
    df2['algorithm'] = df['algorithm']
    df2 = df2[["dataset", "algorithm", "best_score",  "no of features"]].sort_values(by="best_score", ascending=False)
    return df2

def read_best_params(datasets=["artificial", "digits"]):
    return __read_from_cv_results(["best_params"], lambda model, _: model.best_params_, datasets)


def get_indices_of_selected_features(estimator: Pipeline, n_input_features):
    # indices starting with 0
    idx = np.linspace(0, n_input_features-1, n_input_features, dtype="int")
    for step_name, step in estimator.steps:
        if(step_name[-2:] == CONFIG.SELECTOR_SUFFIX):

            selector = step

            # boruta
            if (step_name == f"boruta_{CONFIG.SELECTOR_SUFFIX}"):
                selected_idxs = np.where(selector.support_ == True)
            # sklearn
            else:
                selected_idxs = np.where(selector.get_support() == True)

            idx = idx[selected_idxs]

    return idx


def save_resutls(model: RandomizedSearchCV, dataset: str, X):
    if not os.path.exists(CONFIG.TESTS_BASE_PATH):
        os.makedirs(CONFIG.TESTS_BASE_PATH)

    selected_features = get_indices_of_selected_features(
        model.best_estimator_, X.shape[1])
    oneclass_idx = np.where(model.classes_ == 1)[0][0]
    pred_proba = model.predict_proba(X)[:, oneclass_idx]
    model_name = get_model_name(model, dataset)

    features_path = os.path.join(
        CONFIG.TESTS_BASE_PATH, f"{model_name}_{CONFIG.TEAM_CODE}_{dataset}_features.txt")
    prediciton_path = os.path.join(
        CONFIG.TESTS_BASE_PATH, f"{model_name}_{CONFIG.TEAM_CODE}_{dataset}_prediction.txt")

    np.savetxt(features_path, selected_features.astype(int), fmt='%i')
    np.savetxt(prediciton_path, pred_proba, fmt='%f')


def get_features_counted(my_path, dataset_name, n_input_features):
    counter = np.zeros(n_input_features)
    models_count = 0

    for file in [f for f in listdir(my_path) if isfile(join(my_path, f)) and f.startswith(dataset_name)]:
        cv = deserialize_model(file)
        model = cv.best_estimator_
        idx = get_indices_of_selected_features(model, n_input_features)
        counter[idx] += 1
        models_count += 1

    df = pd.DataFrame(counter)
    df.columns = ['count']
    df = df.sort_values(by='count', ascending=False)
    return df, models_count


def __read_from_cv_results(result_names, read_func, datasets):
    model_names = os.listdir(CONFIG.MODELS_BASE_PATH)
    results = []
    for model_name in model_names:
        if(os.path.isdir(os.path.join(CONFIG.MODELS_BASE_PATH, model_name))):
            continue

        match = re.search('([a-z]*)---([0-9A-Za-z_\-]*)', model_name)
        dataset_name = match.groups()[0]
        algorithm_name = match.groups()[1]

        if dataset_name not in datasets:
            continue

        model = deserialize_model(model_name)

        results_dict = {}
        for result_name in result_names:
            results_dict[result_name] = read_func(model, result_name)

        results.append(
            dict({"dataset": dataset_name, "algorithm": algorithm_name}, **results_dict))

    res_df = pd.DataFrame.from_records(results)
    return res_df

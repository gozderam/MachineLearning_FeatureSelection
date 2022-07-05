from sklearn.model_selection import RandomizedSearchCV
from scoring_functions import artificial_scorer, digits_scorer
from typing import List, Tuple, Dict, Callable
from sklearn.pipeline import Pipeline
import CONFIG 


def tune_model(x_train, y_train, pipe, hyperparams, dataset, random_search_iters=10):
    if dataset == "artificial":
        scoring = artificial_scorer
    elif dataset == "digits":
        scoring = digits_scorer
    else:
        raise Exception(f"Not known dataset: {dataset}")

    rscv = RandomizedSearchCV(pipe, hyperparams, verbose=3, scoring=scoring, n_iter = random_search_iters, n_jobs=CONFIG.N_JOBS)
    rscv.fit(x_train, y_train)

    return rscv


def make_pipe(steps: List[Tuple[str, object, Dict]]) -> Tuple[Pipeline, Dict]:
    """
    Function to create a pipeline with a corresponding hyperparameters dict.
    :param steps: A list of steps, each step is a tuple containing the step name, type and hyperparameters dict.
    :return: A tuple containing the pipeline and its hyperparameters dict.
    """

    pipe = Pipeline(steps=[(step[0], step[1]) for step in steps])
    hyperparams = {}

    for step in steps:
        step_hyperparams = step[2]
        step_name = step[0]
        for h_key, h_val in step_hyperparams.items():
            hyperparams.update({f"{step_name}__{h_key}": h_val})

    return pipe, hyperparams

from typing import Tuple, Callable, Dict
from sklearn.model_selection import ParameterGrid


class AbstractStepFactory:

    components = {
        'step_name': {
            'full_name': 'Step full name',
            'type': 'Callable type',
            'hyperparams': {
                'h1': [1, 2, 3]
            }
        }
    }

    @classmethod
    def get_all_names(cls):
        return list(cls.components.keys())

    @classmethod
    def get(cls, name: str):
        names = cls.get_all_names()
        assert name in names, print(f"name should be in {names}")
        return cls.components[name]

    @classmethod
    def get_tuple_for_pipe(cls, name: str) -> Tuple[str, object, Dict]:
        """
        Returns a given step
        :param name: name of the step
        :return: tuple containing the step name, type and hyperparameters dict
        """

        cls_object = cls.get(name)
        return name, cls_object['type'], cls_object['hyperparams']

    @classmethod
    def get_type(cls, name: str):
        return cls.get(name)['type']

    @classmethod
    def get_hyperparams(cls, name: str):
        return cls.get(name)['hyperparams']

    @classmethod
    def get_full_name(cls, name: str):
        return cls.get(name)['full_name']

    @classmethod
    def get_all(cls):
        return cls.components

    @staticmethod
    def unzip_hyperparams(hyperparams):
        output_hyperparams = {}
        for k, v in hyperparams.items():
            h = []
            for el in v:
                if isinstance(el, dict):
                    args_combination = list(ParameterGrid(el['args']))
                    for c in args_combination:
                        for t in el['type']:
                            h.append(t(**c))
                else:
                    h.append(el)
            output_hyperparams.update({k: h})
        return output_hyperparams

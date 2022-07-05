from sklearn.decomposition import PCA
from factories.abstract_factory import AbstractStepFactory
import CONFIG


class ExtractorsFactory(AbstractStepFactory):
    components = {
        f'pca_{CONFIG.EXTRACTOR_SUFFIX}': {
            'full_name': 'PCA',
            'type': PCA(),
            'hyperparams': {
                'n_components': [0.99],
            }
        }
    }

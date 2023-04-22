import json
import os

import tensorflow as tf
import numpy as np

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

NEAR_ZERO = 1e-6

# TODO: do we need this gpu functions?
def print_gpu_info():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def allow_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if os.environ.get('tf_allow_gpu_growth', 'False') != 'True':
    allow_gpu_growth()
    os.environ['tf_allow_gpu_growth'] = 'True'

def composition_to_array(composition, elements):
    comp_array = np.zeros((len(elements), ), dtype=np.float32)
    for c, v in composition.items():
        comp_array[elements.index(c)] = v
    comp_array /= max(np.sum(comp_array), NEAR_ZERO)
    return comp_array

def load_encoding_model(model_dir):
    model_path = os.path.join(model_dir, 'saved_model')
    model = tf.saved_model.load(model_path)
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as fr:
        model_config = json.load(fr)
    if 'all_ions' in model_config:
        model_config['all_ions'] = [
            tuple(x) for x in model_config['all_ions']
        ]
    return model, model_config
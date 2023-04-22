import json
import os
import pickle

import tensorflow as tf
import numpy as np

from . import utils
from . import model_framework

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def load_framework_model(model_dir):
    cp_path = os.path.join(model_dir, 'saved_model/cp.ckpt')
    config_path = os.path.join(model_dir, 'model_meta.pkl')

    with open(config_path, 'rb') as fr:
        model_meta = pickle.load(fr)
    model_config = model_meta['config']
    model_config['model_path'] = model_dir

    # init model
    model = model_framework.MultiTasksOnRecipes(**model_config)
    batch_size = model_config['batch_size']
    # run on one sample to build model
    zero_composition = np.zeros(
        shape=(len(model_config['all_eles']),),
        dtype=np.float32,
    )
    zero_feature = np.zeros(
        shape=(model_config['mat_feature_len'],),
        dtype=np.float32,
    )
    data_dicts =[
        {
            'reaction_1': [zero_composition]*model_config['max_mats_num'],
            'reaction_2': [zero_composition]*model_config['max_mats_num'],
            'reaction_1_featurized': [zero_feature] * model_config['max_mats_num'],
            'reaction_2_featurized': [zero_feature] * model_config['max_mats_num'],
            'precursors_1_conditional': [zero_composition] * (model_config['max_mats_num']-1),
            'precursors_2_conditional': [zero_composition] * (model_config['max_mats_num']-1),
            'temperature_1': 0.0,
            'temperature_2': 0.0,
            'synthesis_type_1': 'None',
            'synthesis_type_2': 'None',
        }
    ]*batch_size
    data_type, data_shape, padded_data_shape = utils.get_input_format(
        model_type='MultiTasksOnRecipes',
        max_mats_num=model_config['max_mats_num'],
    )
    data_X, data_Y = utils.dict_to_tf_dataset(
        data_dicts,
        data_type,
        data_shape,
        padded_shape=padded_data_shape,
        column_y=None,
        batch_size=batch_size,
    )
    model.fit(
        x=tf.data.Dataset.zip((data_X, data_Y)),
        epochs=1,
    )
    model.load_weights(cp_path)
    return model, model_config






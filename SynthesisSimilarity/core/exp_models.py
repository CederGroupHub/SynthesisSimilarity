import os
import shutil

import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

from .model_framework import MultiTasksOnRecipes

# after experiments, we shall move ReactionTemperature
# to .task_models.py if it is useful
# from .task_models import ReactionTemperature

__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


#################################################
#     ReactionTemperature is an example of task model, please design your task
#     based on this template
#################################################
class ReactionTemperature(keras.Model):
    """
    This is an example of task model, please design your task
    based on this template
    """

    def __init__(self, num_eles, mat_encoder, **kwargs):
        super().__init__(**kwargs)
        self.num_eles = num_eles
        self.mat_encoder = mat_encoder

    def build(self, input_shape):
        """

        :param input_shape: (None, num_eles)
        :return:
        """
        self.alpha = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(1),
            trainable=True,
            name="sim_diff_coeff",
        )

    def call(self, inputs):
        """

        :param inputs: (None, 2, max_mats_num, num_eles)
        :return loss: (None, dim_features)
        """
        # inputs: (None, 2*num_eles+2)
        target_1 = inputs[:, : self.num_eles]
        target_2 = inputs[:, self.num_eles : 2 * self.num_eles]
        T_1 = inputs[:, 2 * self.num_eles]
        T_2 = inputs[:, 2 * self.num_eles + 1]

        # tar_emb_1 (None, latent_dim)
        tar_emb_1, _, _ = self.mat_encoder(target_1)
        tar_emb_2, _, _ = self.mat_encoder(target_2)

        # sim (None, )
        sim_tar = tf.reduce_sum(tar_emb_1 * tar_emb_2, axis=-1)
        sim_T = 1 - tf.abs(T_1 - T_2) / (T_1 + T_2)

        return tf.square(sim_tar - self.alpha * sim_T)


class ExpMultiTasksOnRecipes(MultiTasksOnRecipes):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #################################################
        #     add new tasks here
        #################################################
        task_name = "reaction_T"
        if task_name in self.task_to_add:
            reaction_T = ReactionTemperature(
                num_eles=self.num_eles,
                mat_encoder=self.mat_encoder,
            )
            self.add_task(
                task_name=task_name,
                task_model=reaction_T,
                task_loss=self.get_loss_target_T,
                task_weight=1.0,
            )
            print("reaction_T here")

        print("tasks", self.task_names, self._weight_by_task)

    #################################################
    #     This is an example,  please design your task
    #     based on this template
    #################################################
    def get_loss_target_T(self, inputs, task_name):
        """
        regularize by sim(Tar) - w (|(T_1-T_2|/(T_1+T_2))
        :param inputs: same as call(), which is a dict with keys
                        reaction_1, reaction_1,
                        temperature_1, temperature_2, ...
        :return: (batch_size, ) for each batch data
        """
        # examples
        # to get all materials
        # all_mats: (batch_size*max_mats_num*2, mat_feature_len)
        all_mats_featurized = keras.layers.concatenate(
            [
                tf.reshape(inputs["reaction_1_featurized"], (-1, self.num_eles)),
                tf.reshape(inputs["reaction_2_featurized"], (-1, self.num_eles)),
            ],
            axis=0,
        )
        # to get all reactions
        # all_reactions: (batch_size*2, max_mats_num, mat_feature_len)
        all_reactions_featurized = keras.layers.concatenate(
            [
                inputs["reaction_1_featurized"],
                inputs["reaction_2_featurized"],
            ],
            axis=0,
        )
        # to get all reactions pairs
        # all_pairs: (batch_size, 2, max_mats_num, mat_feature_len)
        all_reaction_pairs_featurizec = tf.stack(
            [
                inputs["reaction_1_featurized"],
                inputs["reaction_2_featurized"],
            ],
            axis=1,
        )
        # to get all target T pairs
        # all_pairs: (batch_size, 2*mat_feature_len+2)
        all_target_T_pairs_featurized = tf.concat(
            [
                inputs["reaction_1_featurized"][:, 0, :],
                inputs["reaction_2_featurized"][:, 0, :],
                tf.reshape(inputs["temperature_1"], (-1, 1)),
                tf.reshape(inputs["temperature_2"], (-1, 1)),
            ],
            axis=1,
        )
        # we can use any one of all_mats, all_reactions,
        # all_reaction_pairs or all_target_T_pairs as the input to
        # a task model, and return the loss
        loss_reaction_T = self._model_by_task[task_name](all_target_T_pairs_featurized)

        ...

        # need to reshape the loss to be consistent with the
        # original shape of inputs, which is (batch_size, )
        ...

        return loss_reaction_T

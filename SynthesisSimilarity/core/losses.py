import tensorflow as tf
from tensorflow import keras
import numpy as np

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


# Custom loss layer
class MultiLossLayer(keras.layers.Layer):
    def __init__(self, task_names, **kwargs):
        super().__init__(**kwargs)
        self.task_names = task_names
        self.num_task = len(self.task_names)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = self.add_weight(
            name='log_vars',
            shape=(self.num_task, ),
            initializer=keras.initializers.Constant(
                np.zeros((self.num_task, ), dtype=np.float32)
            ),
            trainable=True
        )

    def call(self, inputs):
        precision = tf.exp(-self.log_vars)
        multi_loss = tf.reduce_sum(inputs*precision, axis=-1) + \
                     tf.reduce_sum(self.log_vars, axis=-1)
        return multi_loss


class CustomLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        loss = y_pred
        return loss



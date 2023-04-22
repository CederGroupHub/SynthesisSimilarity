import tensorflow as tf
from tensorflow import keras
import numpy as np

from .utils import repeat_in_last_dimension


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


class AddEMBInComposition(keras.layers.Layer):
    """
        add a "EMB" unit in composition vector, just like BERT
        This "EMB" is used as the material embedding because
        it will be used to reconstruct the composition in encoding-
        decoding process

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """

        :param inputs: (None, mat_feature_len)
        :return: (None, 1+mat_feature_len)
        """
        # EMB = tf.ones(tf.concat(tf.shape(inputs)[:-1], [1]) )
        EMB = tf.tile(
            tf.ones((1,1)),
            multiples=[tf.shape(inputs)[0], 1]
        )
        return keras.layers.concatenate([EMB, inputs], axis=-1)


class EmpiricalEmbedding(keras.layers.Layer):
    """
        encode composition vector with empirical element embedding
        the first layer after composition input
        by default it is randomly initialized
        but it is potential to be initialized with elementary
        features
    """
    def __init__(self,
                 mat_feature_len,
                 dim_features,
                 emp_features=None,
                 initializer_max=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.mat_feature_len = mat_feature_len
        self.dim_features = dim_features
        self.emp_features = emp_features
        self.initializer_max = initializer_max

    def build(self, input_shape):
        """
        shape of emp_features: (mat_feature_len, dim_features,)

        :param input_shape: (None, mat_feature_len)
        :return:
        """
        assert input_shape[-1] == self.mat_feature_len
        if self.emp_features == None:
            self.ele_features = self.add_weight(
                shape=(self.mat_feature_len, self.dim_features),
                initializer=keras.initializers.RandomUniform(
                    minval=-self.initializer_max,
                    maxval=self.initializer_max
                ),
                trainable=True,
                name='ele_features'
            )
        else:
            # this is for future use
            # TODO: the features for "EMB" should be trainable,
            #  though those for elements shouldn't
            assert self.emp_features.shape == (
                self.mat_feature_len, self.dim_features
            )
            self.ele_features = self.add_weight(
                shape=(self.mat_feature_len, self.dim_features),
                initializer=keras.initializers.Constant(
                    self.emp_features
                ),
                trainable=False,
                name='ele_features'
            )

    def call(self, inputs):
        """

        :param inputs: (None, mat_feature_len)
        :return _x: (None, mat_feature_len, dim_features)
        """
        # TODO: remove assert after testing
        assert len(inputs.shape) == 2
        # _x = tf.reshape(
        #     tf.tile(
        #         tf.reshape(inputs, (-1, 1)),
        #         multiples=[1, self.dim_features]
        #     ),
        #     (-1, self.mat_feature_len, self.dim_features)
        # )
        _x = repeat_in_last_dimension(
            from_tensor=inputs,
            from_seq_length=self.mat_feature_len,
            to_latent_dim=self.dim_features
        )
        _x = _x* self.ele_features
        return _x


class ZeroShift(keras.layers.Layer):
    """

    """
    def __init__(
        self,
        shift_init_value=-0.5,
        shift_trainable=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shift_init_value = shift_init_value
        self.shift_trainable = shift_trainable

    def build(self, input_shape):
        """
        shape of emp_features: (mat_feature_len, dim_features,)

        :param input_shape: (None, mat_feature_len)
        :return:
        """
        self.shift_bias = self.add_weight(
            shape=(1, ),
            initializer=tf.keras.initializers.Constant(self.shift_init_value),
            trainable=self.shift_trainable,
            name='zero_shift_bias'
        )

    def call(self, inputs):
        """

        :param inputs: (None, mat_feature_len)
        :return _x: (None, mat_feature_len)
        """
        _x = inputs + tf.cast(
            tf.equal(inputs, 0), tf.float32
        ) * self.shift_bias
        return _x


class UnifyVector(keras.layers.Layer):
    """
        make length vector to be 1 along the last dimension
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """

        :param inputs: (None, ..., dim_features)
        :return _x: (None, ..., dim_features)
        """
        # TODO: is 1e-6 ok? utils.NEAR_ZERO
        tensor_norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(inputs),
                axis=-1,
                keepdims=True
            )
        ) + 1e-12
        _x = inputs/tensor_norm
        return _x


class PrecursorsPooling_1(keras.layers.Layer):
    """
        pooling precursor list by reduce mean
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(input, axis=-2)


class Sampling(keras.layers.Layer):
    """
        Uses (z_mean, z_log_var) to sample z,
        the vector encoding a digit.
    """
    def __init__(self,
                 initializer_range,
                 **kwargs):
        super().__init__(**kwargs)
        self.initializer_range = initializer_range

    def call(self, inputs):
        """

        :param inputs: (None, latent_dim)
        :return:
        """
        # TODO: should avoid masks because return is not always
        #  0 for input zero vector
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # TODO: which value of mean and std of random_normal to
        #  pick for initialization?
        epsilon = tf.keras.backend.random_normal(
            shape=(batch, dim),
            stddev=self.initializer_range,
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


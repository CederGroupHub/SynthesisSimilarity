import tensorflow as tf
from tensorflow import keras
import numpy as np

from . import tf_utils
from .utils import get_mat_mask_in_mat_seq
from .layers import UnifyVector
from .layers import Sampling
from .layers import ZeroShift

__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


class MaterialEncoder(keras.Model):
    def __init__(
        self,
        mat_feature_len,
        dim_features,
        latent_dim,
        zero_shift_init_value,
        zero_shift_trainable,
        num_attention_layers,
        num_attention_heads,
        hidden_activation,
        hidden_dropout,
        attention_dropout,
        initializer_range,
        normalize_output=True,
        mask_zero=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mat_feature_len = mat_feature_len
        self.dim_features = dim_features
        self.latent_dim = latent_dim
        self.zero_shift_init_value = zero_shift_init_value
        self.zero_shift_trainable = zero_shift_trainable
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.mask_zero = mask_zero
        self.normalize_output = normalize_output

        if self.hidden_activation in {
            "gelu",
        }:
            self.hidden_activation = tf_utils.get_activation(self.hidden_activation)

        self.zero_shift_layer = ZeroShift(
            shift_init_value=self.zero_shift_init_value,
            shift_trainable=self.zero_shift_trainable,
        )
        self.dens_1 = keras.layers.Dense(
            self.dim_features,
            activation=self.hidden_activation,
        )
        self.hidden_layers = [
            keras.layers.Dense(
                self.latent_dim,
                activation=self.hidden_activation,
            )
            for _ in range(self.num_attention_layers)
        ]

        self.uni_vec = UnifyVector()

        # # by default linear activation is used for Dense
        # self.dense_mean = keras.layers.Dense(self.latent_dim)
        # self.dense_log_var = keras.layers.Dense(self.latent_dim)
        # self.sampling = Sampling(initializer_range=self.initializer_range)

    def call(self, inputs, mask=None):
        """

        :param inputs: (None, mat_feature_len)
        :param mask: (None, ) mask of materials.
                    list of True or False
                    True represents the materials is to be calculated
                    False represents the return is zero by default
        :return z_mean: (None, dim_features)
                z_log_var: (None, dim_features)
                z: (None, dim_features)
        """
        # inputs: (None, mat_feature_len)
        input_shape = tf.shape(inputs)
        if mask is None:
            mask = self.compute_mask(inputs)
        # next: (None, mat_feature_len)
        if mask is not None:
            processed_inputs = inputs[mask]
        else:
            processed_inputs = inputs
        # next: (None, mat_feature_len)
        _x = self.zero_shift_layer(processed_inputs)
        # next: (None, dim_features)
        _x = self.dens_1(_x)
        # more layers here
        # TODO: is dropout useful here?
        # https://github.com/tensorflow/tensor2tensor/blob/3aca2ab360271a4684ffa7ac8767995e264cc558/tensor2tensor/models/transformer.py#L95
        # next: (None, mat_feature_len+1, latent_dim)
        ...
        # next: (None, latent_dim)
        for tmp_layer in self.hidden_layers:
            _x = tmp_layer(_x)

        emb = _x

        if self.normalize_output:
            emb = self.uni_vec(emb)

        if mask is not None:
            effective_indices = tf.range(input_shape[0])[mask]
            output_shape = tf.concat((input_shape[:1], tf.shape(emb)[1:]), axis=0)
            emb = tf.scatter_nd(
                indices=tf.expand_dims(effective_indices, 1),
                updates=emb,
                shape=output_shape,
            )
        return emb, emb, emb

    def compute_mask(self, inputs, mask=None):
        """
        masked position is False.
        True represents a valid material

        :param inputs: (None, mat_feature_len)
        :param mask: not used
        :return mat_mask: (None, )
        """
        if not self.mask_zero:
            return None
        return get_mat_mask_in_mat_seq(inputs)

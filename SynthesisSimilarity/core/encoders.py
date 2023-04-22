import tensorflow as tf
from tensorflow import keras
import numpy as np

# from .attention_layer import SelfAttention
from .bert_modeling import TransformerBlock
from .bert_modeling import create_attention_mask_from_input_mask
from . import tf_utils
from .utils import get_composition_string
from .utils import get_mat_mask_in_mat_seq
from .layers import AddEMBInComposition
from .layers import EmpiricalEmbedding
from .layers import UnifyVector
from .layers import Sampling
from .layers import ZeroShift

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

class MaterialEncoder_2(keras.Model):
    def __init__(self,
                 mat_feature_len,
                 dim_features,
                 latent_dim,
                 num_attention_layers,
                 num_attention_heads,
                 hidden_activation,
                 hidden_dropout,
                 attention_dropout,
                 initializer_range,
                 ele_emb_init_max=10,
                 normalize_output=True,
                 mask_zero=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.mat_feature_len = mat_feature_len
        self.dim_features = dim_features
        self.latent_dim = latent_dim
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.ele_emb_init_max = ele_emb_init_max
        self.mask_zero = mask_zero
        self.normalize_output = normalize_output

        self.add_EMB = AddEMBInComposition()
        self.emp_emb = EmpiricalEmbedding(
            mat_feature_len=self.mat_feature_len,
            dim_features=self.dim_features,
            emp_features=None,
            initializer_max=self.ele_emb_init_max,
        )

        # ref: a bert model coded with TF 2.0 style
        # https://github.com/tensorflow/models/blob/
        # 49ba237d35d2a049be7bede596f4b29fd85cfe28/official/nlp/bert_modeling.py
        self.atten_layers = [TransformerBlock(
            hidden_size=self.dim_features,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.dim_features,
            intermediate_activation=self.hidden_activation,
            hidden_dropout_prob=self.hidden_dropout,
            attention_probs_dropout_prob=self.attention_dropout,
            initializer_range=self.initializer_range,
        ) for _ in range(self.num_attention_layers)]

        self.uni_vec = UnifyVector()

        # by default linear activation is used for Dense
        self.dense_mean = keras.layers.Dense(self.latent_dim)
        self.dense_log_var = keras.layers.Dense(self.latent_dim)
        self.sampling = Sampling(initializer_range=self.initializer_range)

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
        # TDOO: is tf.larger or tf.not_equal better?
        # ele mask in 2d: (None, mat_feature_len)
        mask_2d = tf.cast(tf.not_equal(processed_inputs, 0), tf.float32)

        # next: (None, mat_feature_len, dim_features)
        # currently dim_features=latent_dim
        _x = self.emp_emb(processed_inputs)
        # more layers here
        # TODO: is dropout useful here?
        # TODO: but currently training parameter is not use to separate dropout
        # https://github.com/tensorflow/tensor2tensor/blob/3aca2ab360271a4684ffa7ac8767995e264cc558/tensor2tensor/models/transformer.py#L95
        # next: (None, mat_feature_len+1, latent_dim)
        ...

        # ele_weights in 3d: (None, mat_feature_len, latent_dim)
        ele_weights = tf.expand_dims(processed_inputs, 2)
        # ele_mask in 3d: (None, mat_feature_len, latent_dim)
        ele_mask = tf.expand_dims(mask_2d ,2)

        # attention_mask in 3d: (None, mat_feature_len, mat_feature_len)
        attention_mask = create_attention_mask_from_input_mask(_x, mask_2d)
        for atten in self.atten_layers:
            _x = atten(input_tensor=_x, attention_mask=attention_mask)
            _x = _x*ele_mask

        # next: (None, dim_features)
        emb = tf.reduce_mean(_x, axis=1)
        if self.normalize_output:
            emb = self.uni_vec(emb)

        if mask is not None:
            effective_indices = tf.range(input_shape[0])[mask]
            output_shape = tf.concat(
                (input_shape[:1], tf.shape(emb)[1:]),
                axis=0
            )
            emb = tf.scatter_nd(
                indices=tf.expand_dims(effective_indices, 1),
                updates=emb,
                shape=output_shape,
            )

        # z_mean = self.dense_mean(emb)
        # z_log_var = self.dense_log_var(emb)
        # z = self.sampling((z_mean, z_log_var))
        # return z_mean, z_log_var, z
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
        mats = tf.reduce_sum(inputs, axis=-1)
        mat_mask = tf.not_equal(mats, 0)
        # mat_mask = tf.cast(tf.not_equal(mats, 0), tf.float32)
        return mat_mask

class MaterialEncoder(keras.Model):
    def __init__(self,
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
                 **kwargs):
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

        if self.hidden_activation in {'gelu', }:
            self.hidden_activation = tf_utils.get_activation(self.hidden_activation)

        self.zero_shift_layer = ZeroShift(
            shift_init_value=self.zero_shift_init_value,
            shift_trainable=self.zero_shift_trainable,
        )
        self.dens_1 = keras.layers.Dense(
            self.dim_features,
            activation=self.hidden_activation,
        )
        self.hidden_layers = [keras.layers.Dense(
            self.latent_dim,
            activation=self.hidden_activation,
        ) for _ in range(self.num_attention_layers)]

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
            output_shape = tf.concat(
                (input_shape[:1], tf.shape(emb)[1:]),
                axis=0
            )
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


class MaterialEncoder_3(keras.Model):
    def __init__(self,
                 zero_shift_init_value,
                 zero_shift_trainable,
                 mask_zero=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.zero_shift_init_value = zero_shift_init_value
        self.zero_shift_trainable = zero_shift_trainable
        self.zero_shift_layer = ZeroShift(
            shift_init_value=self.zero_shift_init_value,
            shift_trainable=self.zero_shift_trainable,
        )

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
        _x = processed_inputs
        # next: (None, mat_feature_len)
        _x = self.zero_shift_layer(_x)
        emb = _x
        if mask is not None:
            effective_indices = tf.range(input_shape[0])[mask]
            output_shape = tf.concat(
                (input_shape[:1], tf.shape(emb)[1:]),
                axis=0
            )
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
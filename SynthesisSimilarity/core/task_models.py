import tensorflow as tf
from tensorflow import keras
import numpy as np

from .layers import UnifyVector
from .bert_modeling import TransformerBlock
from .bert_modeling import create_attention_mask_from_input_mask
from . import tf_utils
from .utils import get_mat_mask_in_mat_seq
from .utils import get_mat_pairs_in_reaction
from .utils import get_combination_pairs
from .utils import get_mat_label
from .utils import NEAR_ZERO
from .utils import array_to_formula
from .focal_loss import BinaryFocalLoss
from .circle_loss import CircleLoss

__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


class MaterialDecoder(keras.Model):
    def __init__(
        self,
        mat_feature_len,
        num_eles,
        num_train_reactions,
        latent_dim,
        mat_encoder,
        intermediate_dim,
        final_activation=None,
        loss_fn="mse",
        mask_zero=True,
        stoi_scale=1.0,
        bias_in_element_layer=True,
        constrain_element_layer=False,
        norm_in_element_projection=False,
        dot_prod_scale=1.0,
        balance_PN=True,
        target_ele_labels=None,
        target_ele_counts=None,
        clip_logits=False,
        focal_gamma=2.0,
        focal_alpha=0.25,
        focal_label_smoothing=0.0,
        circle_gamma=64,
        circle_margin=0.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mat_feature_len = mat_feature_len
        self.num_eles = num_eles
        self.num_train_reactions = num_train_reactions
        self.latent_dim = latent_dim
        self.mat_encoder = mat_encoder
        self.intermediate_dim = intermediate_dim
        self.final_activation = final_activation
        if self.final_activation in {
            "gelu",
        }:
            self.final_activation = tf_utils.get_activation(self.final_activation)
        self._all_loss = {
            "mse": self._loss_mse,
            "mae": self._loss_mae,
            "huber": self._loss_huber,
            "mrse": self._loss_mrse,
            "mse_square": self._loss_mse_square,
            "mse_crossing": self._loss_mse_crossing,
            "error_rate": self._loss_error_rate,
            "log_error_rate": self._loss_log_error_rate,
            "cross_entropy": self._loss_cross_entropy,
            "one_hot_mse": self._loss_one_hot_mse,
            "focal": self._loss_focal,
            "circle": self._loss_circle,
        }
        self.loss_fn_name = loss_fn
        if "focal" in self.loss_fn_name:
            self._focal_loss_func = BinaryFocalLoss(
                gamma=focal_gamma,
                pos_weight=focal_alpha,
                label_smoothing=focal_label_smoothing,
                from_logits=True,
            )
        elif "circle" in self.loss_fn_name:
            self._circle_loss_func = CircleLoss(
                gamma=circle_gamma,
                margin=circle_margin,
            )
        elif "mae" in self.loss_fn_name:
            self._mae_loss_func = keras.losses.MeanAbsoluteError(
                reduction=keras.losses.Reduction.NONE,
            )
        elif "huber" in self.loss_fn_name:
            self._huber_loss_func = keras.losses.Huber(
                reduction=keras.losses.Reduction.NONE,
            )

        self._loss_fn = self._all_loss[self.loss_fn_name]
        self.dot_prod_scale = dot_prod_scale
        self.mask_zero = mask_zero
        self.stoi_scale = stoi_scale
        self.balance_PN = balance_PN
        self.target_ele_labels = target_ele_labels
        if self.target_ele_labels is not None:
            assert len(self.target_ele_labels) == self.num_eles
            self.target_ele_labels = tf.constant(self.target_ele_labels)
        else:
            self.target_ele_labels = tf.constant(
                ["DEFAULT_{}".format(i) for i in range(self.num_eles)]
            )
        self.target_ele_counts = target_ele_counts
        if self.target_ele_counts is not None:
            assert len(self.target_ele_counts) == len(self.target_ele_labels)
            self.target_ele_mask = self.target_ele_counts > 0
            self.target_ele_labels = self.target_ele_labels[self.target_ele_mask]
            self.target_ele_counts = self.target_ele_counts[self.target_ele_mask]
            self.element_frequency = tf.constant(
                np.array(self.target_ele_counts).astype(np.float32)
                / self.num_train_reactions
            )
            self.balance_coeff = self.element_frequency / (1 - self.element_frequency)
        else:
            self.target_ele_mask = np.ones((len(self.target_ele_labels),)).astype(
                np.bool
            )
            self.balance_coeff = np.ones((len(self.target_ele_labels),))

        self.bias_in_element_layer = bias_in_element_layer
        self.constrain_element_layer = constrain_element_layer
        self.norm_in_element_projection = norm_in_element_projection
        self.clip_logits = clip_logits

        # TODO: maybe this is useful
        # self.dense_proj = keras.layers.Dense(
        #     self.intermediate_dim,
        #     activation='relu'
        # )

        # TODO: should we use kernel initializer or not
        if self.constrain_element_layer:
            self.element_layer = keras.layers.Dense(
                len(self.target_ele_labels),
                activation=self.final_activation,
                use_bias=self.bias_in_element_layer,
                name="element_layer",
                kernel_constraint=keras.constraints.UnitNorm(axis=0),
            )
        else:
            self.element_layer = keras.layers.Dense(
                len(self.target_ele_labels),
                activation=self.final_activation,
                use_bias=self.bias_in_element_layer,
                name="element_layer",
            )

    def call(self, inputs, mask=None, return_probability=False):
        """

        :param inputs: (None, latent_dim)
        :param mask: (None, ) mask of materials.
                    list of True or False
                    True represents the materials is to be calculated
                    False represents the return is zero by default
        :return: (None, num_eles)
        """
        # TODO: remove assert after testing
        assert inputs.shape[-1] == self.latent_dim
        input_shape = tf.shape(inputs)
        if mask is None:
            mask = self.compute_mask(inputs)
        # next: (None, num_eles)
        if mask is not None:
            processed_inputs = inputs[mask]
        else:
            processed_inputs = inputs

        _x = processed_inputs
        # _x = self.dense_proj(_x)
        # _x = inputs
        _x = self.element_layer(_x)

        if mask is not None:
            effective_indices = tf.range(input_shape[0])[mask]
            output_shape = tf.concat((input_shape[:1], tf.shape(_x)[1:]), axis=0)
            _x = tf.scatter_nd(
                indices=tf.expand_dims(effective_indices, 1),
                updates=_x,
                shape=output_shape,
            )

        if self.norm_in_element_projection:
            # self.precursor_layer.kernel: (latent_dim, num_precursors, )
            # kernel_norm: (num_precursors, )
            kernel_norm = (
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(tf.transpose(self.element_layer.kernel)),
                        axis=-1,
                    )
                )
                + 1e-12
            )
            _x = _x / kernel_norm
        # _x: (None, num_eles)
        _x = _x * self.dot_prod_scale

        if self.clip_logits:
            _x = tf.clip_by_value(_x, clip_value_min=-10.0, clip_value_max=10.0)

        if return_probability:
            # _x: (None, num_eles)
            _x = tf.sigmoid(_x)

        return _x

    def compute_mask(self, inputs, mask=None):
        """
        masked position is False.
        True represents a valid material

        :param inputs: (None, latent_dim)
        :param mask: not used
        :return mat_mask: (None, )
        """
        if not self.mask_zero:
            return None
        return get_mat_mask_in_mat_seq(inputs)

    def _loss_mse(self, materials, materials_featurized):
        """
        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)
        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)
        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask)
        # loss_mat: (None, )
        loss_mat = tf.reduce_sum(
            tf.math.squared_difference(
                tf.boolean_mask(materials, self.target_ele_mask, axis=1)
                * self.stoi_scale,
                x_mat,
            ),
            -1,
        )
        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_mae(self, materials, materials_featurized):
        """
        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)
        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)
        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask)
        # loss_mat: (None, )
        loss_mat = self._mae_loss_func(
            tf.boolean_mask(materials, self.target_ele_mask, axis=1) * self.stoi_scale,
            x_mat,
        )
        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_huber(self, materials, materials_featurized):
        """
        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)
        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)
        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask)
        # loss_mat: (None, )
        loss_mat = self._huber_loss_func(
            tf.boolean_mask(materials, self.target_ele_mask, axis=1) * self.stoi_scale,
            x_mat,
        )
        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_one_hot_mse(self, materials, materials_featurized):
        """
        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)
        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)
        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask, return_probability=True)

        # y_real: (None, num_eles)
        y_real = tf.cast(tf.greater(materials, 0), tf.float32)
        y_real = tf.boolean_mask(y_real, self.target_ele_mask, axis=1)

        # loss_mat: (None, )
        loss_mat = tf.reduce_sum(tf.math.squared_difference(y_real, x_mat), -1)
        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_mrse(self, materials, materials_featurized):
        """
        sqrt(MSE)
        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        loss_mat = self._loss_mse(materials, materials_featurized)
        loss_mat = tf.sqrt(loss_mat)
        return loss_mat

    def _loss_mse_square(self, materials, materials_featurized):
        """
         ((\sum x_i^2)^2 / 2 = (MSE)^2 / 2
        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        loss_mat = self._loss_mse(materials, materials_featurized)
        loss_mat = tf.pow(loss_mat, 2) / 2.0
        return loss_mat

    def _loss_mse_crossing(self, materials, materials_featurized):
        """
         \sum_{i \neq j} (x_i^2 x_j^2) = ((\sum x_i^2)^2 - (\sum x_i^4))/2
        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)
        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)
        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask)
        # loss_mat: (None, num_eles)
        loss_mat = tf.math.squared_difference(
            tf.boolean_mask(materials, self.target_ele_mask, axis=1), x_mat
        )
        # loss_mat: (None, )
        loss_mat = tf.pow(tf.reduce_sum(loss_mat, -1), 2) - tf.reduce_sum(
            tf.pow(loss_mat, 2), -1
        )
        loss_mat = loss_mat / 2.0
        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_error_rate(self, materials, materials_featurized):
        """
        0/1 label of elements, here decoder should output sigmoid logits
        max(correct_rate) = max( \sum E[1[y_i&y_j]]/(n(n-1)/2))

        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # TODO: might also able to use single-label to calculate
        #  precision/F-measure 10.1007/s10994-012-5285-8
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)
        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)
        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask)

        # y_real: (None, num_eles)
        y_real = tf.cast(tf.greater(materials, 0), tf.float32)
        y_real = tf.boolean_mask(y_real, self.target_ele_mask, axis=1)
        # y_correctness: (None, num_eles)
        y_error = tf.abs(x_mat - y_real)
        # loss_mat: (None, )
        y_error = tf.pow(tf.reduce_sum(y_error, -1), 2) - tf.reduce_sum(
            tf.pow(y_error, 2), -1
        )
        y_error = y_error / 2.0
        # loss_mat: (None, )
        loss_mat = y_error
        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_log_error_rate(self, materials, materials_featurized):
        """
        0/1 label of elements, here decoder should output sigmoid logits
        max(correct_rate) = max( \sum E[1[y_i&y_j]]/(n(n-1)/2))

        recover composition of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (batch_size, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)
        # y_correctness: (None, )
        error_rate = self._loss_error_rate(materials, materials_featurized)
        mask = tf.cast(mask, tf.float32)
        # y_correctness: (None, )
        error_rate = error_rate + (1.0 - mask) * NEAR_ZERO
        # loss_mat: (None, )
        loss_mat = tf.math.log(error_rate)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_cross_entropy(self, materials, materials_featurized):
        """
        0/1 label of elements, here decoder should output sigmoid logits

        recover elements of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (None, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)

        # y_real: (None, num_eles)
        y_real = tf.cast(tf.greater(materials, 0), tf.float32)
        y_real = tf.boolean_mask(y_real, self.target_ele_mask, axis=1)

        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)

        # loss_mat: (None, num_eles)
        if self.balance_PN:
            # x_mat: (None, num_eles)
            x_mat = self.__call__(x_mat, mask=mask, return_probability=True)
            loss_mat = (
                y_real * (-tf.math.log(x_mat))
                + (1 - y_real) * (-tf.math.log(1 - x_mat)) * self.balance_coeff
            )
        else:
            # x_mat: (None, num_eles)
            x_mat = self.__call__(x_mat, mask=mask, return_probability=True)
            loss_mat = y_real * (-tf.math.log(x_mat)) + (1 - y_real) * (
                -tf.math.log(1 - x_mat)
            )
        # loss_mat: (None, )
        loss_mat = tf.reduce_sum(loss_mat, axis=-1)

        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_focal(self, materials, materials_featurized):
        """
        0/1 label of elements, here decoder should output sigmoid logits

        recover elements of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (None, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)

        # y_real: (None, num_eles)
        y_real = tf.cast(tf.greater(materials, 0), tf.float32)
        y_real = tf.boolean_mask(y_real, self.target_ele_mask, axis=1)

        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)

        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask, return_probability=False)
        # loss_mat: (None, num_eles)
        loss_mat = self._focal_loss_func.call(y_true=y_real, y_pred=x_mat)

        # loss_mat: (None, )
        loss_mat = tf.reduce_sum(loss_mat, axis=-1)

        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def _loss_circle(self, materials, materials_featurized):
        """
        0/1 label of elements, here decoder should output sigmoid logits

        recover elements of materials
        single material embedding
        :param materials: (None, num_eles)
        :param materials_featurized: (None, mat_feature_len)
        :return: (None, ) for each batch data
        """
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(materials_featurized)

        # y_real: (None, num_eles)
        y_real = tf.cast(tf.greater(materials, 0), tf.float32)
        y_real = tf.boolean_mask(y_real, self.target_ele_mask, axis=1)

        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(materials_featurized)

        # x_mat: (None, num_eles)
        x_mat = self.__call__(x_mat, mask=mask, return_probability=False)
        # loss_mat: (None, )
        loss_mat = self._circle_loss_func.call(y_true=y_real, y_pred=x_mat)

        mask = tf.cast(mask, tf.float32)
        # loss_mat: (None, )
        loss_mat = loss_mat * mask
        return loss_mat

    def predict_elements(self, targets, output_thresh=0.5):
        """

        :param targets: (None, mat_feature_len)
        :return all_ele_lists: list of list of (element, score)
        """
        all_ele_lists = []
        # inputs: (None, mat_feature_len)
        # mask: (None, )
        # though inputs is (None, mat_feature_len) rather than (None, latent_dim)
        # we can still use compute_mask() because the size of last
        # dimension does not matter
        mask = self.compute_mask(targets)
        # x_mat: (None, latent_dim)
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(targets)
        # y_pred: (None, num_eles)
        y_pred = self.__call__(x_mat, mask=mask)

        for a_y in y_pred:
            ele_list_pred = (
                self.target_ele_labels[a_y > output_thresh].numpy().astype("U")
            )
            ele_score_pred = a_y[a_y > output_thresh].numpy()
            all_ele_lists.append(
                [
                    {
                        "element": ele,
                        "score": score,
                    }
                    for (ele, score) in zip(ele_list_pred, ele_score_pred)
                ]
            )
        return all_ele_lists


class PrecursorsPredict(keras.Model):
    def __init__(
        self,
        mat_feature_len,
        max_mats_num,
        latent_dim,
        mat_encoder,
        precursor_labels,
        precursor_compositions,
        precursor_counts,
        num_train_reactions,
        batch_size,
        num_reserved_ids=10,
        constrain_precursor_layer=False,
        bias_in_precursor_layer=False,
        predict_precursor_under_mask=True,
        attention_num_heads=1,
        attention_hidden_activation="gelu",
        attention_hidden_dropout=0.1,
        attention_dropout=0.1,
        attention_initializer_range=0.02,
        loss_fn="cross_entropy",
        kernel_initializer="glorot_uniform",
        initializer_max=0.05,
        regularization_lambda=1.0,
        dot_prod_scale=1.0,
        balance_PN=True,
        clip_logits=True,
        focal_gamma=2.0,
        focal_alpha=0.25,
        focal_label_smoothing=0.0,
        circle_gamma=64,
        circle_margin=0.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mat_feature_len = mat_feature_len
        self.max_mats_num = max_mats_num
        self.latent_dim = latent_dim
        self.mat_encoder = mat_encoder
        # precursor_labels is a tf lookup of {mat_str: mat_label}
        # label is the index in mat_counts
        # the real index starts from num_reserved_ids
        # the first num_reserved_ids indices are for
        # reserved symbols such as <UNK> <MASK>
        # each label is int64
        self.precursor_labels = precursor_labels
        # mat_compositions is tf list of compositions, index is mat_label
        self.precursor_compositions = precursor_compositions
        # mat_counts is python list of int, index is mat_label
        self.precursor_counts = precursor_counts
        self.num_train_reactions = num_train_reactions
        self.batch_size = batch_size

        self.vocab_size = len(precursor_counts)
        self.num_reserved_ids = num_reserved_ids
        self.num_precursors = self.vocab_size - self.num_reserved_ids

        self.constrain_precursor_layer = constrain_precursor_layer
        self.bias_in_precursor_layer = bias_in_precursor_layer
        # self.norm_in_precursor_projection = norm_in_precursor_projection
        # self.uni_vec = UnifyVector()

        self.predict_precursor_under_mask = predict_precursor_under_mask

        self.attention_num_heads = attention_num_heads
        self.attention_hidden_activation = attention_hidden_activation
        self.attention_hidden_dropout = attention_hidden_dropout
        self.attention_dropout = attention_dropout
        self.attention_initializer_range = attention_initializer_range

        self._all_loss = {
            "cross_entropy": self._loss_cross_entropy,
            "ce_sig_arctanh": self._loss_cross_entropy_linear,
            "ce_linear": self._loss_cross_entropy_sig_arctanh,
            "error_rate_unary": self._loss_error_rate_unary,
            "log_error_rate_unary": self._loss_log_error_rate_unary,
            "error_rate": self._loss_error_rate,
            "log_error_rate": self._loss_log_error_rate,
            "regularization_1": self._loss_regularization_1,
            "cross_entropy_regularized_1": self._loss_cross_entropy_regularized_1,
            "error_rate_unary_regularized_1": self._loss_error_rate_unary_regularized_1,
            "log_error_rate_unary_regularized_1": self._loss_log_error_rate_unary_regularized_1,
            "error_rate_regularized_1": self._loss_error_rate_regularized_1,
            "log_error_rate_regularized_1": self._loss_log_error_rate_regularized_1,
            "focal": self._loss_focal,
            "circle": self._loss_circle,
        }
        self.loss_fn_name = loss_fn
        if "focal" in self.loss_fn_name:
            self._focal_loss_func = BinaryFocalLoss(
                gamma=focal_gamma,
                pos_weight=focal_alpha,
                label_smoothing=focal_label_smoothing,
                from_logits=True,
            )
        elif "circle" in self.loss_fn_name:
            self._circle_loss_func = CircleLoss(
                gamma=circle_gamma,
                margin=circle_margin,
            )

        self._loss_fn = self._all_loss[self.loss_fn_name]
        self.dot_prod_scale = dot_prod_scale
        if self.loss_fn_name in {"ce_sig_arctanh", "ce_linear"}:
            # some functions are not defined on the boundary such as arctanh
            self.dot_prod_scale *= 1 - NEAR_ZERO

        self.initializer_max = initializer_max
        self.kernel_initializer = kernel_initializer
        if self.kernel_initializer == "random_uniform":
            self.kernel_initializer = keras.initializers.RandomUniform(
                minval=-self.initializer_max, maxval=self.initializer_max
            )
        self.regularization_lambda = regularization_lambda
        self.balance_PN = balance_PN
        self.precursor_frequency = tf.constant(
            np.array(self.precursor_counts[self.num_reserved_ids :]).astype(np.float32)
            / self.num_train_reactions
        )
        self.balance_coeff = self.precursor_frequency / (1 - self.precursor_frequency)
        self.clip_logits = clip_logits

        if self.predict_precursor_under_mask:
            self.incomplete_reaction_atten_layer = TransformerBlock(
                hidden_size=self.latent_dim,
                num_attention_heads=self.attention_num_heads,
                intermediate_size=self.latent_dim,
                intermediate_activation=self.attention_hidden_activation,
                hidden_dropout_prob=self.attention_hidden_dropout,
                attention_probs_dropout_prob=self.attention_dropout,
                initializer_range=self.attention_initializer_range,
            )
        else:
            self.incomplete_reaction_atten_layer = None

        if self.constrain_precursor_layer:
            self.precursor_layer = keras.layers.Dense(
                self.num_precursors,
                activation=None,
                kernel_initializer=self.kernel_initializer,
                use_bias=self.bias_in_precursor_layer,
                name="precursor_layer",
                kernel_constraint=keras.constraints.UnitNorm(axis=0),
            )
        else:
            self.precursor_layer = keras.layers.Dense(
                self.num_precursors,
                activation=None,
                kernel_initializer=self.kernel_initializer,
                use_bias=self.bias_in_precursor_layer,
                name="precursor_layer",
            )
        self.has_been_built = False

    def call(
        self, targets, precursors_conditional_indices=None, return_probability=True
    ):
        """

        :param targets: (None, mat_feature_len)
        :param precursors_conditional_indices: (None, max_mats_num-1, )
        :param return_probability: bool
        :return y_pred: (None, num_precursors)
        """
        # targets_emb: (None, latent_dim)
        targets_emb, _, _ = self.mat_encoder(targets)

        # Attention: this is redundant, just used to build precursor_layer first
        #   otherwise, tf would say self.precursor_layer has no attribute kernel
        # Change to torch in the future!
        if not self.has_been_built:
            self.precursor_layer(targets_emb)
            self.has_been_built = True

        if precursors_conditional_indices is not None:
            # precursor_kernel: (latent_dim, num_precursors, )
            precursor_kernel = self.precursor_layer.kernel
            # get mask (None, max_mats_num-1) in bool
            precursors_conditional_mask = tf.greater_equal(
                precursors_conditional_indices, 0
            )
            # precursors_conditional_indices: (None, max_mats_num-1, )
            precursors_conditional_indices = precursors_conditional_indices * tf.cast(
                precursors_conditional_mask, tf.int32
            )
            # get mask (None, max_mats_num-1) in float
            precursors_conditional_mask = tf.cast(
                precursors_conditional_mask, tf.float32
            )
            # precursors_conditional_emb: (None, max_mats_num-1, latent_dim)
            precursors_conditional_emb = tf.gather(
                tf.transpose(precursor_kernel), precursors_conditional_indices
            )
            precursors_conditional_emb = precursors_conditional_emb * tf.expand_dims(
                precursors_conditional_mask, axis=2
            )
            # precursors_conditional_emb: (None, max_mats_num, latent_dim)
            incomplete_reaction_emb = tf.concat(
                (tf.expand_dims(targets_emb, axis=1), precursors_conditional_emb),
                axis=1,
            )
            # incomplete_reaction_mask (None, max_mats_num) in float
            incomplete_reaction_mask = tf.concat(
                (
                    tf.ones(shape=(tf.shape(targets_emb)[0], 1), dtype=tf.float32),
                    precursors_conditional_mask,
                ),
                axis=1,
            )
            # TODO: is this attention_mask correct?
            # attention_mask in 3d: (None, max_mats_num, max_mats_num) in float
            attention_mask = create_attention_mask_from_input_mask(
                from_tensor=incomplete_reaction_emb,
                to_mask=incomplete_reaction_mask,
            )
            # reactions_emb: (None, max_mats_num, latent_dim)
            reactions_emb = self.incomplete_reaction_atten_layer(
                input_tensor=incomplete_reaction_emb,
                attention_mask=attention_mask,
            )
            # reactions_emb: (None, latent_dim)
            reactions_emb = reactions_emb[:, 0, :]
        else:
            # reactions_emb: (None, latent_dim)
            reactions_emb = targets_emb

        # y_pred: (None, num_precursors)
        y_pred = self.precursor_layer(reactions_emb)

        # print('dir(self.precursor_layer)', dir(self.precursor_layer))

        # if self.norm_in_precursor_projection:
        #     # self.precursor_layer.kernel: (latent_dim, num_precursors, )
        #     # kernel_norm: (num_precursors, )
        #     kernel_norm = tf.sqrt(
        #         tf.reduce_sum(
        #             tf.square(tf.transpose(self.precursor_layer.kernel)),
        #             axis=-1,
        #         )
        #     ) + 1e-12
        #     # y_pred: (None, num_precursors)
        #     y_pred = y_pred/kernel_norm
        #     # the activation function in precursor layer is None if norm_in_precursor_projection == True
        y_pred = y_pred * self.dot_prod_scale

        if self.clip_logits:
            y_pred = tf.clip_by_value(y_pred, clip_value_min=-10.0, clip_value_max=10.0)

        if return_probability:
            # y_pred: (None, num_precursors)
            y_pred = tf.sigmoid(y_pred)
        return y_pred

    def _loss_cross_entropy(
        self, reactions, reactions_featurized, precursors_conditional=None
    ):
        """

        :param reactions: (None, max_mats_num, num_eles)
        :param reactions_featurized: (None, max_mats_num, mat_feature_len)
        :param precursors_conditional: (None, max_mats_num-1, num_eles)
        :return loss: (None, )
        """
        # targets: (None, mat_feature_len)
        targets = reactions_featurized[:, 0, :]

        if self.predict_precursor_under_mask and precursors_conditional is not None:
            # precursors_conditional_labels: (None, max_mats_num-1, )
            precursors_conditional_labels = get_mat_label(
                precursors_conditional, self.precursor_labels
            )
            # precursors_conditional_indices: (None, max_mats_num-1, )
            precursors_conditional_indices = tf.cast(
                precursors_conditional_labels - self.num_reserved_ids, tf.int32
            )
            # y_pred: (None, num_precursors)
            y_pred = self.__call__(
                targets, precursors_conditional_indices=precursors_conditional_indices
            )
        else:
            # y_pred: (None, num_precursors)
            y_pred = self.__call__(targets)

        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)

        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )
        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)

        # loss: (None, num_precursors, )
        if self.balance_PN:
            loss = (
                y_real * (-tf.math.log(y_pred))
                + (1 - y_real) * (-tf.math.log(1 - y_pred)) * self.balance_coeff
            )
        else:
            loss = y_real * (-tf.math.log(y_pred)) + (1 - y_real) * (
                -tf.math.log(1 - y_pred)
            )
        # loss: (None, )
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    def _loss_focal(self, reactions, reactions_featurized, precursors_conditional=None):
        """

        :param reactions: (None, max_mats_num, num_eles)
        :param reactions_featurized: (None, max_mats_num, mat_feature_len)
        :param precursors_conditional: (None, max_mats_num-1, num_eles)
        :return loss: (None, )
        """
        # targets: (None, mat_feature_len)
        targets = reactions_featurized[:, 0, :]

        if self.predict_precursor_under_mask and precursors_conditional is not None:
            # precursors_conditional_labels: (None, max_mats_num-1, )
            precursors_conditional_labels = get_mat_label(
                precursors_conditional, self.precursor_labels
            )
            # precursors_conditional_indices: (None, max_mats_num-1, )
            precursors_conditional_indices = tf.cast(
                precursors_conditional_labels - self.num_reserved_ids, tf.int32
            )
            # y_pred: (None, num_precursors)
            y_pred = self.__call__(
                targets,
                precursors_conditional_indices=precursors_conditional_indices,
                return_probability=False,
            )
        else:
            # y_pred: (None, num_precursors)
            y_pred = self.__call__(targets, return_probability=False)

        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)
        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )
        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)

        # loss: (None, num_precursors, )
        loss = self._focal_loss_func.call(y_true=y_real, y_pred=y_pred)

        # loss: (None, )
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    def _loss_circle(
        self, reactions, reactions_featurized, precursors_conditional=None
    ):
        """

        :param reactions: (None, max_mats_num, num_eles)
        :param reactions_featurized: (None, max_mats_num, mat_feature_len)
        :param precursors_conditional: (None, max_mats_num-1, num_eles)
        :return loss: (None, )
        """
        # y_pred: (None, num_precursors)
        targets = reactions_featurized[:, 0, :]

        if self.predict_precursor_under_mask and precursors_conditional is not None:
            # precursors_conditional_labels: (None, max_mats_num-1, )
            precursors_conditional_labels = get_mat_label(
                precursors_conditional, self.precursor_labels
            )
            # precursors_conditional_indices: (None, max_mats_num-1, )
            precursors_conditional_indices = tf.cast(
                precursors_conditional_labels - self.num_reserved_ids, tf.int32
            )
            # y_pred: (None, num_precursors)
            y_pred = self.__call__(
                targets,
                precursors_conditional_indices=precursors_conditional_indices,
                return_probability=False,
            )
        else:
            # y_pred: (None, num_precursors)
            y_pred = self.__call__(targets, return_probability=False)

        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)
        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )
        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)

        # loss: (None, )
        loss = self._circle_loss_func.call(y_true=y_real, y_pred=y_pred)

        return loss

    def _loss_cross_entropy_sig_arctanh(self, reactions, reactions_featurized):
        """
        equivalent to sigmoid(arctanh(dot(x_l,x_r)))

        :param inputs: (None, max_mats_num, mat_feature_len)
        :return loss: (None, )
        """
        assert self.norm_in_precursor_projection == True
        # y_pred: (None, num_precursors)
        targets = reactions_featurized[:, 0, :]
        y_pred = self.__call__(targets, return_probability=False)
        # y_pred is the logits here
        logits = y_pred
        part_a = tf.sqrt(1 + logits)
        part_b = tf.sqrt(1 - logits)
        prob = part_a / (part_a + part_b)

        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)
        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )
        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)

        # loss: (None, num_precursors, )
        if self.balance_PN:
            loss = (
                y_real * (-tf.math.log(prob))
                + (1 - y_real) * (-tf.math.log(1 - prob)) * self.balance_coeff
            )
        else:
            loss = y_real * (-tf.math.log(prob)) + (1 - y_real) * (
                -tf.math.log(1 - prob)
            )
        # loss: (None, )
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    def _loss_cross_entropy_linear(self, reactions, reactions_featurized):
        """
        equivalent to sigmoid(2*arctanh(dot(x_l,x_r)))

        :param inputs: (None, max_mats_num, mat_feature_len)
        :return loss: (None, )
        """
        assert self.norm_in_precursor_projection == True
        # y_pred: (None, num_precursors)
        targets = reactions_featurized[:, 0, :]
        y_pred = self.__call__(targets, return_probability=False)
        # y_pred is the logits here
        logits = y_pred
        prob = (1.0 + logits) / 2.0

        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)
        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )
        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)

        # loss: (None, num_precursors, )
        if self.balance_PN:
            loss = (
                y_real * (-tf.math.log(prob))
                + (1 - y_real) * (-tf.math.log(1 - prob)) * self.balance_coeff
            )
        else:
            loss = y_real * (-tf.math.log(prob)) + (1 - y_real) * (
                -tf.math.log(1 - prob)
            )
        # loss: (None, )
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    def _loss_error_rate_unary(self, reactions, reactions_featurized):
        """

        :param inputs: (None, max_mats_num, mat_feature_len)
        :return loss: (None, )
        """
        # y_pred: (None, num_precursors)
        targets = reactions_featurized[:, 0, :]
        y_pred = self.__call__(targets)
        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)
        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )
        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)
        # y_correctness: (None, num_precursors, )
        # y_correctness = (2.0*y_real-1.0)*y_pred + (1.0-y_real)
        y_error = tf.pow(y_pred - y_real, 2)

        # y_correctness: (None, )
        y_error = tf.reduce_sum(y_error, -1)
        # loss: (None, )
        loss = y_error
        return loss

    def _loss_log_error_rate_unary(self, reactions, reactions_featurized):
        error_rate = self._loss_error_rate_unary(reactions, reactions_featurized)
        loss = tf.math.log(error_rate)
        return loss

    def _loss_error_rate(self, reactions, reactions_featurized):
        """

        :param inputs: (None, max_mats_num, mat_feature_len)
        :return loss: (None, )
        """
        # y_pred: (None, num_precursors)
        targets = reactions_featurized[:, 0, :]
        y_pred = self.__call__(targets)
        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)
        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )
        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)
        # y_correctness: (None, num_precursors, )
        y_error = tf.abs(y_pred - y_real)
        # y_correctness: (None, )
        y_error = tf.pow(tf.reduce_sum(y_error, -1), 2) - tf.reduce_sum(
            tf.pow(y_error, 2), -1
        )
        y_error = y_error / 2.0
        # loss: (None, )
        loss = y_error
        return loss

    def _loss_log_error_rate(self, reactions, reactions_featurized):
        error_rate = self._loss_error_rate(reactions, reactions_featurized)
        loss = tf.math.log(error_rate)
        return loss

    def _loss_regularization_1(self, reactions, reactions_featurized):
        """

        :param inputs: (None, max_mats_num, mat_feature_len)
        :return loss: (None, )
        """
        # y_pred: (None, num_precursors)
        targets = reactions_featurized[:, 0, :]
        y_pred = self.__call__(targets)
        # precursors: (None, max_mats_num-1, num_eles)
        precursors = reactions[:, 1:, :]
        # precursors_labels: (None, max_mats_num-1, )
        precursors_labels = get_mat_label(precursors, self.precursor_labels)
        # get mask (None, max_mats_num-1)
        precursors_mask = tf.greater_equal(precursors_labels, self.num_reserved_ids)
        # y_real: (None, vocab_size, )
        y_real = tf.map_fn(
            lambda x: tf.scatter_nd(
                indices=tf.expand_dims(x, axis=1),
                updates=tf.ones(tf.shape(x), x.dtype),
                shape=(self.vocab_size,),
            ),
            precursors_labels,
        )

        # y_real: (None, num_precursors, )
        y_real = y_real[:, self.num_reserved_ids :]
        y_real = tf.cast(y_real, tf.float32)

        # precursors_indices: (None, max_mats_num-1, )
        # indices for precursor layer kernel matrix
        precursors_indices = tf.cast(
            precursors_labels - self.num_reserved_ids, tf.int32
        )
        precursors_indices = precursors_indices * tf.cast(precursors_mask, tf.int32)
        precursors_mask = tf.cast(precursors_mask, tf.float32)

        # if self.norm_in_precursor_projection:
        #     # self.precursor_layer.kernel: (latent_dim, num_precursors, )
        #     # precursor_kernel: (latent_dim, num_precursors, )
        #     precursor_kernel = tf.transpose(self.uni_vec(tf.transpose(self.precursor_layer.kernel)))
        # else:

        precursor_kernel = self.precursor_layer.kernel
        # precursors_vectors: (None, max_mats_num-1, latent_dim)
        precursors_vectors = tf.gather(
            tf.transpose(precursor_kernel), precursors_indices
        )
        precursors_vectors = precursors_vectors * tf.expand_dims(
            precursors_mask, axis=2
        )
        # precursors_dot: (None, max_mats_num-1, num_precursors)
        precursors_dot = tf.matmul(precursors_vectors, precursor_kernel)
        # y_real: (None, max_mats_num-1, num_precursors, )
        dot_labels = tf.tile(
            tf.expand_dims(y_real, axis=1), multiples=[1, self.max_mats_num - 1, 1]
        )
        # regularization_term: (None, max_mats_num-1, num_precursors, )
        # '-' is used in sigmoid_cross_entropy_with_logits
        # therefore is already loss rather likelihood
        regularization_term = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=dot_labels,
            logits=precursors_dot,
        )
        regularization_term = regularization_term * tf.expand_dims(
            precursors_mask, axis=2
        )
        # regularization_term: (None, )
        regularization_term = tf.reduce_sum(regularization_term, axis=[1, 2])
        loss = self.regularization_lambda * regularization_term
        return loss

    def _loss_cross_entropy_regularized_1(self, reactions, reactions_featurized):
        loss = self._loss_cross_entropy(reactions, reactions_featurized)
        regularization_term_1 = self._loss_regularization_1(
            reactions, reactions_featurized
        )
        loss = loss + regularization_term_1
        return loss

    def _loss_error_rate_unary_regularized_1(self, reactions, reactions_featurized):
        loss = self._loss_error_rate_unary(reactions, reactions_featurized)
        regularization_term_1 = self._loss_regularization_1(
            reactions, reactions_featurized
        )
        loss = loss + regularization_term_1
        return loss

    def _loss_log_error_rate_unary_regularized_1(self, reactions, reactions_featurized):
        loss = self._loss_log_error_rate_unary(reactions, reactions_featurized)
        regularization_term_1 = self._loss_regularization_1(
            reactions, reactions_featurized
        )
        loss = loss + regularization_term_1
        return loss

    def _loss_error_rate_regularized_1(self, reactions, reactions_featurized):
        loss = self._loss_error_rate(reactions, reactions_featurized)
        regularization_term_1 = self._loss_regularization_1(
            reactions, reactions_featurized
        )
        loss = loss + regularization_term_1
        return loss

    def _loss_log_error_rate_regularized_1(self, reactions, reactions_featurized):
        loss = self._loss_log_error_rate(reactions, reactions_featurized)
        regularization_term_1 = self._loss_regularization_1(
            reactions, reactions_featurized
        )
        loss = loss + regularization_term_1
        return loss

    def predict_precursors(
        self, targets, precursors_conditional=None, output_thresh=0.5
    ):
        """

        :param targets: (None, mat_feature_len)
        :return all_pre_lists: list of list of (composition array, score)
        """
        all_pre_lists = []

        if precursors_conditional is not None:
            # precursors_conditional_labels: (None, max_mats_num-1, )
            precursors_conditional_labels = get_mat_label(
                precursors_conditional, self.precursor_labels
            )
            # precursors_conditional_indices: (None, max_mats_num-1, )
            precursors_conditional_indices = tf.cast(
                precursors_conditional_labels - self.num_reserved_ids, tf.int32
            )

            # y_pred: (None, num_precursors)
            y_pred = self.__call__(
                targets, precursors_conditional_indices=precursors_conditional_indices
            )
        else:
            # y_pred: (None, num_precursors)
            y_pred = self.__call__(targets)

        for a_y in y_pred:
            pre_list_pred = self.precursor_compositions[self.num_reserved_ids :][
                a_y > output_thresh
            ].numpy()
            pre_score_pred = a_y[a_y > output_thresh].numpy()
            all_pre_lists.append(
                [
                    {
                        "composition": comp,
                        "score": score,
                    }
                    for (comp, score) in zip(pre_list_pred, pre_score_pred)
                ]
            )
            all_pre_lists[-1] = sorted(
                all_pre_lists[-1],
                key=lambda x: x["score"],
                reverse=True,
            )
        return all_pre_lists

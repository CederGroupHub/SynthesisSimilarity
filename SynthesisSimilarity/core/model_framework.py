import os
import shutil
from pprint import pprint

import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import inspect
from typing import Optional

from .bert_optimization import create_optimizer
from .encoders import MaterialEncoder
from .task_models import MaterialDecoder
from .task_models import PrecursorsPredict
from .losses import CustomLoss
from .losses import MultiLossLayer

__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


class MultiTasksOnRecipes(keras.Model):
    def __init__(
        self,
        max_mats_num,
        all_eles,
        ele_counts,
        all_ions,
        mat_feature_len,
        ele_dim_features,
        num_attention_layers,
        num_attention_heads,
        hidden_activation,
        hidden_dropout,
        attention_dropout,
        initializer_range,
        task_to_add,
        num_train_reactions,
        batch_size,
        num_train_steps,
        featurizer_type="default",
        zero_shift_init_value=-0.5,
        zero_shift_trainable=False,
        decoder_final_activation=None,
        decoder_loss_fn="mse",
        ele_pred_stoi_scale=1.0,
        bias_in_element_layer=True,
        constrain_element_layer=False,
        norm_in_element_projection=False,
        ele_pred_dot_prod_scale=1.0,
        ele_pred_balance_PN=True,
        ele_pred_clip_logits=False,
        ele_pred_focal_gamma=2.0,
        ele_pred_focal_alpha=0.25,
        ele_pred_focal_label_smoothing=0.0,
        ele_pred_circle_gamma=64,
        ele_pred_circle_margin=0.25,
        mat_variance_loss_fn="abs_dot_sim",
        pre_pred_under_mask=True,
        pre_pred_atten_num_heads=1,
        pre_pred_atten_hidden_activation="gelu",
        pre_pred_atten_hidden_dropout=0.1,
        pre_pred_atten_dropout=0.1,
        pre_pred_atten_initializer_range=0.02,
        pre_pred_loss_fn="cross_entropy",
        pre_pred_dot_prod_scale=1.0,
        pre_pred_kernel_initializer="glorot_uniform",
        pre_pred_initializer_max=0.05,
        pre_pred_lambda=1.0,
        pre_pred_balance_PN=True,
        pre_pred_clip_logits=True,
        pre_pred_focal_gamma=2.0,
        pre_pred_focal_alpha=0.25,
        pre_pred_focal_label_smoothing=0.0,
        pre_pred_circle_gamma=64,
        pre_pred_circle_margin=0.25,
        constrain_precursor_layer=False,
        bias_in_precursor_layer=False,
        syn_type_pred_loss_fn="cross_entropy",
        syn_type_pred_kernel_initializer="glorot_uniform",
        syn_type_pred_initializer_max=0.05,
        syn_type_pred_lambda=1.0,
        syn_type_pred_balance_PN=True,
        syn_type_pred_clip_logits=True,
        syn_type_pred_focal_gamma=2.0,
        syn_type_pred_focal_alpha=0.25,
        syn_type_pred_focal_label_smoothing=0.0,
        syn_type_pred_circle_gamma=64,
        syn_type_pred_circle_margin=0.25,
        syn_type_pred_dot_prod_scale=1.0,
        constrain_syn_type_layer=False,
        bias_in_syn_type_layer=True,
        norm_in_syn_type_projection=False,
        num_reserved_ids=10,
        mat_labels: Optional[list] = None,
        mat_compositions: Optional[list] = None,
        mat_counts: Optional[list] = None,
        syn_type_labels: Optional[list] = None,
        syn_type_counts: Optional[list] = None,
        lr_method_name="adam",
        init_learning_rate=5e-5,
        num_warmup_steps=10000,
        encoder_type="simple_hidden",
        encoder_normalize_output=True,
        ele_emb_init_max=10,
        weight_mat_decoder=1.0,
        weight_pre_predict=1.0,
        weight_syn_type_predict=1.0,
        weight_sim_between_react=1.0,
        weight_mat_variance=1.0,
        use_adaptive_multi_loss=True,
        model_path=None,
        model_name=None,
        reload_model=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Model location
        self.set_model_path(model_path, model_name)
        if reload_model:
            self.reload(self.model_path)
        else:
            # init
            self.max_mats_num = max_mats_num
            self.all_eles = all_eles
            self.ele_counts = ele_counts
            self.all_ions = all_ions
            self.num_eles = len(self.all_eles)
            self.mat_feature_len = mat_feature_len
            self.ele_dim_features = ele_dim_features
            self.mat_dim = ele_dim_features
            self.num_attention_layers = num_attention_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_activation = hidden_activation
            self.hidden_dropout = hidden_dropout
            self.attention_dropout = attention_dropout
            self.initializer_range = initializer_range
            self.task_to_add = [t.lower() for t in task_to_add]
            self.featurizer_type = featurizer_type
            self.zero_shift_init_value = zero_shift_init_value
            self.zero_shift_trainable = zero_shift_trainable
            self.decoder_final_activation = decoder_final_activation
            self.decoder_loss_fn = decoder_loss_fn
            self.ele_pred_stoi_scale = ele_pred_stoi_scale
            self.bias_in_element_layer = bias_in_element_layer
            self.constrain_element_layer = constrain_element_layer
            self.norm_in_element_projection = norm_in_element_projection
            self.ele_pred_dot_prod_scale = ele_pred_dot_prod_scale
            self.ele_pred_balance_PN = ele_pred_balance_PN
            self.ele_pred_clip_logits = ele_pred_clip_logits
            self.ele_pred_focal_gamma = ele_pred_focal_gamma
            self.ele_pred_focal_alpha = ele_pred_focal_alpha
            self.ele_pred_focal_label_smoothing = ele_pred_focal_label_smoothing
            self.ele_pred_circle_gamma = ele_pred_circle_gamma
            self.ele_pred_circle_margin = ele_pred_circle_margin
            self.mat_variance_loss_fn = mat_variance_loss_fn
            self.pre_pred_under_mask = pre_pred_under_mask
            self.pre_pred_atten_num_heads = pre_pred_atten_num_heads
            self.pre_pred_atten_hidden_activation = pre_pred_atten_hidden_activation
            self.pre_pred_atten_hidden_dropout = pre_pred_atten_hidden_dropout
            self.pre_pred_atten_dropout = pre_pred_atten_dropout
            self.pre_pred_atten_initializer_range = pre_pred_atten_initializer_range
            self.pre_pred_loss_fn = pre_pred_loss_fn
            self.pre_pred_dot_prod_scale = pre_pred_dot_prod_scale
            self.pre_pred_kernel_initializer = pre_pred_kernel_initializer
            self.pre_pred_initializer_max = pre_pred_initializer_max
            self.pre_pred_lambda = pre_pred_lambda
            self.pre_pred_balance_PN = pre_pred_balance_PN
            self.pre_pred_clip_logits = pre_pred_clip_logits
            self.pre_pred_focal_gamma = pre_pred_focal_gamma
            self.pre_pred_focal_alpha = pre_pred_focal_alpha
            self.pre_pred_focal_label_smoothing = pre_pred_focal_label_smoothing
            self.pre_pred_circle_gamma = pre_pred_circle_gamma
            self.pre_pred_circle_margin = pre_pred_circle_margin
            self.constrain_precursor_layer = constrain_precursor_layer
            self.bias_in_precursor_layer = bias_in_precursor_layer
            self.syn_type_pred_loss_fn = syn_type_pred_loss_fn
            self.syn_type_pred_dot_prod_scale = syn_type_pred_dot_prod_scale
            self.syn_type_pred_kernel_initializer = syn_type_pred_kernel_initializer
            self.syn_type_pred_initializer_max = syn_type_pred_initializer_max
            self.syn_type_pred_lambda = syn_type_pred_lambda
            self.syn_type_pred_balance_PN = syn_type_pred_balance_PN
            self.syn_type_pred_clip_logits = syn_type_pred_clip_logits
            self.syn_type_pred_focal_gamma = syn_type_pred_focal_gamma
            self.syn_type_pred_focal_alpha = syn_type_pred_focal_alpha
            self.syn_type_pred_focal_label_smoothing = (
                syn_type_pred_focal_label_smoothing
            )
            self.syn_type_pred_circle_gamma = syn_type_pred_circle_gamma
            self.syn_type_pred_circle_margin = syn_type_pred_circle_margin
            self.constrain_syn_type_layer = constrain_syn_type_layer
            self.bias_in_syn_type_layer = bias_in_syn_type_layer
            self.norm_in_syn_type_projection = norm_in_syn_type_projection
            self.num_reserved_ids = num_reserved_ids
            if mat_labels is not None:
                # mat_labels is a tf lookup of {mat_str: mat_label}
                # label is the index in mat_counts
                # mat_label is int64
                # default 1 for <UNK>
                # 0 is for <MASK>
                self.mat_labels = tf.lookup.StaticHashTable(
                    initializer=tf.lookup.KeyValueTensorInitializer(
                        keys=mat_labels,
                        values=tf.range(len(mat_labels), dtype=tf.int64),
                    ),
                    default_value=tf.constant(1, dtype=tf.int64),
                    name="mat_labels",
                )
                # mat_compositions is tf list of compositions, index is
                # mat_label
                self.mat_compositions = tf.constant(mat_compositions)
                # mat_counts is python list of int, index is mat_label
                self.mat_counts = mat_counts
            if syn_type_labels is not None:
                self.syn_type_labels = tf.lookup.StaticHashTable(
                    initializer=tf.lookup.KeyValueTensorInitializer(
                        keys=syn_type_labels,
                        values=tf.range(len(syn_type_labels), dtype=tf.int64),
                    ),
                    default_value=tf.constant(1, dtype=tf.int64),
                    name="syn_type_labels",
                )
                self.syn_type_strings = tf.constant(syn_type_labels)
                self.syn_type_counts = syn_type_counts
            self.num_train_reactions = num_train_reactions
            self.batch_size = batch_size
            self.num_train_steps = num_train_steps
            self.lr_method_name = lr_method_name.lower()
            self.init_learning_rate = init_learning_rate
            self.num_warmup_steps = num_warmup_steps
            self.encoder_type = encoder_type
            self.encoder_normalize_output = encoder_normalize_output
            self.ele_emb_init_max = ele_emb_init_max
            self.weight_mat_decoder = weight_mat_decoder
            self.weight_pre_predict = weight_pre_predict
            self.weight_syn_type_predict = weight_syn_type_predict
            self.weight_sim_between_react = weight_sim_between_react
            self.weight_mat_variance = weight_mat_variance
            self.use_adaptive_multi_loss = use_adaptive_multi_loss

            if self.encoder_type == "simple_hidden":
                self.mat_encoder = MaterialEncoder(
                    mat_feature_len=self.mat_feature_len,
                    dim_features=self.ele_dim_features,
                    latent_dim=self.mat_dim,
                    zero_shift_init_value=self.zero_shift_init_value,
                    zero_shift_trainable=self.zero_shift_trainable,
                    num_attention_layers=self.num_attention_layers,
                    num_attention_heads=self.num_attention_heads,
                    hidden_activation=self.hidden_activation,
                    hidden_dropout=self.hidden_dropout,
                    attention_dropout=self.attention_dropout,
                    initializer_range=self.initializer_range,
                    normalize_output=self.encoder_normalize_output,
                )
            else:
                self.mat_encoder = None
                raise ValueError("encoder_type is not properly specified!")

            # add tasks
            self.task_names = []
            self._weight_by_task = {}
            self._model_by_task = {}
            self._loss_by_task = {}

            task_name = "mat"
            if task_name in self.task_to_add:
                mat_decoder = MaterialDecoder(
                    mat_feature_len=self.mat_feature_len,
                    num_eles=self.num_eles,
                    num_train_reactions=self.num_train_reactions,
                    latent_dim=self.mat_dim,
                    intermediate_dim=64,
                    mat_encoder=self.mat_encoder,
                    final_activation=self.decoder_final_activation,
                    loss_fn=self.decoder_loss_fn,
                    stoi_scale=self.ele_pred_stoi_scale,
                    bias_in_element_layer=self.bias_in_element_layer,
                    constrain_element_layer=self.constrain_element_layer,
                    norm_in_element_projection=self.norm_in_element_projection,
                    dot_prod_scale=self.ele_pred_dot_prod_scale,
                    balance_PN=self.ele_pred_balance_PN,
                    clip_logits=self.ele_pred_clip_logits,
                    target_ele_labels=self.all_eles,
                    target_ele_counts=self.ele_counts,
                    focal_gamma=self.ele_pred_focal_gamma,
                    focal_alpha=self.ele_pred_focal_alpha,
                    focal_label_smoothing=self.ele_pred_focal_label_smoothing,
                    circle_gamma=self.ele_pred_circle_gamma,
                    circle_margin=self.ele_pred_circle_margin,
                )
                self.add_task(
                    task_name=task_name,
                    task_model=mat_decoder,
                    task_loss=self.get_loss_mat,
                    task_weight=self.weight_mat_decoder,
                )

            task_name = "reaction_pre"
            if task_name in self.task_to_add:
                pre_predict = PrecursorsPredict(
                    mat_feature_len=self.mat_feature_len,
                    max_mats_num=self.max_mats_num,
                    latent_dim=self.mat_dim,
                    mat_encoder=self.mat_encoder,
                    precursor_labels=self.mat_labels,
                    precursor_compositions=self.mat_compositions,
                    precursor_counts=self.mat_counts,
                    num_train_reactions=self.num_train_reactions,
                    batch_size=self.batch_size,
                    num_reserved_ids=self.num_reserved_ids,
                    constrain_precursor_layer=self.constrain_precursor_layer,
                    bias_in_precursor_layer=self.bias_in_precursor_layer,
                    predict_precursor_under_mask=self.pre_pred_under_mask,
                    attention_num_heads=self.pre_pred_atten_num_heads,
                    attention_hidden_activation=self.pre_pred_atten_hidden_activation,
                    attention_hidden_dropout=self.pre_pred_atten_hidden_dropout,
                    attention_dropout=self.pre_pred_atten_dropout,
                    attention_initializer_range=self.pre_pred_atten_initializer_range,
                    loss_fn=self.pre_pred_loss_fn,
                    kernel_initializer=self.pre_pred_kernel_initializer,
                    initializer_max=self.pre_pred_initializer_max,
                    regularization_lambda=self.pre_pred_lambda,
                    dot_prod_scale=self.pre_pred_dot_prod_scale,
                    balance_PN=self.pre_pred_balance_PN,
                    clip_logits=self.pre_pred_clip_logits,
                    focal_gamma=self.pre_pred_focal_gamma,
                    focal_alpha=self.pre_pred_focal_alpha,
                    focal_label_smoothing=self.pre_pred_focal_label_smoothing,
                    circle_gamma=self.pre_pred_circle_gamma,
                    circle_margin=self.pre_pred_circle_margin,
                )
                self.add_task(
                    task_name=task_name,
                    task_model=pre_predict,
                    task_loss=self.get_loss_reaction_pre,
                    task_weight=self.weight_pre_predict,
                )

            # optimizer
            if self.lr_method_name == "sgd":
                self.optimizer = keras.optimizers.SGD(
                    learning_rate=self.init_learning_rate,
                )
            elif self.lr_method_name == "adam":
                self.optimizer = keras.optimizers.Adam(
                    learning_rate=self.init_learning_rate,
                )
            elif self.lr_method_name == "adamdecay":
                # a good ref
                # https://github.com/google-research/bert/blob/master/optimization.py
                self.optimizer = create_optimizer(
                    init_lr=self.init_learning_rate,
                    num_train_steps=self.num_train_steps,
                    num_warmup_steps=self.num_warmup_steps,
                )
            else:
                raise ValueError(
                    "Not implemented learning method: {}".format(self.lr_method_name)
                )

            # loss function
            # https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
            # https://arxiv.org/pdf/1705.07115.pdf
            self.loss = CustomLoss(name="loss_layer")
            # self.loss = keras.losses.MeanAbsoluteError()
        self.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )

    def set_model_path(self, model_path, model_name):
        if not model_path:
            if model_name:
                self.model_path = os.path.join("../generated", model_name)
            else:
                self.model_path = os.path.join("../generated", "model_0")
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        else:
            self.model_path = model_path

    def reload(self, model_path):
        pass

    def call(self, inputs, training=None):
        loss = []
        for task in self.task_names:
            loss.append(
                self._weight_by_task[task] * self._loss_by_task[task](inputs, task)
            )

        loss = tf.stack(loss, axis=1)
        if self.use_adaptive_multi_loss and len(self.task_names) > 1:
            loss = self.adaptive_loss(loss)
        else:
            loss = tf.reduce_sum(loss, axis=1)

        # regularize by Sim(P)lnSim(T)
        ...

        return loss

    def add_task(self, task_name, task_model, task_loss, task_weight=1.0):
        if task_name in self._model_by_task:
            return
        self.task_names.append(task_name)
        self._model_by_task[task_name] = task_model
        self._loss_by_task[task_name] = task_loss
        self._weight_by_task[task_name] = task_weight
        if (
            self.use_adaptive_multi_loss
            and len(self.task_names) > 1
            and set(self.task_to_add).issubset(set(self.task_names))
        ):
            self.adaptive_loss = MultiLossLayer(
                task_names=self.task_names,
            )

    def get_loss_mat(self, inputs, task_name):
        """
        recover composition of materials
        single material embedding
        :param inputs: same as call()
        :param task_name: string, will be filled according to the record added by add_task()
        :return: (batch_size, ) for each batch data
        """

        # all_reactions: (batch_size, 2, max_mats_num, num_eles)
        all_reactions = tf.stack(
            [
                inputs["reaction_1"],
                inputs["reaction_2"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num, num_eles)
        all_reactions = tf.reshape(
            all_reactions, (-1, self.max_mats_num, self.num_eles)
        )

        # all_reactions: (batch_size, 2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.stack(
            [
                inputs["reaction_1_featurized"],
                inputs["reaction_2_featurized"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.reshape(
            all_reactions_featurized, (-1, self.max_mats_num, self.mat_feature_len)
        )

        # only use target for stoichiometry recovery
        # all_mats: (batch_size*2, num_eles)
        all_mats = all_reactions[:, 0, :]
        # all_mats_featurized: (batch_size*2, mat_feature_len)
        all_mats_featurized = all_reactions_featurized[:, 0, :]

        # loss_mat: (batch_size*2, )
        loss_mat = self._model_by_task[task_name]._loss_fn(
            all_mats, all_mats_featurized
        )
        # loss_mat: (batch_size, 2)
        loss_mat = tf.reshape(loss_mat, (-1, 2))
        # loss_mat: (batch_size, )
        loss_mat = tf.reduce_sum(loss_mat, -1)
        return loss_mat

    def get_loss_mat_variance(self, inputs, task_name):
        """
        recover composition of materials
        single material embedding
        :param inputs: same as call()
        :param task_name: string, will be filled according to the record added by add_task()
        :return: (batch_size, ) for each batch data
        """

        # all_reactions: (batch_size, 2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.stack(
            [
                inputs["reaction_1_featurized"],
                inputs["reaction_2_featurized"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.reshape(
            all_reactions_featurized, (-1, self.max_mats_num, self.mat_feature_len)
        )

        # all_mats_featurized: (batch_size*2, mat_feature_len)
        all_mats_featurized = all_reactions_featurized[:, 0, :]

        # loss_mat: (batch_size*2, )
        loss_mat_variance = self._model_by_task[task_name]._loss_fn(all_mats_featurized)

        # loss_mat_variance: (batch_size, 2)
        loss_mat_variance = tf.reshape(loss_mat_variance, (-1, 2))
        # loss_mat_variance: (batch_size, )
        loss_mat_variance = tf.reduce_sum(loss_mat_variance, -1)

        return loss_mat_variance

    def get_loss_reaction_pre(self, inputs, task_name):
        """
        word2vec/bert - like material embedding in recipe
        :param inputs: same as call()
        :param task_name: string, will be filled according to the record added by add_task()
        :return: (batch_size, ) for each batch data
        """
        # all_reactions: (batch_size, 2, max_mats_num, num_eles)
        all_reactions = tf.stack(
            [
                inputs["reaction_1"],
                inputs["reaction_2"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num, num_eles)
        all_reactions = tf.reshape(
            all_reactions,
            (-1, self.max_mats_num, self.num_eles),
        )

        # all_reactions_featurized: (batch_size, 2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.stack(
            [
                inputs["reaction_1_featurized"],
                inputs["reaction_2_featurized"],
            ],
            axis=1,
        )
        # all_reactions_featurized: (batch_size*2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.reshape(
            all_reactions_featurized,
            (-1, self.max_mats_num, self.mat_feature_len),
        )

        # all_reactions: (batch_size, 2, max_mats_num-1, num_eles)
        all_precursors_conditional = tf.stack(
            [
                inputs["precursors_1_conditional"],
                inputs["precursors_2_conditional"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num-1, num_eles)
        all_precursors_conditional = tf.reshape(
            all_precursors_conditional,
            (-1, self.max_mats_num - 1, self.num_eles),
        )

        # all_reactions = tf.random.shuffle(all_reactions)
        loss_reaction = self._model_by_task[task_name]._loss_fn(
            reactions=all_reactions,
            reactions_featurized=all_reactions_featurized,
            precursors_conditional=all_precursors_conditional,
        )
        loss_reaction = tf.reshape(loss_reaction, (-1, 2))
        loss_reaction = tf.reduce_sum(loss_reaction, -1)
        return loss_reaction

    def get_mat_vector(self, comps: np.ndarray, **kwargs):
        # TODO: comps here should be featurized already, maybe automate this process
        assert comps.shape[1] == self.mat_feature_len
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(comps)
        return x_mat_mean

    def recover_mat(self, comps, **kwargs):
        assert comps.shape[1] == self.mat_feature_len
        x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(comps)
        mat = self._model_by_task["mat"](x_mat_mean)
        tf.print(x_mat_mean, x_mat_log_var, x_mat)
        return mat


class ExportModel(keras.Model):
    def __init__(
        self,
        max_mats_num,
        all_eles,
        all_ions,
        mat_feature_len,
        ele_dim_features,
        num_attention_layers,
        num_attention_heads,
        hidden_activation,
        hidden_dropout,
        attention_dropout,
        initializer_range,
        encoder_type="attention",
        encoder_normalize_output=True,
        ele_emb_init_max=10,
        featurizer_type="default",
        zero_shift_init_value=-0.5,
        zero_shift_trainable=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # init
        self.max_mats_num = max_mats_num
        self.all_eles = all_eles
        self.num_eles = len(self.all_eles)
        self.all_ions = all_ions
        self.mat_feature_len = mat_feature_len
        self.ele_dim_features = ele_dim_features
        self.mat_dim = ele_dim_features
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.encoder_type = encoder_type
        self.encoder_normalize_output = encoder_normalize_output
        self.ele_emb_init_max = ele_emb_init_max
        self.featurizer_type = featurizer_type
        self.zero_shift_init_value = zero_shift_init_value
        self.zero_shift_trainable = zero_shift_trainable

        if self.encoder_type == "simple_hidden":
            self.mat_encoder = MaterialEncoder(
                mat_feature_len=self.mat_feature_len,
                dim_features=self.ele_dim_features,
                latent_dim=self.mat_dim,
                zero_shift_init_value=self.zero_shift_init_value,
                zero_shift_trainable=self.zero_shift_trainable,
                num_attention_layers=self.num_attention_layers,
                num_attention_heads=self.num_attention_heads,
                hidden_activation=self.hidden_activation,
                hidden_dropout=self.hidden_dropout,
                attention_dropout=self.attention_dropout,
                initializer_range=self.initializer_range,
                normalize_output=self.encoder_normalize_output,
            )
        else:
            self.mat_encoder = None
            raise ValueError("encoder_type is not properly specified!")


def export_model(model_dir, model_config, multi_task_model):
    mat_feature_len = model_config["mat_feature_len"]
    input_shape = [None, mat_feature_len]
    input_type = tf.float32

    class CallableExportModel(ExportModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @tf.function(input_signature=[tf.TensorSpec(input_shape, input_type)])
        def __call__(self, x):
            assert x.shape[1] == self.mat_feature_len
            x_mat_mean, x_mat_log_var, x_mat = self.mat_encoder(x)
            return x_mat_mean

    signature = inspect.signature(ExportModel.__init__)
    model_config = dict(
        filter(lambda x: x[0] in signature.parameters, model_config.items())
    )
    model = CallableExportModel(**model_config)
    cp_dir = os.path.join(model_dir, "saved_checkpoint")
    cp_path = os.path.join(cp_dir, "saved_cp.ckpt")
    multi_task_model.mat_encoder.save_weights(cp_path)
    if model.encoder_type not in {"empty", None}:
        model.mat_encoder.load_weights(cp_path)
    if os.path.exists(cp_dir):
        shutil.rmtree(cp_dir)
    # Save the entire model as a SavedModel.
    model_path = os.path.join(model_dir, "saved_model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    tf.saved_model.save(model, model_path)
    print("encoder model saved to {}".format(model_dir))

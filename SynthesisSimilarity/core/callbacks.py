import datetime
import os
import pdb
import shutil
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import collections
from pprint import pprint
import scipy.spatial
import matplotlib.pyplot as plt
from adjustText import adjust_text
from typing import List
from typing import Optional

from SynthesisSimilarity.core.bert_modeling import create_attention_mask_from_input_mask
from SynthesisSimilarity.core.utils import (
    repeat_in_last_dimension,
    composition_to_array,
)
from SynthesisSimilarity.core.utils import dict_to_simple_formula, group_precursors
from SynthesisSimilarity.core.utils import formula_to_array, get_composition_string
from SynthesisSimilarity.core.utils import array_to_formula, plot_heatmap
from SynthesisSimilarity.core.model_framework import export_model, MultiTasksOnRecipes
from SynthesisSimilarity.core.mat_featurization import (
    featurize_list_of_composition,
    mat_featurizer,
)
from SynthesisSimilarity.core.tf_utils import get_variables_by_name
from SynthesisSimilarity.core.layers import UnifyVector
from SynthesisSimilarity.core import vector_utils


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


class PrintVarCallback(tf.keras.callbacks.Callback):
    def __init__(self, variables_to_print, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.variables_to_print = variables_to_print
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        print("the {}th epoch ends".format(epoch))
        print("logs", logs, logs.get("loss"))

        if self.variables_to_print == "all":
            # print all
            if self.verbose:
                print(self.model.variables)
            else:
                for v in self.model.variables:
                    print(v.name)
            pass
        elif isinstance(self.variables_to_print, list):
            for v in filter(
                lambda v: v.name in self.variables_to_print, self.model.variables
            ):
                print(v)
        print()


class PrintCompCallback(tf.keras.callbacks.Callback):
    def __init__(self, comps, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.comps = comps
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        print("the {}th epoch ends".format(epoch))
        print("logs", logs, logs.get("loss"))
        if epoch == 0:
            print("inputs:")
            print(tf.transpose(self.comps))
        # mask in 2d: (None, num_eles+1)
        inputs = self.comps
        processed_inputs = inputs
        mask_2d = tf.cast(tf.not_equal(processed_inputs, 0), tf.float32)
        _x = self.model.mat_encoder.emp_emb(processed_inputs)

        # ele_weights in 3d: (None, num_eles+1, latent_dim)
        ele_weights = repeat_in_last_dimension(
            from_tensor=processed_inputs,
            from_seq_length=mask_2d.shape[-1],
            to_latent_dim=_x.shape[-1],
        )
        # ele_mask in 3d: (None, num_eles+1, latent_dim)
        ele_mask = repeat_in_last_dimension(
            from_tensor=mask_2d,
            from_seq_length=mask_2d.shape[-1],
            to_latent_dim=_x.shape[-1],
        )
        # mask in 3d: (None, num_eles+1, num_eles+1)
        attention_mask = create_attention_mask_from_input_mask(_x, mask_2d)
        print("after emp_emb: ", tf.transpose(_x))
        for atten in self.model.mat_encoder.atten_layers:

            from_tensor_norm = (
                tf.sqrt(tf.reduce_sum(tf.square(_x), axis=-1, keepdims=True)) + 1e-12
            )
            from_tensor_univec = _x / from_tensor_norm
            print("from_tensor_norm")
            print(from_tensor_norm)
            # query_tensor = atten.attention_layer.query_dense(from_tensor_univec)
            query_tensor = atten.attention_layer.query_dense(_x)

            # `key_tensor` = [B, T, N, H]
            key_tensor = atten.attention_layer.key_dense(_x)

            # `value_tensor` = [B, T, N, H]
            value_tensor = atten.attention_layer.value_dense(_x)

            # Take the dot product between "query" and "key" to get the raw
            # attention scores.
            attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
            attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(1)))

            if attention_mask is not None:
                # `attention_mask` = [B, 1, F, T]
                attention_mask_4d = tf.expand_dims(attention_mask, axis=[1])

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                adder = (
                    1.0 - tf.cast(attention_mask_4d, attention_scores.dtype)
                ) * -10000.0

                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_scores += adder
            print("attention_scores: ", attention_scores.shape)
            print(attention_scores[:, :, -2, :])
            # Normalize the attention scores to probabilities.
            # `attention_probs` = [B, N, F, T]
            attention_probs = tf.nn.softmax(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            # attention_probs = self.attention_probs_dropout(attention_probs)

            # print('query: ', query_tensor.shape)
            # print(query_tensor)
            # print('key: ', key_tensor.shape)
            # print(key_tensor)
            # print('value: ', value_tensor.shape )
            # print(value_tensor)
            print("attention_probs: ", attention_probs.shape)
            print(attention_probs[:, :, -2, :])

            attention_output = atten.attention_layer(
                from_tensor=_x, to_tensor=_x, attention_mask=attention_mask
            )
            _x = atten(input_tensor=_x, attention_mask=attention_mask)
            _x = _x * ele_mask

        # print('after atten attention_output:')
        # print(tf.transpose(attention_output))
        # print('after atten ele_mask:')
        # print(tf.transpose(_x))
        emb = tf.reduce_mean(_x, axis=1)
        print("emb:")
        print(emb)
        print()


class EarlyStopCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        val_data,
        test_data,
        encoder_dir,
        model_config,
        opt_cp_path,
        num_parallel_calls=1,
        best_val_loss=1e10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.val_data = val_data
        self.test_data = test_data
        self.best_val_loss = best_val_loss
        self.encoder_dir = encoder_dir
        self.model_config = model_config
        self.num_parallel_calls = num_parallel_calls
        self.opt_cp_path = opt_cp_path

    def on_epoch_end(self, epoch, logs=None):
        print("the {}th epoch ends".format(epoch))
        print("logs", logs, logs.get("loss"))

        time_1 = datetime.datetime.now()

        # val_loss = self.model.predict(self.val_data)
        # val_loss = np.mean(val_loss)
        # print('val_loss old', val_loss)
        losses = self.get_loss_for_each_task(self.val_data)
        for i, task in enumerate(self.model.task_names):
            print("loss from task {}: {}".format(task, losses[i]))
        val_loss = self.get_total_loss(losses)
        print("val_loss", val_loss)

        if val_loss < self.best_val_loss:
            # find a better model
            # save model
            print("better model found in epoch {}!".format(epoch))
            print("last val_loss: ", self.best_val_loss)
            print("new val_loss: ", val_loss)
            self.best_val_loss = val_loss

            losses = self.get_loss_for_each_task(self.test_data)
            for i, task in enumerate(self.model.task_names):
                print("loss from task {}: {}".format(task, losses[i]))
            test_loss = self.get_total_loss(losses)
            print("test_loss", test_loss)

            export_model(self.encoder_dir, self.model_config, self.model)
            self.model.save_weights(self.opt_cp_path)
            if self.model.encoder_type not in {"attention", None}:
                print(
                    "zero_shift_layer", self.model.mat_encoder.zero_shift_layer.weights
                )

        time_2 = datetime.datetime.now()
        print("time cost in validation: ", time_2 - time_1)
        print()

    def get_loss_for_each_task(self, inputs):
        losses = []

        for task in self.model.task_names:
            tmp_loss = inputs.map(
                lambda x: self.model._loss_by_task[task](x, task),
                num_parallel_calls=self.num_parallel_calls,
            )
            tmp_loss = tmp_loss.map(tf.reduce_mean)
            tmp_loss = np.mean(list(tmp_loss))
            tmp_loss *= self.model._weight_by_task[task]
            losses.append(tmp_loss)

        return losses

    def get_total_loss(self, losses):
        losses = np.array(losses, dtype=np.float32)
        if self.model.use_adaptive_multi_loss and len(self.model.task_names) > 1:
            total_loss = self.model.adaptive_loss(losses)
            total_loss = total_loss.numpy()
        else:
            total_loss = np.sum(losses)
        return total_loss


class SimTestCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        all_elements,
        all_ions,
        eles_order=[],
        composition_pairs=[],
        ref_values=[],
        figure_dir=".",
        featurizer_type="default",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.all_elements = all_elements
        self.all_ions = all_ions
        self.composition_pairs = composition_pairs
        self.ref_values = ref_values
        self.eles_order = eles_order
        self.figure_dir = figure_dir
        self.featurizer_type = featurizer_type

        self.uni_vec = UnifyVector()

        self._f_id = 0

    def on_epoch_end(self, epoch, logs=None):
        print("the {}th epoch ends".format(epoch))
        print("logs", logs, logs.get("loss"))
        self.similarity_test(self.model, self.composition_pairs)
        print()

    def similarity_test(self, model, composition_pairs, to_print=True):
        time_1 = datetime.datetime.now()

        all_sims = []

        for pair in composition_pairs:
            has_error = False
            (comp_1, comp_2) = pair

            comps = []
            for x in pair:
                try:
                    comps.append(composition_to_array(x, self.all_elements))
                except:
                    has_error = True
            if has_error:
                all_sims.append(-100.0)
                continue
            comps = featurize_list_of_composition(
                comps=comps,
                ele_order=self.all_elements,
                featurizer_type=self.featurizer_type,
                ion_order=self.all_ions,
            )
            comps = np.array(comps)
            if isinstance(model, MultiTasksOnRecipes):
                v = model.get_mat_vector(comps)
            else:
                # is encoder model
                v = model(comps)
            # TODO: remove this reaction.prob_fn
            if (
                isinstance(model, MultiTasksOnRecipes)
                and "reaction" in model.task_names
            ):
                left_embs = v[0:1]
                right_embs = v[1:2]
                similarity = 1.0
                similarity = model._model_by_task["reaction"].prob_fn(
                    left_embs, right_embs
                )
                similarity = similarity[0]
            else:
                similarity = 1.0 - scipy.spatial.distance.cosine(v[0], v[1])

            if to_print:
                print(
                    "Similarity between {} and {} = {}".format(
                        dict_to_simple_formula(comp_1),
                        dict_to_simple_formula(comp_2),
                        similarity,
                    )
                )

            all_sims.append(similarity)
        time_2 = datetime.datetime.now()
        if to_print:
            print("time cost in SimTest: ", time_2 - time_1)
        return all_sims

    def plot_sim_against_ref(
        self, model, composition_pairs, ref_values, show_labels=True, show_fig=False
    ):
        pred_values = self.similarity_test(model, composition_pairs, to_print=False)
        # plotting:q

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        # plt.plot([0,2000], [0, 2000], color='gray', linewidth=2)
        plt.scatter(ref_values, pred_values, c="lightseagreen", alpha=1, s=50)
        if show_labels:
            texts = []
            for i, (x, y) in enumerate(zip(ref_values, pred_values)):
                (comp_1, comp_2) = composition_pairs[i]
                text = "{}\n{}".format(
                    dict_to_simple_formula(comp_1),
                    dict_to_simple_formula(comp_2),
                )
                texts.append(ax.text(x, y, text, size=20, color="black"))
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color="red"))
        plt.xlabel("log(probability of ion substitution)", size=40)
        plt.ylabel("cosine similarity", size=40)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.tick_params(axis="both", which="minor", labelsize=28)
        # plt.xlim(-5.4, -2.6)
        # plt.ylim(0, 2000)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(
            os.path.join(self.figure_dir, "scatter_{}.png".format(self._f_id)), dpi=100
        )
        self._f_id += 1
        if show_fig:
            plt.show()
        plt.close()

    def plot_final_matrix_crossing(
        self, model, task_name, layer_name, selected_precursors="all", show_fig=False
    ):
        assert task_name in {"mat", "reaction_pre"}
        if task_name == "mat":
            self.all_precursors = list(
                map(
                    lambda x: formula_to_array(x.decode("utf-8"), self.all_elements),
                    model._model_by_task[task_name].target_ele_labels.numpy(),
                )
            )
            self.num_reserved_ids = 0
            self.mat_labels = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=get_composition_string(self.all_precursors),
                    values=tf.range(len(self.all_precursors), dtype=tf.int64),
                ),
                default_value=tf.constant(-1, dtype=tf.int64),
                name="ele_labels",
            )
        elif task_name == "reaction_pre":
            self.all_precursors = model.mat_compositions[
                model.num_reserved_ids :
            ].numpy()
            self.num_reserved_ids = model.num_reserved_ids
            self.mat_labels = model.mat_labels

        assert isinstance(model, MultiTasksOnRecipes)
        pre_vectors = get_variables_by_name(
            getattr(model._model_by_task[task_name], layer_name),
            "{}/{}".format(layer_name, "kernel"),
        )
        assert len(pre_vectors) == 1
        pre_vectors = tf.transpose(pre_vectors[0])
        pre_vectors = pre_vectors.numpy()
        pre_normal_vectors = pre_vectors / np.linalg.norm(
            pre_vectors, axis=1, keepdims=True
        )
        matrix_crossing = pre_normal_vectors @ pre_normal_vectors.T

        assert len(self.all_precursors) == len(matrix_crossing)
        groups_by_ele_order = group_precursors(
            self.all_precursors, self.all_elements, self.eles_order
        )
        if selected_precursors == "all":
            selected_precursors = self.all_precursors
        if isinstance(selected_precursors[0], str):
            selected_precursors = list(
                map(
                    lambda x: formula_to_array(x, self.all_elements),
                    selected_precursors,
                )
            )
        selected_precursors = set(get_composition_string(selected_precursors).numpy())
        selected_groups = []
        for group in groups_by_ele_order:
            group_precursor_strings = get_composition_string(
                group["precursors"]
            ).numpy()
            tmp_group = list(
                filter(
                    lambda i: group_precursor_strings[i] in selected_precursors,
                    range(len(group_precursor_strings)),
                )
            )
            tmp_precursors = list(map(lambda i: group["precursors"][i], tmp_group))
            tmp_precursor_strings = list(
                map(lambda i: group_precursor_strings[i], tmp_group)
            )
            if len(tmp_group) > 0:
                tmp_indices = (
                    self.mat_labels.lookup(tf.constant(tmp_precursor_strings)).numpy()
                    - self.num_reserved_ids
                )
                selected_groups.append(
                    {
                        "ele": group["ele"],
                        "precursors": tmp_precursors,
                        "precursor_strings": tmp_precursor_strings,
                        "indices": tmp_indices,
                    }
                )
        selected_precursors_index = np.concatenate(
            [x["indices"] for x in selected_groups], axis=-1
        )
        matrix_crossing = matrix_crossing[selected_precursors_index, :][
            :, selected_precursors_index
        ]
        labels = sum([x["precursors"] for x in selected_groups], [])
        labels = list(map(lambda x: array_to_formula(x, self.all_elements), labels))
        sep_lines = [len(x["precursors"]) for x in selected_groups[:-1]]
        # True is valid, False is invalid
        mask = ~np.eye(len(matrix_crossing)).astype(np.bool)
        plot_heatmap(
            labels,
            matrix_crossing,
            sep_lines=sep_lines,
            mask=mask,
            title="{}:{}".format(task_name, layer_name),
            save_path=os.path.join(
                self.figure_dir, "heatmap_{}.png".format(self._f_id)
            ),
            show_fig=show_fig,
        )
        self._f_id += 1


class PredictElementsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        all_elements,
        mat_feature_len,
        test_data=None,
        output_thresh=0.0,
        featurizer_type="default",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.all_elements = all_elements
        self.num_eles = len(self.all_elements)
        self.mat_feature_len = mat_feature_len
        self.test_data = test_data
        if self.test_data is not None:
            self.reformat_reaction_pair_data(self.test_data)
        self.output_thresh = output_thresh
        self.featurizer_type = featurizer_type

    def reformat_reaction_pair_data(self, input_data):
        self.max_mats_num = input_data["reaction_1"].shape[1]
        # all_reactions: (batch_size, 2, max_mats_num, num_eles)
        all_reactions = tf.stack(
            [
                input_data["reaction_1"],
                input_data["reaction_2"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num, num_eles)
        all_reactions = tf.reshape(
            all_reactions, (-1, self.max_mats_num, self.num_eles)
        )

        # all_reactions_featurized: (batch_size, 2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.stack(
            [
                input_data["reaction_1_featurized"],
                input_data["reaction_2_featurized"],
            ],
            axis=1,
        )
        # all_reactions_featurized: (batch_size*2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.reshape(
            all_reactions_featurized, (-1, self.max_mats_num, self.mat_feature_len)
        )

        self.target_compositions = all_reactions[:, 0, :]
        self.target_compositions_featurized = all_reactions_featurized[:, 0, :]

    def on_epoch_end(self, epoch, logs=None):
        print("the {}th epoch ends".format(epoch))
        print("logs", logs, logs.get("loss"))
        self.predict_elements(
            self.model,
            self.target_compositions,
            target_features=self.target_compositions_featurized,
            to_print=True,
        )
        print()

    def predict_elements(
        self, model, target_compositions, target_features, to_print=True
    ):
        time_1 = datetime.datetime.now()

        assert isinstance(model, MultiTasksOnRecipes)
        assert "mat" in model.task_names
        assert len(target_compositions) > 0
        assert len(target_compositions) == len(target_features)

        ele_lists_pred = model._model_by_task["mat"].predict_elements(
            target_features, output_thresh=self.output_thresh
        )

        if to_print:
            for (i, tar_comp) in enumerate(target_compositions):
                ele_str_list = [
                    (ele["element"], ele["score"]) for ele in ele_lists_pred[i]
                ]
                ele_str_list = sorted(ele_str_list, key=lambda x: x[1], reverse=True)

                print("target: ", array_to_formula(tar_comp, self.all_elements))
                print(
                    "target normalized: ",
                    {
                        e: "{:.4}".format(v)
                        for (e, v) in zip(self.all_elements, tar_comp)
                        if v > 0
                    },
                )
                print("predicted elements: ", str(ele_str_list))
                print()

        time_2 = datetime.datetime.now()
        if to_print:
            print("time cost in PredictElements: ", time_2 - time_1)
        return ele_lists_pred


class PredictPrecursorsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        all_elements,
        mat_feature_len,
        all_ions=None,
        test_data=None,
        output_thresh=0.5,
        featurizer_type="default",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.all_elements = all_elements
        self.all_ions = all_ions
        self.num_eles = len(self.all_elements)
        self.mat_feature_len = mat_feature_len
        self.test_data = test_data
        if self.test_data is not None:
            self.reformat_reaction_pair_data(self.test_data)
        self.output_thresh = output_thresh
        # TODO: add featurization here to all offline call
        self.featurizer_type = featurizer_type

    def reformat_reaction_pair_data(self, input_data):
        self.max_mats_num = input_data["reaction_1"].shape[1]
        # all_reactions: (batch_size, 2, max_mats_num, num_eles)
        all_reactions = tf.stack(
            [
                input_data["reaction_1"],
                input_data["reaction_2"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num, num_eles)
        all_reactions = tf.reshape(
            all_reactions, (-1, self.max_mats_num, self.num_eles)
        )

        # all_reactions_featurized: (batch_size, 2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.stack(
            [
                input_data["reaction_1_featurized"],
                input_data["reaction_2_featurized"],
            ],
            axis=1,
        )
        # all_reactions_featurized: (batch_size*2, max_mats_num, mat_feature_len)
        all_reactions_featurized = tf.reshape(
            all_reactions_featurized, (-1, self.max_mats_num, self.mat_feature_len)
        )

        # all_reactions: (batch_size, 2, max_mats_num-1, num_eles)
        all_precursors_conditional = tf.stack(
            [
                input_data["precursors_1_conditional"],
                input_data["precursors_2_conditional"],
            ],
            axis=1,
        )
        # all_reactions: (batch_size*2, max_mats_num-1, num_eles)
        all_precursors_conditional = tf.reshape(
            all_precursors_conditional, (-1, self.max_mats_num - 1, self.num_eles)
        )

        # TODO: need to print all real precursors too if available
        self.target_compositions = all_reactions[:, 0, :]
        self.target_compositions_featurized = all_reactions_featurized[:, 0, :]
        self.precursors_compositions = all_reactions[:, 1:, :]
        self.precursors_conditional = all_precursors_conditional

    def on_epoch_end(self, epoch, logs=None):
        print("the {}th epoch ends".format(epoch))
        print("logs", logs, logs.get("loss"))
        if self.model._model_by_task["reaction_pre"].predict_precursor_under_mask:
            print("Precursor prediction with targets only")
            self.predict_precursors(
                self.model,
                self.target_compositions,
                target_features=self.target_compositions_featurized,
                precursors_conditional=tf.zeros_like(self.precursors_conditional),
                real_precursors=self.precursors_compositions,
                to_print=True,
            )
            print("Precursor prediction with targets and conditional precursors")
            self.predict_precursors(
                self.model,
                self.target_compositions,
                target_features=self.target_compositions_featurized,
                precursors_conditional=self.precursors_conditional,
                real_precursors=self.precursors_compositions,
                to_print=True,
            )
        else:
            print("Precursor prediction with targets only")
            self.predict_precursors(
                self.model,
                self.target_compositions,
                target_features=self.target_compositions_featurized,
                precursors_conditional=None,
                real_precursors=self.precursors_compositions,
                to_print=True,
            )
        print()

    def predict_precursors(
        self,
        model,
        target_compositions,
        target_features=None,
        precursors_conditional=None,
        real_precursors=None,
        to_print=True,
    ):
        time_1 = datetime.datetime.now()

        assert isinstance(model, MultiTasksOnRecipes)
        assert "reaction_pre" in model.task_names
        assert len(target_compositions) > 0
        if precursors_conditional is not None:
            assert len(target_compositions) == len(precursors_conditional)
        if real_precursors is not None:
            assert len(target_compositions) == len(real_precursors)

        if target_features is None:
            target_features = featurize_list_of_composition(
                comps=target_compositions,
                ele_order=self.all_elements,
                featurizer_type=self.featurizer_type,
                ion_order=self.all_ions,
            )
            target_features = np.array(target_features)

        assert len(target_compositions) == len(target_features)

        pre_lists_pred = model._model_by_task["reaction_pre"].predict_precursors(
            target_features,
            precursors_conditional=precursors_conditional,
            output_thresh=self.output_thresh,
        )
        pre_str_lists_pred = []
        for (i, tar_comp) in enumerate(target_compositions):
            pre_str_list = [
                (
                    array_to_formula(comp["composition"], self.all_elements),
                    comp["score"],
                )
                for comp in pre_lists_pred[i]
            ]
            # already sorted from model._model_by_task['reaction_pre'].predict_precursors
            # pre_str_list = sorted(pre_str_list, key=lambda x: x[1], reverse=True)
            pre_str_lists_pred.append(pre_str_list)

            if to_print:
                if precursors_conditional is not None:
                    pre_cond = []
                    for comp in precursors_conditional[i]:
                        if not np.any(np.not_equal(comp, 0)):
                            continue
                        pre_cond.append(array_to_formula(comp, self.all_elements))
                else:
                    pre_cond = None

                if real_precursors is not None:
                    pre_real = []
                    for comp in real_precursors[i]:
                        if not np.any(np.not_equal(comp, 0)):
                            continue
                        pre_real.append(array_to_formula(comp, self.all_elements))
                else:
                    pre_real = None

                print("target: ", array_to_formula(tar_comp, self.all_elements))
                print("real precursors: ", str(pre_real))
                print("conditional precursors: ", str(pre_cond))
                print("predicted precursors: ", str(pre_str_list))
                print()

        time_2 = datetime.datetime.now()
        if to_print:
            print("time cost in PredictPrecursors: ", time_2 - time_1)

        return pre_lists_pred, pre_str_lists_pred


class VectorMathCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        all_elements,
        all_ions=None,
        featurizer_type="default",
        max_mats_num=6,
        top_n=10,
        test_data=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.all_elements = all_elements
        self.all_ions = all_ions
        # TODO: add featurization here to all offline call
        self.featurizer_type = featurizer_type
        self.max_mats_num = max_mats_num
        self.top_n = top_n
        self.test_data = test_data

    def on_train_end(self, logs=None):
        print("Vector math similar to King - Male + Female = Queen")
        self.find_similar_targets(
            target_formulas=self.test_data["target_formulas"],
            target_candidate_formulas=self.test_data["target_candidate_formulas"],
            positive_formulas=self.test_data["positive_formulas"],
            negative_formulas=self.test_data["negative_formulas"],
            framework_model=self.model,
            mode=self.test_data["mode"],
        )
        print()

    def get_decoder(self, model):
        assert isinstance(model, MultiTasksOnRecipes)
        decoder = model._model_by_task["reaction_pre"]
        return decoder

    def get_pre_vec_mapping(self, decoder):

        # get pre_vec_mapping
        pre_vec_mapping = {}

        # decoder_atten = decoder.incomplete_reaction_atten_layer.attention_layer
        all_precursor_compositions = decoder.precursor_compositions[
            decoder.num_reserved_ids :
        ].numpy()
        # all_precursor_vecs: (num_precursors, latent_dim, )
        # precursor_kernel: (latent_dim, num_precursors, )
        all_precursor_vecs = tf.transpose(decoder.precursor_layer.kernel)
        all_precursor_vecs = all_precursor_vecs.numpy()
        for i in range(len(all_precursor_compositions)):
            formula = array_to_formula(all_precursor_compositions[i], self.all_elements)
            pre_vec_mapping[formula] = all_precursor_vecs[i]
        return pre_vec_mapping

    def get_pre_vecs_from_mapping(
        self,
        precursor_formulas: List[str],
        pre_vec_mapping,
    ):
        precursor_vecs = []
        for pre in precursor_formulas:
            # unify formula via formula -> comp -> formula
            comp = formula_to_array(pre, self.all_elements)
            formula = array_to_formula(comp, self.all_elements)
            precursor_vecs.append(pre_vec_mapping[formula])
        return precursor_vecs

    def get_tar_vec_mapping(
        self,
        target_formulas: List[str],
        model,
        project_w_attention=True,
        decoder=None,
        max_mats_num=6,
        target_features: Optional[list] = None,
    ):
        # get target_candidate_normal_vecs
        target_compositions = [
            formula_to_array(formula, self.all_elements) for formula in target_formulas
        ]
        if target_features is None:
            target_features = featurize_list_of_composition(
                comps=target_compositions,
                ele_order=self.all_elements,
                featurizer_type=self.featurizer_type,
                ion_order=self.all_ions,
            )
        target_features = np.array(target_features)
        # target_vecs: (num_tar_candidates, latent_dim)
        target_vecs = model.get_mat_vector(target_features)
        if project_w_attention:
            # precursors_conditional_emb: (None, max_mats_num-1, latent_dim)
            precursors_conditional_emb = tf.zeros(
                shape=(target_vecs.shape[0], max_mats_num - 1, target_vecs.shape[-1]),
                dtype=tf.float32,
            )
            # precursors_conditional_emb: (None, max_mats_num, latent_dim)
            incomplete_reaction_emb = tf.concat(
                (tf.expand_dims(target_vecs, axis=1), precursors_conditional_emb),
                axis=1,
            )
            # get mask (None, max_mats_num-1) in float
            precursors_conditional_mask = tf.zeros(
                shape=(target_vecs.shape[0], max_mats_num - 1), dtype=tf.float32
            )
            # incomplete_reaction_mask (None, max_mats_num) in float
            incomplete_reaction_mask = tf.concat(
                (
                    tf.ones(shape=(tf.shape(target_vecs)[0], 1), dtype=tf.float32),
                    precursors_conditional_mask,
                ),
                axis=1,
            )
            # attention_mask in 3d: (None, max_mats_num, max_mats_num) in float
            attention_mask = create_attention_mask_from_input_mask(
                from_tensor=incomplete_reaction_emb,
                to_mask=incomplete_reaction_mask,
            )
            # reactions_emb: (None, max_mats_num, latent_dim)
            reactions_emb = decoder.incomplete_reaction_atten_layer(
                input_tensor=incomplete_reaction_emb,
                attention_mask=attention_mask,
            )
            # reactions_emb: (None, latent_dim)
            target_vecs = reactions_emb[:, 0, :]

        target_vecs = target_vecs.numpy()
        target_normal_vecs = target_vecs / np.linalg.norm(
            target_vecs, axis=1, keepdims=True
        )
        target_vec_mapping = {k: v for (k, v) in zip(target_formulas, target_vecs)}
        return target_vec_mapping, target_normal_vecs

    def find_similar_targets(
        self,
        target_formulas: List[str],
        target_candidate_formulas: List[str],
        positive_formulas: List[List[str]],
        negative_formulas: List[List[str]],
        framework_model,
        top_n=10,
        mode="target",
        target_candidate_features: Optional[list] = None,
    ):
        assert len(target_formulas) == len(positive_formulas)
        assert len(target_formulas) == len(negative_formulas)

        # get decoder
        decoder = None
        project_w_attention = False
        if mode == "precursor":
            decoder = self.get_decoder(framework_model)
            if decoder.predict_precursor_under_mask:
                project_w_attention = True

        # get vecs for target_candidate
        (
            target_candidate_vec_mapping,
            target_candidate_normal_vecs,
        ) = self.get_tar_vec_mapping(
            target_formulas=target_candidate_formulas,
            model=framework_model,
            project_w_attention=project_w_attention,
            decoder=decoder,
            max_mats_num=self.max_mats_num,
            target_features=target_candidate_features,
        )

        # get vecs for input targets
        # get target compositions, target_positive_compositions, target_negative_compositions as a batch
        target_input_formulas = []
        target_input_formulas += target_formulas
        if mode == "target":
            target_input_formulas += sum(positive_formulas, [])
            target_input_formulas += sum(negative_formulas, [])
        target_input_vec_mapping, target_input_normal_vecs = self.get_tar_vec_mapping(
            target_formulas=target_input_formulas,
            model=framework_model,
            project_w_attention=project_w_attention,
            decoder=decoder,
            max_mats_num=self.max_mats_num,
        )

        if mode == "precursor":
            pre_vec_mapping = self.get_pre_vec_mapping(
                decoder=decoder,
            )
            for i in range(len(target_formulas)):
                similar_ranking = vector_utils.most_similar_by_vector(
                    target_vec=target_input_vec_mapping[target_formulas[i]],
                    target_candidate_formulas=target_candidate_formulas,
                    target_candidate_normal_vecs=target_candidate_normal_vecs,
                    positive_vecs=self.get_pre_vecs_from_mapping(
                        precursor_formulas=positive_formulas[i],
                        pre_vec_mapping=pre_vec_mapping,
                    ),
                    negative_vecs=self.get_pre_vecs_from_mapping(
                        precursor_formulas=negative_formulas[i],
                        pre_vec_mapping=pre_vec_mapping,
                    ),
                    top_n=top_n,
                )
                print("target: ", target_formulas[i])
                print("precursor positive", positive_formulas[i])
                print("precursor negative", negative_formulas[i])
                for j in range(len(similar_ranking)):
                    print(j, similar_ranking[j])
                print()

        elif mode == "target":
            for i in range(len(target_formulas)):
                similar_ranking = vector_utils.most_similar_by_vector(
                    target_vec=target_input_vec_mapping[target_formulas[i]],
                    target_candidate_formulas=target_candidate_formulas,
                    target_candidate_normal_vecs=target_candidate_normal_vecs,
                    positive_vecs=[
                        target_input_vec_mapping[formula]
                        for formula in positive_formulas[i]
                    ],
                    negative_vecs=[
                        target_input_vec_mapping[formula]
                        for formula in negative_formulas[i]
                    ],
                    top_n=top_n,
                )
                print("target: ", target_formulas[i])
                print("target positive", positive_formulas[i])
                print("target negative", negative_formulas[i])
                for j in range(len(similar_ranking)):
                    print(j, similar_ranking[j])
                print()
        else:
            raise NotImplementedError

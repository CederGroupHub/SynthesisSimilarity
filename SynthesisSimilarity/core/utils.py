import json
import os
import psutil
import regex
import tensorflow as tf
import numpy as np
import random
import sys
import re
import collections
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from matplotlib.colors import ListedColormap
from pymatgen.core import Composition
from pprint import pprint
from typing import List, Tuple, Union
import pkgutil

if pkgutil.find_loader("ValenceSolver"):
    from ValenceSolver.core.composition_inhouse import CompositionInHouse


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


NEAR_ZERO = 1e-6
allNonMetalElements = ["C", "H", "O", "N", "Cl", "F", "P", "S", "Br", "I", "Se"] + [
    "He",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
    "Rn",
    "Og",
]
valence_cache = {}
pattern_prototype = regex.compile(r"(.*)/[^/]+<-.*|(.*)/NoDoping.*")


def print_gpu_info():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def allow_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        num_python_proc = 0
        path = os.path.abspath(".")
        tmp_m = re.match(".*TPSimilarity_([0-9]+).*", path)
        if tmp_m:
            num_python_proc = int(tmp_m.group(1))
        else:
            for proc in psutil.process_iter():
                if "python" in proc.name():
                    num_python_proc += 1
        try:
            # Currently, memory growth needs to be the same across GPUs
            gpu_to_use = gpus[num_python_proc % len(gpus)]
            tf.config.experimental.set_visible_devices(gpu_to_use, "GPU")
            tf.config.experimental.set_memory_growth(gpu_to_use, True)
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if os.environ.get("tf_allow_gpu_growth", "False") != "True":
    allow_gpu_growth()
    os.environ["tf_allow_gpu_growth"] = "True"


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def use_file_as_stdout(file_path):
    sys.stdout = open(file_path, "w")
    sys.stdout = Unbuffered(sys.stdout)
    print("this is printed in the console")


def composition_to_array(composition, elements):
    """

    :param composition: a dict
    :param elements: a list
    :return:
    """
    comp_array = np.zeros((len(elements),), dtype=np.float32)
    for c, v in composition.items():
        comp_array[elements.index(c)] = v
    comp_array /= max(np.sum(comp_array), NEAR_ZERO)
    return comp_array


def formula_to_array(formula, elements):
    """

    :param formula: str
    :param elements: a list
    :return:
    """
    comp = Composition(formula).as_dict()
    return composition_to_array(comp, elements)


def array_to_composition(comp_array, elements):
    composition = dict(filter(lambda x: x[1] > 0, zip(elements, comp_array)))
    return composition


def array_to_formula(comp_array, elements):
    composition = array_to_composition(comp_array, elements)
    formula = dict_to_simple_formula(composition)
    return formula


def get_composition_string(materials):
    """
    convert ordered composition to string as mat name
    :param materials: (None, num_eles)
    :return comp_str: (None, )
    """
    comp_str = tf.strings.as_string(materials, precision=6)
    comp_str = tf.strings.reduce_join(comp_str, separator=" ", axis=-1)
    return comp_str


def get_elements_in_formula(mat_str):
    comp = Composition(mat_str)
    return [str(ele) for ele in comp.elements]


def convert_list_to_dico(
    all_labels: Union[List[str], List[bytes]],
    count_weights: Union[List[int], List[float]],
    num_reserved_ids: int = 10,
    least_count: int = 5,
):
    """
    convert a list of composition dict to a dico used in the ReactionContext task

    :param all_compositions:
    :return:
    """
    # goal
    out_labels = []
    out_counts = []

    if count_weights is None:
        count_weights = [1.0] * len(all_labels)
    else:
        assert len(count_weights) == len(all_labels)

    # get big_dico of all mats
    big_dico = collections.Counter()
    for i in range(len(all_labels)):
        w = count_weights[i]
        s = all_labels[i]
        if not isinstance(s, bytes):
            s = bytes(s, encoding="utf-8")
        big_dico[s] += w
    big_dico = big_dico.most_common()
    big_dico = list(filter(lambda x: x[1] >= least_count, big_dico))
    print("len(big_dico)", len(big_dico))

    if num_reserved_ids >= 2:
        # add some placeholder for flexible usage
        placeholder = ["<PLACEHOLDER>_{}".format(i) for i in range(num_reserved_ids)]
        placeholder[0] = "<MASK>"
        placeholder[1] = "<UNK>"
        for p in reversed(placeholder):
            big_dico.insert(
                0,
                (
                    bytes(p, encoding="utf-8"),
                    0,
                ),
            )

    # convert big_dico to objects required by MultiTasksOnRecipes model
    out_labels = [x[0] for x in big_dico]
    out_counts = [x[1] for x in big_dico]

    return out_labels, out_counts


def get_input_format(model_type="MultiTasksOnRecipes", max_mats_num=6):
    data_type = None
    data_shape = None
    padded_data_shape = None
    if model_type == "MultiTasksOnRecipes":
        data_type = {
            "reaction_1": tf.float32,
            "reaction_2": tf.float32,
            "reaction_1_featurized": tf.float32,
            "reaction_2_featurized": tf.float32,
            "precursors_1_conditional": tf.float32,
            "precursors_2_conditional": tf.float32,
            "temperature_1": tf.float32,
            "temperature_2": tf.float32,
            "synthesis_type_1": tf.string,
            "synthesis_type_2": tf.string,
        }

        data_shape = {
            "reaction_1": tf.TensorShape([None, None]),
            "reaction_2": tf.TensorShape([None, None]),
            "reaction_1_featurized": tf.TensorShape([None, None]),
            "reaction_2_featurized": tf.TensorShape([None, None]),
            "precursors_1_conditional": tf.TensorShape([None, None]),
            "precursors_2_conditional": tf.TensorShape([None, None]),
            "temperature_1": tf.TensorShape([]),
            "temperature_2": tf.TensorShape([]),
            "synthesis_type_1": tf.TensorShape([]),
            "synthesis_type_2": tf.TensorShape([]),
        }

        padded_data_shape = {
            "reaction_1": tf.TensorShape([max_mats_num, None]),
            "reaction_2": tf.TensorShape([max_mats_num, None]),
            "reaction_1_featurized": tf.TensorShape([max_mats_num, None]),
            "reaction_2_featurized": tf.TensorShape([max_mats_num, None]),
            "precursors_1_conditional": tf.TensorShape([max_mats_num - 1, None]),
            "precursors_2_conditional": tf.TensorShape([max_mats_num - 1, None]),
            "temperature_1": tf.TensorShape([]),
            "temperature_2": tf.TensorShape([]),
            "synthesis_type_1": tf.TensorShape([]),
            "synthesis_type_2": tf.TensorShape([]),
        }
    return data_type, data_shape, padded_data_shape


def dict_to_tf_dataset(
    data, data_type, data_shape, padded_shape=None, column_y=None, batch_size=1
):
    """

    :param data: list of dict contain all different types of data,
            all variables should be numerical numbers
    :param data_type: tensorflow data type of the numerical numbers
    :param data_shape: tensorflow shape of the values data
    :param column_y: name of y columns.
            If not specified, the value of dataset_y returned
            would be zeros with the same length of X.
            It seems that tf v2.0 only support y as a array rather than a dict.
    :param batch_size: size of batch
    :return: dataset_x: dataset formatted from dict of features
    :return: dataset_y: dataset formatted from array of y
    """
    features = set(data[0].keys()) & set(data_type.keys())
    if column_y:
        features -= {
            column_y,
        }
    features_type = {k: data_type[k] for k in features}
    features_shape = {k: data_shape[k] for k in features}
    if padded_shape == None:
        padded_shape = data_shape
    padded_features_shape = {k: padded_shape[k] for k in features}

    # another way to generate dict is to use dataset.map
    # https://github.com/tensorflow/tensorflow/issues/28643
    def feature_dict_gen():
        for d in data:
            feature_dict = {}
            for k, v in d.items():
                if k not in features:
                    continue
                feature_dict[k] = v
            # print('generted x sample')
            yield feature_dict

    def y_array_gen():
        for d in data:
            assert column_y in d
            # print('generted y sample')
            yield d[column_y]

    # or use dataset.map
    # https://github.com/tensorflow/tensorflow/issues/28643
    dataset_x = tf.data.Dataset.from_generator(
        feature_dict_gen,
        output_types=features_type,
        output_shapes=features_shape,
    )
    if column_y:
        dataset_y = tf.data.Dataset.from_generator(
            y_array_gen,
            output_types=data_type[column_y],
            output_shapes=data_shape[column_y],
        )
    else:
        dataset_y = tf.data.Dataset.from_tensor_slices(
            np.zeros((len(data),), dtype=np.float32)
        )

    data_batch_x = dataset_x.padded_batch(
        batch_size, padded_shapes=padded_features_shape
    )
    if column_y:
        data_batch_y = dataset_y.padded_batch(
            batch_size, padded_shapes=padded_features_shape[column_y]
        )
    else:
        data_batch_y = dataset_y.padded_batch(batch_size, padded_shapes=[])
    return data_batch_x, data_batch_y


def split_reactions(
    reactions,
    val_frac=0.1,
    test_frac=0.1,
    keys=("raw_index",),
    random_seed=None,
    by_year=False,
):
    def default_key_exists(key_default: Union[str, int], key_records):
        if key_default in key_records:
            return True
        else:
            return False

    def composition_exists(composition, composition_records):
        result = False

        comp_str = get_composition_string(composition).numpy()

        if len(comp_str.shape) == 0:
            # is one composition
            if comp_str in composition_records:
                result = True
        elif len(comp_str.shape) == 1:
            # is a list of compositions
            for comp_s in comp_str:
                if comp_s in composition_records:
                    result = True
        else:
            raise NotImplementedError

        return result

    def prototype_exists(prototype_paths, prototype_records):
        result = False

        for p_p in prototype_paths:
            prototype = get_prototype(p_p)
            if prototype in prototype_records:
                result = True

        return result

    def default_key_save(key_default, key_records):
        key_records.add(key_default)

    def composition_save(composition, composition_records):
        comp_str = get_composition_string(composition).numpy()

        if len(comp_str.shape) == 0:
            # is one composition
            composition_records.add(comp_str)
        elif len(comp_str.shape) == 1:
            # is a list of compositions
            for comp_s in comp_str:
                composition_records.add(comp_s)
        else:
            raise NotImplementedError

    def prototype_save(prototype_paths, prototype_records):
        for p_p in prototype_paths:
            prototype = get_prototype(p_p)
            prototype_records.add(prototype)

    func_key_exists = {
        "default": default_key_exists,
        "target_comp": composition_exists,
        "prototype_path": prototype_exists,
    }
    func_key_save = {
        "default": default_key_save,
        "target_comp": composition_save,
        "prototype_path": prototype_save,
    }

    # rank reactions first to prevent leakage of key reactions
    # Because the reactions are expanded, one raw_index could
    # be converted to several reactions.
    # If one target is recorded as a key, other reactions corresponding
    # to the raw index may not be records.
    # Then those targets are missed.
    # If other raw indexed reactions has one of those targets,
    # then the target family are recorded again.
    # TODO: doi and raw index should always exist. change to assert
    assert "year" in reactions[0]
    assert "doi" in reactions[0]
    assert "raw_index" in reactions[0]
    assert "target_comp" in reactions[0]
    assert "prototype_path" in reactions[0]
    reactions = sorted(reactions, key=lambda x: (x["year"], x["doi"], x["raw_index"]))

    # get all reaction keys
    reactions_keys = {k: set() for k in keys}
    reactions_key_indices = []
    for i, r in enumerate(reactions):
        any_key_exists = False
        for k in keys:
            any_key_exists = any_key_exists | func_key_exists.get(
                k, func_key_exists["default"]
            )(r[k], reactions_keys[k])
            func_key_save.get(k, func_key_save["default"])(r[k], reactions_keys[k])

        if not any_key_exists:
            reactions_key_indices.append(i)

    # splitting by keys
    if random_seed is not None:
        random_gen = random.Random(random_seed)
    else:
        random_gen = random
    if by_year == True:
        all_years = sorted(set([reactions[i]["year"] for i in reactions_key_indices]))
        train_years, val_years, test_years = data_list_split(
            all_years,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        print("train_years", train_years)
        print("val_years", val_years)
        print("test_years", test_years)
        train_years = set(train_years)
        train_key_indices = list(
            filter(lambda i: reactions[i]["year"] in train_years, reactions_key_indices)
        )
        val_years = set(val_years)
        val_key_indices = list(
            filter(lambda i: reactions[i]["year"] in val_years, reactions_key_indices)
        )
        test_years = set(test_years)
        test_key_indices = list(
            filter(lambda i: reactions[i]["year"] in test_years, reactions_key_indices)
        )
    else:
        random_gen.shuffle(reactions_key_indices)
        train_key_indices, val_key_indices, test_key_indices = data_list_split(
            reactions_key_indices,
            val_frac=val_frac,
            test_frac=test_frac,
        )

    # collect split reactions by key
    # Attention: key need to be expanded first
    # For example, it is possible only one composition from a raw index is recorded
    # We need to expand to all composition for the same raw index first.
    # Then, use the new composition set as key so that the same composition
    # would all go to the same set.
    # Otherwise, we might miss reactions because of the strategy of saving keys
    reaction_index_selected = set()
    train_val_test_sets = []
    for key_indices_to_select in [
        train_key_indices,
        val_key_indices,
        test_key_indices,
    ]:
        # get selected_keys from reactions w/ keys
        selected_keys = {k: set() for k in keys}

        # get selected_reactions
        selected_reactions = []
        last_len_selected_reactions = -1
        while len(selected_reactions) > last_len_selected_reactions:
            # record last_len_selected_reactions.
            # If len(selected_reactions) increased after the loop because of expanded keys,
            # do the loop again until no more reations are added in
            last_len_selected_reactions = len(selected_reactions)
            for i, r in enumerate(reactions):
                if i in reaction_index_selected:
                    continue

                to_select = False
                if i in key_indices_to_select:
                    to_select = True
                for k in keys:
                    if func_key_exists.get(k, func_key_exists["default"])(
                        r[k], selected_keys[k]
                    ):
                        to_select = True
                        break

                if to_select:
                    reaction_index_selected.add(i)
                    selected_reactions.append(r)
                    for k in keys:
                        func_key_save.get(k, func_key_save["default"])(
                            r[k], selected_keys[k]
                        )
        train_val_test_sets.append(selected_reactions)

    print("split_reactions input_data: ", len(reactions))
    print("split_reactions train: ", len(train_val_test_sets[0]))
    print("split_reactions val: ", len(train_val_test_sets[1]))
    print("split_reactions test: ", len(train_val_test_sets[2]))

    assert len(reaction_index_selected) == len(
        reactions
    ), "Reaction might be missed in data spliting!"

    return tuple(train_val_test_sets)


def data_list_split(input_data, val_frac=0.1, test_frac=0.1):
    """

    :param input_data: a list
    :param val_frac: float between (0, 1)
    :param test_frac: float between (0, 1)
    :return:
    """
    data_len = len(input_data)
    train = input_data[: int(data_len * (1 - val_frac - test_frac))]
    val = input_data[
        int(data_len * (1 - val_frac - test_frac)) : int(data_len * (1 - test_frac))
    ]
    test = input_data[int(data_len * (1 - test_frac)) :]

    print("data_list_split input_data: ", data_len)
    print("data_list_split train: ", len(train))
    print("data_list_split val: ", len(val))
    print("data_list_split test: ", len(test))

    return train, val, test


def random_drop_in_list(input_data: List, sample_shape: Union[Tuple, List], drop_n=0):
    if drop_n < 0:
        # TODO: why len(input_data) could be zero sometimes?
        drop_n = random.randint(1, max(min(-drop_n, len(input_data)), 1))
    samples = random.sample(input_data, max(len(input_data) - drop_n, 0))
    # Zero-length array cannot be padding without explicit expanded dimension
    # Therefore, a default zero list is appended when no sample exists
    if len(samples) == 0:
        samples.append(np.zeros(shape=sample_shape, dtype=np.float32))
    return samples


def repeat_in_last_dimension(from_tensor, from_seq_length=None, to_latent_dim=None):
    """Create 3D weight/mask array from a 2D tensor.

    Args:
    from_tensor: int32 Tensor of shape [batch_size, from_seq_length].

    Returns:
    float Tensor of shape [batch_size, from_seq_length, 1].
    could be broadcast in multiplication with [batch_size, from_seq_length, to_latent_dim].
    """

    # repeated_tensor = tf.reshape(
    #     tf.tile(
    #         tf.reshape(from_tensor, (-1, 1)),
    #         multiples=[1, to_latent_dim]
    #     ),
    #     (-1, from_seq_length, to_latent_dim)
    # )
    repeated_tensor = tf.expand_dims(from_tensor, 2)
    return repeated_tensor


def get_mat_mask_in_mat_seq(mat_seq, dtype=None):
    """
    masked position is False.
    True represents a valid material
    or
    masked position is 0.
    1 represents a valid material
    :param mat_seq: (None, num_eles) or (None, None, num_eles)
    :param dtype: if None, use bool as default.
    :return mat_mask: (None, ) or or (None, None, )
    """
    mat_mask = tf.not_equal(mat_seq, 0)
    mat_mask = tf.reduce_any(mat_mask, axis=-1)
    if dtype is not None:
        mat_mask = tf.cast(mat_mask, dtype)
    return mat_mask


def get_combination_pairs(batch_data, to_shape, data_to_use=[]):
    """
    get combination along axis=1
    :param batch_data: (None, repeat_times, dim) or (None, repeat_times, )
    :param to_shape: (-1, dim) or (-1, )
    :param data_to_use: empty list of list of two elements [start, end]. declare the
                which columns of combination matrix to use
    :return pairs_left: (None*repeat_times*repeat_times, dim)
                or (None*repeat_times*repeat_times, )
                or (None*repeat_times*(data_to_use[1]-data_to_use[0]), dim)
                or (None*repeat_times*(data_to_use[1]-data_to_use[0]), )
            pairs_right: (None*repeat_times*repeat_times, dim)
                or (None*repeat_times*repeat_times, )
                or (None*repeat_times*(data_to_use[1]-data_to_use[0]), dim)
                or (None*repeat_times*(data_to_use[1]-data_to_use[0]), )
    """
    repeat_times = batch_data.shape[1]
    dim_num = len(batch_data.shape)
    repeat_shape_left = np.ones(dim_num + 1).astype(np.int32)
    repeat_shape_left[1] = repeat_times
    repeat_shape_right = np.ones(dim_num + 1).astype(np.int32)
    repeat_shape_right[2] = repeat_times

    # pairs_left: (None, repeat_times, max_mats_num, num_eles)
    # ((m1, m2, m3)) ->
    # (
    #     (m1, m2, m3)
    #     (m1, m2, m3)
    #     (m1, m2, m3)
    # )
    pairs_left = tf.tile(tf.expand_dims(batch_data, 1), repeat_shape_left)
    if len(data_to_use) == 2:
        # pairs_left: (None, repeat_times, data_to_use[1]-data_to_use[0], num_eles)
        pairs_left = pairs_left[:, :, data_to_use[0] : data_to_use[1]]
    # pairs_left: (None*repeat_times*max_mats_num, num_eles)
    #     or (None*repeat_times*(data_to_use[1]-data_to_use[0]), num_eles)
    pairs_left = tf.reshape(pairs_left, to_shape)

    # pairs_right: (None, max_mats_num, repeat_times, num_eles)
    # ((m1, m2, m3)) ->
    # (
    #     (m1, m1, m1)
    #     (m2, m2, m2)
    #     (m3, m3, m3)
    # )
    pairs_right = tf.tile(tf.expand_dims(batch_data, 2), repeat_shape_right)
    if len(data_to_use) == 2:
        # pairs_right: (None, max_mats_num, data_to_use[1]-data_to_use[0], num_eles)
        pairs_right = pairs_right[:, :, data_to_use[0] : data_to_use[1]]
    # pairs_right: (None*max_mats_num*repeat_times, num_eles)
    #     or (None*max_mats_num*(data_to_use[1]-data_to_use[0]), num_eles)
    pairs_right = tf.reshape(pairs_right, to_shape)

    return pairs_left, pairs_right


def get_mat_pairs_in_reaction(reactions, mat_mask, data_to_use=[]):
    """

    :param reactions: (None, max_mats_num, num_eles)
    :param mat_mask: (None, max_mats_num)
    :return mat_pairs_left: (None*max_mats_num*max_mats_num, num_eles)
                or (None*max_mats_num*(data_to_use[1]-data_to_use[0]), num_eles)
            mat_pairs_right: (None*max_mats_num*max_mats_num, num_eles)
                or (None*max_mats_num*(data_to_use[1]-data_to_use[0]), num_eles)
            pair_mask: (None*max_mats_num*max_mats_num, )
                or (None*max_mats_num*(data_to_use[1]-data_to_use[0]),)
    """
    # repeat_times = max_mats_num
    repeat_times = tf.shape(reactions)[1]
    num_eles = tf.shape(reactions)[2]
    # mat_pairs_left: (None, repeat_times, max_mats_num, num_eles)
    # mat_pairs_right: (None*max_mats_num*repeat_times, num_eles)
    mat_pairs_left, mat_pairs_right = get_combination_pairs(
        reactions,
        to_shape=(-1, num_eles),
        data_to_use=data_to_use,
    )

    # mat_pairs_left: (None, repeat_times, max_mats_num)
    pair_mask_left = tf.tile(
        tf.expand_dims(mat_mask, 1),
        [
            1,
            repeat_times,
            1,
        ],
    )
    # pair_mask_right: (None, max_mats_num, repeat_times)
    pair_mask_right = tf.tile(
        tf.expand_dims(mat_mask, 2),
        [
            1,
            1,
            repeat_times,
        ],
    )
    # pair_mask: (None, max_mats_num, max_mats_num)
    pair_mask = pair_mask_left * pair_mask_right
    pair_mask = tf.linalg.set_diag(
        pair_mask,
        tf.zeros(tf.shape(pair_mask)[:-1]),
    )
    if len(data_to_use) == 2:
        # pair_mask: (None, max_mats_num, data_to_use[1]-data_to_use[0])
        pair_mask = pair_mask[:, :, data_to_use[0] : data_to_use[1]]
    # pair_mask: (None*max_mats_num*max_mats_num)
    #   or (None * max_mats_num * (data_to_use[1]-data_to_use[0]),)
    pair_mask = tf.reshape(pair_mask, (-1,))

    mat_pairs_left = mat_pairs_left * tf.expand_dims(pair_mask, 1)
    mat_pairs_right = mat_pairs_right * tf.expand_dims(pair_mask, 1)

    return mat_pairs_left, mat_pairs_right, pair_mask


def get_mat_label(materials, mat_labels_lookup):
    """

    :param materials: (None, num_eles)
    :param mat_labels_lookup: tf lookup of
        {composition_string: label}
    :return labels: (None, ) dtype defined by input lookup table
    """
    comp_str = get_composition_string(materials)
    labels = mat_labels_lookup.lookup(comp_str)
    return labels


def ordereddict_to_simple_text(input_dict):
    simple_text = ""
    for k, v in input_dict.items():
        simple_text += str(k) + str(v)
    return simple_text


def dict_to_simple_formula(input_dict):
    input_dict = {k: float(v) for (k, v) in input_dict.items()}
    comp = Composition(input_dict)
    if len(comp) == 0:
        return None
    else:
        # Attention: use a large max_denominator here because some compsition has trace elements (<0.001)
        comp, inte_factor = comp.get_integer_formula_and_factor(
            max_denominator=1000000,
        )
        return comp


def group_precursors(all_precursors, all_elements, eles_order):
    """

    :param all_precursors: list of array, each array is a representation
            of a precursor composition
    :param all_elements: length same as each precursor array
    :param eles_order: list of elements. order of elements to follow
    :return: list of precursor groups following eles_order
    """
    groups_by_ele_order = []
    eles_separated = set(eles_order) | set(allNonMetalElements)
    eles_order = (
        eles_order
        + list(filter(lambda x: x not in eles_separated, all_elements))
        + allNonMetalElements
    )
    eles_order = list(filter(lambda x: x in all_elements, eles_order))
    precursors_grouped_index = set()
    for ele in eles_order:
        ele_index = all_elements.index(ele)
        tmp_group = list(
            filter(
                lambda i: (
                    all_precursors[i][ele_index] > 0
                    and i not in precursors_grouped_index
                ),
                range(len(all_precursors)),
            )
        )
        if len(tmp_group) > 0:
            groups_by_ele_order.append(
                {
                    "ele": ele,
                    "precursors": list(map(lambda i: all_precursors[i], tmp_group)),
                }
            )
            precursors_grouped_index.update(tmp_group)
    return groups_by_ele_order


def plot_heatmap(
    precursors,
    matrix_crossing,
    sep_lines=[],
    mask=None,
    title="",
    save_path="",
    show_fig=False,
):
    # plot heatmaps
    df = pd.DataFrame(columns=["P1", "P2", "crossing_term"])
    for i in range(len(matrix_crossing)):
        for j in range(len(matrix_crossing[i])):
            dfa = pd.DataFrame(
                [[precursors[i], precursors[j], float(matrix_crossing[i][j])]],
                columns=["P1", "P2", "crossing_term"],
            )
            df = df.append(dfa)

    df["P1"] = pd.Categorical(df["P1"], precursors)
    df["P2"] = pd.Categorical(df["P2"], precursors)
    df = df.pivot("P1", "P2", "crossing_term")
    if mask is not None:
        # seaborn use a reversed mask, 1 is to mask (invalid)
        mask = ~mask

    fig = plt.figure(figsize=(12, 10))
    subplot1 = fig.add_subplot(111)
    cmap = ListedColormap(cc.b_diverging_rainbow_bgymr_45_85_c67)
    sns.set_style("whitegrid")

    # g = sns.heatmap(df, mask=mask, cmap=cmap, vmin=-5, vmax=5,
    #                 cbar_kws={"shrink": 1.0, 'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]})
    # g = sns.heatmap(df, mask=mask, cmap=cmap, vmin=-1, vmax=1, cbar_kws={"shrink": 1.0, })
    g = sns.heatmap(
        df,
        mask=mask,
        cmap=cmap,
        cbar_kws={
            "shrink": 1.0,
        },
    )

    # title things
    # g.set_title('Substitution of precursors', size=20)
    # g.set_xlabel('Precursor A', size=32)
    # g.set_ylabel('Precursor B', size=32)
    g.set_xlabel("", size=32)
    g.set_ylabel("", size=32)
    plt.setp(g.get_xticklabels(), rotation=90, size=28)
    plt.setp(g.get_yticklabels(), rotation=0, size=28)

    cbar = g.collections[0].colorbar
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=28)
    # cbar.ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=28)
    if len(title) > 0:
        cbar.set_label(title, size=28)
    plt.tight_layout()
    if len(save_path) > 0:
        fig.savefig(save_path, dpi=100)
    if show_fig:
        plt.show()
    plt.close()


def get_material_valence_details(composition: dict):
    # goal
    oxi_details = None

    formula = dict_to_simple_formula(composition)
    if formula in valence_cache:
        oxi_details = valence_cache[formula]
    else:
        (
            oxi_state,
            is_usual,
            comments,
            oxi_details,
        ) = CompositionInHouse.get_most_possible_oxi_state_of_composition(
            composition,
            return_details=True,
        )
        if len(oxi_details) == 1:
            oxi_details = oxi_details[0]
            oxi_details = {
                ion: oxi_details[ion]
                for ion in list(oxi_details.keys())
                if oxi_details[ion] > 0
            }
        else:
            oxi_details = None
        # store in cache:
        valence_cache[formula] = oxi_details
    return oxi_details


def valence_to_array(mat_ion, ion_order):
    """

    :param mat_ion: a dict
    :param ion_order: a list
    :return:
    """
    ion_array = np.zeros((len(ion_order),), dtype=np.float32)
    ion_sum = max(sum(mat_ion.values()), NEAR_ZERO)
    for i, ion in enumerate(ion_order):
        if ion not in mat_ion:
            continue
        ion_array[i] = mat_ion[ion] / ion_sum
    return ion_array


def get_prototype(prototype_path):
    tmp_m = pattern_prototype.match(prototype_path)
    if tmp_m:
        prototype = tmp_m.group(1) or tmp_m.group(2)
    else:
        prototype = prototype_path
    return prototype

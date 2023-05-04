import itertools
import random
from pprint import pprint
import warnings
import numpy as np
import tensorflow as tf
from pymatgen.core import Composition

from SynthesisSimilarity.scripts_utils import train_utils
from SynthesisSimilarity.core import mat_featurization
from SynthesisSimilarity.core import utils


def load_and_generate_test_set(
    reload_path="generated/data.npz",
    featurizer_type="default",
    random_seed_str=None,
    all_elements=None,
    all_ions=None,
    ion_freq_threshold=0,
    common_ion_not_feature=False,
):
    #########################################
    # load data
    #########################################
    eles, reactions, ions, ion_counter = train_utils.load_synthesis_data(
        data_path=None,
        reload_path=reload_path,
        max_mats_num=6,
        reload=True,
    )
    reactions, ions = train_utils.truncate_valence_array_in_reactions(
        reactions=reactions,
        all_ions=ions,
        ion_counter=ion_counter,
        ion_freq_threshold=ion_freq_threshold,
        common_ion_not_feature=common_ion_not_feature,
    )

    if all_elements is not None:
        if all_elements != eles:
            warnings.warn(
                "all_elements (from model) != eles (from dataset)!"
                "Check the dataset version to make sure they are "
                "are consistent with each other! Temporarily, "
                "all_elements from model is used. "
            )
            # map indices of eles to all_elements
            assert len(eles) == len(all_elements)
            mapping_indices = [eles.index(e) for e in all_elements]
            for r in reactions:
                for i in range(len(r["target_comp"])):
                    r["target_comp"][i] = r["target_comp"][i][mapping_indices]
                for i in range(len(r["precursors_comp"])):
                    for j in range(len(r["precursors_comp"][i])):
                        r["precursors_comp"][i][j] = r["precursors_comp"][i][j][
                            mapping_indices
                        ]
            eles = all_elements

    if all_ions is not None:
        assert all_ions == ions
        # TODO: add code to allow all_ions to be a subset of ions

    reactions, mat_feature_len = mat_featurization.featurize_reactions(
        reactions,
        ele_order=eles,
        featurizer_type=featurizer_type,
        ion_order=ions,
    )
    print("mat_feature_len", mat_feature_len)

    # random seed
    if random_seed_str:
        print("random_seed_str", random_seed_str)
        random_seed = sum(map(ord, random_seed_str.strip()))
    else:
        random_seed = None

    # split data to train/val/test sets
    train_reactions, val_reactions, test_reactions = utils.split_reactions(
        reactions,
        val_frac=0.05,
        test_frac=0.10,
        # keys=('doi', 'raw_index', 'target_comp'),
        keys=(
            "doi",
            "raw_index",
            "target_comp",
            "prototype_path",
        ),
        # keys=('raw_index',),
        # keys=(),
        random_seed=random_seed,
        by_year=True,
    )

    num_train_reactions = train_utils.get_num_reactions(train_reactions)
    ele_counts = train_utils.get_ele_counts(train_reactions)
    print("num_train_reactions", num_train_reactions)
    print("len(eles)", len(eles))
    print("len(ions)", len(ions))
    print("len(ele_counts)", len(ele_counts))
    print([(e, c) for (e, c) in zip(eles, ele_counts)])

    #########################################
    # get train, val, test in batch format
    #########################################
    train_X, train_Y = train_utils.train_data_generator(
        train_reactions,
        num_batch=2000,
        max_mats_num=6,
        batch_size=8,
    )
    train_XY = tf.data.Dataset.zip((train_X, train_Y))
    train_XY = train_XY.prefetch(buffer_size=10)

    # print(next(iter(train_XY.unbatch())))

    val_X, val_Y = train_utils.prepare_dataset(
        val_reactions,
        max_mats_num=6,
        batch_size=8,
        sampling_ratio=1e-3,
        random_seed=random_seed,
    )
    test_X, test_Y = train_utils.prepare_dataset(
        test_reactions,
        max_mats_num=6,
        batch_size=8,
        sampling_ratio=1e-3,
        random_seed=random_seed,
    )

    data = {
        "train_reactions": train_reactions,
        "val_reactions": val_reactions,
        "test_reactions": test_reactions,
        "train_X": train_X,
        "val_X": val_X,
        "test_X": test_X,
        "all_eles": eles,
        "all_ions": ions,
    }
    return data


if __name__ == "__main__":

    from SynthesisSimilarity.core import model_utils

    print("---------------------loading data------------------------------")
    model_dir = "../models/SynthesisRecommendation"
    npz_reload_path = "../rsc_preparation/data_ss.npz"
    framework_model, model_config = model_utils.load_framework_model(model_dir)
    all_elements = model_config["all_eles"]
    featurizer_type = model_config["featurizer_type"]

    test_data = load_and_generate_test_set(
        reload_path=npz_reload_path,
        featurizer_type=featurizer_type,
        all_elements=all_elements,
    )
    train_reactions = test_data["train_reactions"]
    val_reactions = test_data["val_reactions"]
    test_reactions = test_data["test_reactions"]

    print("---------------------saving data------------------------------")
    print("len(train_reactions)", len(train_reactions))
    print("len(val_reactions)", len(val_reactions))
    print("len(test_reactions)", len(test_reactions))
    npz_save_path = "../rsc/data_split.npz"
    np.savez(
        npz_save_path,
        train_reactions=train_reactions,
        val_reactions=val_reactions,
        test_reactions=test_reactions,
    )

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


def generate_test_set_0():
    # based on precursor pairs
    list_1 = ["MgCr2S4"]
    list_2 = ["MgCr2O4", "MgV2S4", "MgS", "Cr2S3", "LiF", "TiO2"]

    list_1 = [Composition(x).as_dict() for x in list_1]
    list_2 = [Composition(x).as_dict() for x in list_2]

    all_pairs = list(itertools.product(list_1, list_2))
    ref_value = []
    return all_pairs, ref_value


def generate_test_set_1():
    # based on precursor pairs
    list_1 = [
        "Li2CO3",
        "LiOH",
        "LiNO3",
        "Li2O",
        "LiCH3COO",
        "LiH2PO4",
        "LiF",
        "Li",
        "Li2S",
    ]
    list_2 = ["Fe2O3", "Fe(NO3)2", "Fe3O4", "FeC2O4", "FePO4", "Fe"]
    list_3 = ["CaO", "CaCO3", "Ca(NO3)2", "CaF2"]
    list_4 = ["Co2O3", "Co3O4", "Co(NO3)2", "CoO", "CoCO3", "Co(CH3COO)2", "Co"]
    list_5 = ["Ba(NO3)2", "BaCO3", "BaO", "Ba(CH3COO)2", "BaO2", "BaF2"]
    list_6 = [
        "MnO2",
        "MnCO3",
        "Mn2O3",
        "Mn3O4",
        "MnO",
        "Mn(NO3)2",
        "Mn(CH3COO)2",
        "Mn",
        "MnC2O4",
        "MnOOH",
    ]
    list_7 = ["LiMnO2", "LiMn2O4", "Li2MnO3", "Li4Mn5O12"]
    list_8 = ["NaNiO2", "Na5NiO4", "Na4(NiO2)9"]
    list_9 = [
        "YVO4",
        "YVO3",
        "YV3O9",
    ]
    list_10 = [
        "Li4TiO4",
        "Li2TiO3",
        "Li4Ti5O12",
        "Li7Ti11O24",
        "LiTi2O4",
        "LiTiO2",
        "Li2Ti3O7",
        "Li7Ti8O20",
        "Li3TiO4",
    ]
    all_lists = [
        list_1,
        list_2,
        list_3,
        list_4,
        list_5,
        list_6,
        list_7,
        list_8,
        list_9,
        list_10,
    ]
    all_pairs = []
    ref_value = []
    for l in all_lists:
        l = [Composition(x).as_dict() for x in l]
        all_pairs.extend(list(itertools.combinations(l, 2)))
    return all_pairs, ref_value


def generate_test_set_2(
    reload_path="generated/evaluation_data.npz",
    common_eles=set(),
    num_pairs=30,
    random_seed=7,
):
    # based on ion substitution
    all_pairs = []
    ref_value = []
    random.seed(random_seed)

    local_data = np.load(reload_path, allow_pickle=True)
    local_data = local_data["eval_data"].item()
    local_data = local_data["ion_substitution"]
    all_elements = list(local_data["all_elements"])
    sub_pairs = local_data["sub_pairs"]
    random.shuffle(sub_pairs)
    common_eles = set(common_eles)
    for pair in sub_pairs:
        if len(common_eles) > 0 and not (
            common_eles.issubset(set(pair["m1"]["elements"].keys()))
            and common_eles.issubset(set(pair["m2"]["elements"].keys()))
            and len(pair["m1"]["elements"]) == len(common_eles) + 1
            and len(pair["m2"]["elements"]) == len(common_eles) + 1
        ):
            continue
        if pair["prob"] is None:
            continue
        all_pairs.append((pair["m1"]["elements"], pair["m2"]["elements"]))
        ref_value.append(np.log(pair["prob"]))
        if len(all_pairs) > num_pairs:
            break
    return all_pairs, ref_value


def generate_test_set_2_2(reload_path="generated/evaluation_data.npz", random_seed=7):
    # based on ion substitution
    all_pairs = []
    ref_value = []
    random.seed(random_seed)

    local_data = np.load(reload_path, allow_pickle=True)
    local_data = local_data["eval_data"].item()
    local_data = local_data["ion_substitution"]
    all_elements = list(local_data["all_elements"])
    sub_pairs = local_data["sub_pairs"]
    random.shuffle(sub_pairs)
    diff_eles_to_find = {"Ba", "Ca"}
    num_pairs = 20
    print("len(sub_pairs)", len(sub_pairs))
    for pair in sub_pairs:
        if len(pair["m2"]["elements"]) != len(pair["m2"]["elements"]):
            continue
        common_eles = set(pair["m1"]["elements"].keys()) & set(
            pair["m2"]["elements"].keys()
        )
        if len(common_eles) + 1 != len(pair["m2"]["elements"]):
            continue
        diff_eles = (
            set(pair["m1"]["elements"].keys()) | set(pair["m2"]["elements"].keys())
        ) - common_eles
        if not diff_eles == diff_eles_to_find:
            continue
        all_pairs.append((pair["m1"]["elements"], pair["m2"]["elements"]))
        ref_value.append(np.log(pair["prob"]))
        if len(all_pairs) > num_pairs:
            break
    return all_pairs, ref_value


def generate_test_set_3():
    # find a list of ternaries with common elements
    pass


def generate_test_set_4(
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


def generate_test_set_vec_math_1():
    target_formulas = [
        "BaTiO3",
        "BaTiO3",
        "SrZnO2",
        "SrZnO2",
        "FeCuO2",
        "Mn2CuO4",
        "BaAl2O4",
        "K2TiO3",
    ]
    precursor_positive_formulas = [
        [
            "SrCO3",
        ],
        [
            "Fe2O3",
        ],
        [
            "BaCO3",
        ],
        [
            "CuO",
        ],
        [
            "Y2O3",
        ],
        [
            "Co3O4",
        ],
        [
            "Cr2O3",
        ],
        [
            "Na2CO3",
        ],
    ]
    precursor_negative_formulas = [
        [
            "BaCO3",
        ],
        [
            "TiO2",
        ],
        [
            "SrCO3",
        ],
        [
            "ZnO",
        ],
        [
            "Fe2O3",
        ],
        [
            "MnO2",
        ],
        [
            "Al2O3",
        ],
        [
            "K2CO3",
        ],
    ]
    return {
        "target_formulas": target_formulas,
        "positive_formulas": precursor_positive_formulas,
        "negative_formulas": precursor_negative_formulas,
        "mode": "precursor",
    }


def generate_test_set_vec_math_2():
    target_formulas = [
        "Li3MnCoO5",
        "Li3MnCoO5",
        "Li3MnCoO5",
        "LiCoO2",
    ]
    target_positive_formulas = [
        [
            "LiNiO2",
        ],
        [
            "LiNiO2",
        ],
        [
            "LiNiO2",
        ],
        [
            "LiNiO2",
        ],
    ]
    target_negative_formulas = [
        [
            "LiMnO2",
        ],
        [
            "LiCoO2",
        ],
        [],
        [],
    ]
    return {
        "target_formulas": target_formulas,
        "positive_formulas": target_positive_formulas,
        "negative_formulas": target_negative_formulas,
        "mode": "target",
    }


if __name__ == "__main__":

    from SynthesisSimilarity.core import model_utils

    print("---------------------loading data------------------------------")
    model_dir = "../models/SynthesisRecommendation"
    npz_reload_path = "../rsc_preparation/data_ss.npz"
    framework_model, model_config = model_utils.load_framework_model(model_dir)
    all_elements = model_config["all_eles"]
    featurizer_type = model_config["featurizer_type"]

    test_data = generate_test_set_4(
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

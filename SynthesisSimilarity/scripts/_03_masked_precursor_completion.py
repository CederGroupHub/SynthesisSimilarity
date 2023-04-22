"""
    Predict complete the precursor list for the given target material with conditional precursors.
"""

import os
import sys

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("parent_folder", parent_folder)
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

import numpy as np

from SynthesisSimilarity.core import utils
from SynthesisSimilarity.core import model_utils
from SynthesisSimilarity.core import callbacks


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def get_decode_test_examples():
    # decode target w conditional precursors
    target_formulas = [
        "LaAlO3",
        "LaAlO3",
        "LaAlO3",
        "LaAlO3",
        "LaAlO3",
    ]

    precursor_formulas_conditional = [
        [
            "La(NO3)3",
        ],
        [
            "Al(NO3)3",
        ],
        [
            "La2O3",
        ],
        [
            "Al2O3",
        ],
        [],
    ]
    return {
        "target_formulas": target_formulas,
        "precursor_formulas_conditional": precursor_formulas_conditional,
    }


def decode_target_w_conditional_precursors(
    target_formulas,
    precursor_formulas_conditional,
    framework_model,
    all_elements,
    mat_feature_len,
    featurizer_type,
    max_mats_num,
):

    assert len(target_formulas) == len(precursor_formulas_conditional)

    predict_precursor_callback = callbacks.PredictPrecursorsCallback_2(
        all_elements=all_elements,
        mat_feature_len=mat_feature_len,
        test_data=None,
        output_thresh=0.5,
        featurizer_type=featurizer_type,
    )

    target_compositions = []
    precursors_conditional = []
    zero_composition = np.zeros(
        shape=(len(all_elements),),
        dtype=np.float32,
    )
    for (tar, pres) in zip(target_formulas, precursor_formulas_conditional):
        target_compositions.append(
            utils.formula_to_array(tar, all_elements),
        )
        precursors_conditional.append([])
        for i in range(max_mats_num - 1):
            if i < len(pres):
                precursors_conditional[-1].append(
                    utils.formula_to_array(pres[i], all_elements)
                )
            else:
                precursors_conditional[-1].append(zero_composition)

    target_compositions = np.array(target_compositions)
    precursors_conditional = np.array(precursors_conditional)

    if "reaction_pre" in framework_model.task_names:
        (
            pre_lists_pred,
            pre_str_lists_pred,
        ) = predict_precursor_callback.predict_precursors(
            framework_model,
            target_compositions,
            precursors_conditional=precursors_conditional,
            to_print=True,
        )


if __name__ == "__main__":
    model_dir = "../models/SynthesisRecommendation"

    framework_model, model_config = model_utils.load_framework_model(model_dir)
    all_elements = model_config["all_eles"]
    max_mats_num = model_config["max_mats_num"]
    featurizer_type = model_config["featurizer_type"]
    mat_feature_len = model_config["mat_feature_len"]

    # decode target w conditional precursors
    decode_examples = get_decode_test_examples()
    target_formulas = decode_examples["target_formulas"]
    precursor_formulas_conditional = decode_examples["precursor_formulas_conditional"]

    decode_target_w_conditional_precursors(
        target_formulas=target_formulas,
        precursor_formulas_conditional=precursor_formulas_conditional,
        framework_model=framework_model,
        all_elements=all_elements,
        mat_feature_len=mat_feature_len,
        featurizer_type=featurizer_type,
        max_mats_num=max_mats_num,
    )

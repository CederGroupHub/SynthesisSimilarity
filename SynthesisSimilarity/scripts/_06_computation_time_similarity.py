"""
    Benchmark the time efficiency for batched and non-batched similarity calculation.
"""

import os
import sys

parent_folder = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)
print("parent_folder", parent_folder)
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

import numpy as np
from timebudget import timebudget

from SynthesisSimilarity.core.utils import formula_to_array
from SynthesisSimilarity.core.mat_featurization import featurize_list_of_composition
from SynthesisSimilarity.scripts_utils.recommendation_utils import (
    collect_targets_in_reactions,
)
from SynthesisSimilarity.scripts_utils.precursors_recommendation_utils import (
    PrecursorsRecommendation,
)


def similarity_time():
    with timebudget("Loading model:"):
        precursors_recommendator = PrecursorsRecommendation(
            model_dir="../models/SynthesisRecommendation",
            freq_path="../rsc/pre_count_normalized_by_rxn_ss.json",
            data_path="../rsc/data_split.npz",
            all_to_knowledge_base=False,
        )

        data_path = "../rsc/data_split.npz"
        test_data = np.load(data_path, allow_pickle=True)
        test_reactions = test_data["test_reactions"]
        (
            test_targets,
            test_targets_formulas,
            test_targets_features,
        ) = collect_targets_in_reactions(
            test_reactions,
            precursors_recommendator.all_elements,
            precursors_recommendator.common_precursors_set,
        )

    with timebudget("Evaluate similarity without batching"):
        all_distances = []
        for x in test_targets_formulas:
            formulas = [x]
            # get target_candidate_normal_vecs
            # TODO: should this test_targets_compositions be ndarray?
            test_targets_compositions = [
                formula_to_array(formula, precursors_recommendator.all_elements)
                for formula in formulas
            ]
            test_targets_features = featurize_list_of_composition(
                comps=test_targets_compositions,
                ele_order=precursors_recommendator.all_elements,
                featurizer_type=precursors_recommendator.featurizer_type,
            )

            # TP similarity
            # train_targets_features is pre-transformed features
            # TODO: convert test_targets_features to np in advance
            test_targets_vecs = precursors_recommendator.framework_model.get_mat_vector(
                np.array(test_targets_features)
            ).numpy()

            test_targets_vecs = test_targets_vecs / (
                np.linalg.norm(test_targets_vecs, axis=-1, keepdims=True)
            )

            distance = test_targets_vecs @ precursors_recommendator.train_targets_vecs.T
            all_distances.append(distance)
        all_distances = np.concatenate(
            all_distances,
            axis=0,
        )
        print("all_distances.shape", all_distances.shape)

    with timebudget("Evaluate similarity with batching"):
        all_distances = []

        # get target_candidate_normal_vecs
        # TODO: should this test_targets_compositions be ndarray?
        test_targets_compositions = [
            formula_to_array(formula, precursors_recommendator.all_elements)
            for formula in test_targets_formulas
        ]
        test_targets_features = featurize_list_of_composition(
            comps=test_targets_compositions,
            ele_order=precursors_recommendator.all_elements,
            featurizer_type=precursors_recommendator.featurizer_type,
        )

        # TP similarity
        # train_targets_features is pre-transformed features
        # TODO: convert test_targets_features to np in advance
        test_targets_vecs = precursors_recommendator.framework_model.get_mat_vector(
            np.array(test_targets_features)
        ).numpy()

        test_targets_vecs = test_targets_vecs / (
            np.linalg.norm(test_targets_vecs, axis=-1, keepdims=True)
        )

        all_distances = (
            test_targets_vecs @ precursors_recommendator.train_targets_vecs.T
        )
        print("all_distances.shape", all_distances.shape)


if __name__ == "__main__":
    similarity_time()

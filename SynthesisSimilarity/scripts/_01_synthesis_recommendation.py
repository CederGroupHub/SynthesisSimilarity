"""
    Recommend precursors for given target materials using PrecursorSelector.
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

from pprint import pprint

from SynthesisSimilarity.scripts_utils.precursors_recommendation_utils import (
    PrecursorsRecommendation,
)


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def run_recommendations():
    precursors_recommendator = PrecursorsRecommendation(
        model_dir="../models/SynthesisRecommendation",
        freq_path="../rsc/pre_count_normalized_by_rxn_ss.json",
        data_path="../rsc/data_split.npz",
        all_to_knowledge_base=False,
    )

    test_targets_formulas = [
        "SrZnSO",
        "Na3TiV(PO4)3",
        "GdLu(MoO4)3",
        "BaYSi2O5N",
        "Cu3Yb(SeO3)2O2Cl",
    ]

    print("len(test_targets_formulas)", len(test_targets_formulas))
    print("test_targets_formulas", test_targets_formulas)

    all_predicts = precursors_recommendator.recommend_precursors(
        target_formula=test_targets_formulas,
        top_n=10,
    )

    for i in range(len(test_targets_formulas)):
        pprint(all_predicts[i])
        print()


if __name__ == "__main__":
    run_recommendations()

"""
    Recommend precursors for given target materials using PrecursorSelector.
"""

from pprint import pprint

from SynthesisSimilarity import PrecursorsRecommendation


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def run_recommendations():
    precursors_recommendator = PrecursorsRecommendation()

    test_targets_formulas = [
        "LiFePO4",
        "LiNi0.333Mn0.333Co0.333O2",
    ]

    print("len(test_targets_formulas)", len(test_targets_formulas))
    print("test_targets_formulas", test_targets_formulas)

    all_predicts = precursors_recommendator.recommend_precursors(
        target_formula=test_targets_formulas,
        top_n=10,
        validate_reaction=True,
    )

    for i in range(len(test_targets_formulas)):
        pprint(all_predicts[i])
        print()


if __name__ == "__main__":
    run_recommendations()

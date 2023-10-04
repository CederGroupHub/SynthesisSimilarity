"""
    Calculate similarity between two target materials using the PrecursorSelector encoding.
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

from SynthesisSimilarity.scripts_utils.TarMatSimilarity_utils import TarMatSimilarity


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def calc_similarity(
    model_dir: str,
):
    formula1 = [
        "NaZr2(PO4)3",
    ]
    formula2 = [
        "Zr3(PO4)4",
        "Na3Zr2Si2PO12",
        "Na3Zr1.8Ge0.2Si2PO12",
        "Na3Ca0.1Zr1.9Si2PO11.9",
        "Na3Zr1.9Ti0.1Si2PO12",
        "LiZr2(PO4)3",
        "NaLa(PO3)4",
        "Sr0.125Ca0.375Zr2(PO4)3",
        "Na5Cu2(PO4)3",
        "LiGe2(PO4)3",
        "Li1.8ZrO3",
        "NaNbO3",
        "Li2Mg2(MoO4)3",
        "Sr2Ce2Ti5O16",
        "Ga0.75Al0.25FeO3",
        "Cu2Te",
        "Ni60Fe30Mn10",
        "AgCrSe2",
        "Zn0.1Cd0.9Cr2S4",
        "Cr2AlC",
    ]
    sim_calculator = TarMatSimilarity(model_dir)
    for f1 in formula1:
        for f2 in formula2:
            print("\nComparing %s to %s:" % (f1, f2))
            print("Similarity = %.3f" % sim_calculator.compare(f1, f2))


if __name__ == "__main__":
    calc_similarity(
        model_dir="../models/SynthesisEncoding",
    )

import pdb
import numpy as np

from SynthesisSimilarity.core.utils import array_to_composition
from SynthesisSimilarity.core.utils import composition_to_array
from SynthesisSimilarity.core.utils import get_material_valence_details
from SynthesisSimilarity.core.utils import valence_to_array
from SynthesisSimilarity.core.utils import array_to_formula
from SynthesisSimilarity.core.utils import NEAR_ZERO


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def default_mat_featurizer(composition, **kwargs):
    ######################################################
    # this is only an example, please add a similar one to this to
    # design custom features
    ######################################################
    return composition.copy()


def ion_frac_mat_featurizer(
    composition,
    ele_order,
    ion_order,
    feature_array=None,
    **kwargs,
):
    if feature_array is None:
        mat_composition = array_to_composition(
            comp_array=composition,
            elements=ele_order,
        )
        oxi_details = get_material_valence_details(mat_composition)
        mat_features = valence_to_array(
            mat_ion=oxi_details,
            ion_order=ion_order,
        )
    else:
        mat_features = feature_array.copy()
    return mat_features


_featurizers = {
    "default": default_mat_featurizer,
    "ion_frac": ion_frac_mat_featurizer,
}


def mat_featurizer(
    composition,
    ele_order,
    featurizer_type="default",
    **kwargs,
):
    mat_features = _featurizers[featurizer_type](
        composition=composition,
        ele_order=ele_order,
        **kwargs,
    )

    return mat_features


def featurize_list_of_composition(
    comps,
    ele_order,
    featurizer_type="default",
    **kwargs,
):
    feature_vectors = [
        mat_featurizer(
            composition=comp,
            ele_order=ele_order,
            featurizer_type=featurizer_type,
            **kwargs,
        )
        for comp in comps
    ]
    return feature_vectors


def featurize_reactions(
    reactions,
    ele_order,
    featurizer_type="default",
    ion_order=None,
):
    # convert normalized composition array to custom features
    # Note: reactions is changed here because deepcopy is not used

    for r in reactions:
        r["target_comp_featurized"] = []
        for i, comp in enumerate(r["target_comp"]):
            r["target_comp_featurized"].append(
                mat_featurizer(
                    composition=comp,
                    ele_order=ele_order,
                    featurizer_type=featurizer_type,
                    ion_order=ion_order,
                    feature_array=r["target_valence"][i],
                )
            )
        r["precursors_comp_featurized"] = []
        for i in range(len(r["precursors_comp"])):
            r["precursors_comp_featurized"].append([])
            for j, comp in enumerate(r["precursors_comp"][i]):
                r["precursors_comp_featurized"][-1].append(
                    mat_featurizer(
                        composition=comp,
                        ele_order=ele_order,
                        featurizer_type=featurizer_type,
                        ion_order=ion_order,
                        feature_array=r["precursors_valence"][i][j],
                    )
                )
    mat_feature_len = len(reactions[0]["target_comp_featurized"][0])

    return reactions, mat_feature_len

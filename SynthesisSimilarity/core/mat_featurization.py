import pdb
import numpy as np

from SynthesisSimilarity.core.utils import array_to_composition
from SynthesisSimilarity.core.utils import composition_to_array
from SynthesisSimilarity.core.utils import get_material_valence_details
from SynthesisSimilarity.core.utils import valence_to_array
from SynthesisSimilarity.core.utils import array_to_formula
from SynthesisSimilarity.core.utils import NEAR_ZERO


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'



def default_mat_featurizer(composition, **kwargs):
    ######################################################
    # this is only an example, please add a similar one to this to
    # design custom features
    ######################################################
    return composition.copy()

def shift_zero_mat_featurizer(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features==0] = -1.0
    return mat_features

def shift_zero_mat_featurizer_1(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features==0] = -0.25
    return mat_features

def shift_zero_mat_featurizer_2(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features==0] = -0.5
    return mat_features

def shift_zero_mat_featurizer_3(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features==0] = -0.75
    return mat_features

def shift_zero_mat_featurizer_4(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features==0] = -2.0
    return mat_features

def shift_zero_mat_featurizer_5(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features==0] = 0.5
    return mat_features

def shift_zero_mat_featurizer_6(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features==0] = 1.0
    return mat_features

def exp_non_zero_mat_featurizer(composition, **kwargs):
    mat_features = composition.copy()
    mat_features[mat_features>0] = np.exp(mat_features[mat_features>0])
    return mat_features

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

def ion_frac_shift_zero_mat_featurizer(
    composition,
    ele_order,
    ion_order,
    feature_array=None,
    **kwargs,
):
    mat_features = ion_frac_mat_featurizer(
        composition=composition,
        ele_order=ele_order,
        ion_order=ion_order,
        feature_array=feature_array,
    )
    mat_features[mat_features==0] = -1.0
    return mat_features


def ion_frac_shift_zero_mat_featurizer_2(
    composition,
    ele_order,
    ion_order,
    feature_array=None,
    **kwargs,
):
    mat_features = ion_frac_mat_featurizer(
        composition=composition,
        ele_order=ele_order,
        ion_order=ion_order,
        feature_array=feature_array,
    )
    mat_features[mat_features==0] = -0.5
    return mat_features


def comp_ion_shift_zero_mat_featurizer(
    composition,
    ele_order,
    ion_order,
    feature_array=None,
    **kwargs,
):
    mat_features_1 = shift_zero_mat_featurizer(
        composition=composition,
    )
    mat_features_2 = ion_frac_shift_zero_mat_featurizer(
        composition=composition,
        ele_order=ele_order,
        ion_order=ion_order,
        feature_array=feature_array,
    )
    mat_features = np.concatenate(
        [mat_features_1, mat_features_2]
    )
    return mat_features


def comp_ion_shift_zero_mat_featurizer_2(
    composition,
    ele_order,
    ion_order,
    feature_array=None,
    **kwargs,
):
    mat_features_1 = shift_zero_mat_featurizer_2(
        composition=composition,
    )
    mat_features_2 = ion_frac_shift_zero_mat_featurizer_2(
        composition=composition,
        ele_order=ele_order,
        ion_order=ion_order,
        feature_array=feature_array,
    )
    mat_features = np.concatenate(
        [mat_features_1, mat_features_2]
    )
    return mat_features


_featurizers = {
    'default': default_mat_featurizer,
    'shift_zero': shift_zero_mat_featurizer,
    'exp_non_zero': exp_non_zero_mat_featurizer,
    'ion_frac': ion_frac_mat_featurizer,
    'ion_frac_shift_zero': ion_frac_shift_zero_mat_featurizer,
    'comp_ion_shift_zero': comp_ion_shift_zero_mat_featurizer,
    'shift_zero_1': shift_zero_mat_featurizer_1,
    'shift_zero_2': shift_zero_mat_featurizer_2,
    'shift_zero_3': shift_zero_mat_featurizer_3,
    'shift_zero_4': shift_zero_mat_featurizer_4,
    'shift_zero_5': shift_zero_mat_featurizer_5,
    'shift_zero_6': shift_zero_mat_featurizer_6,
    'comp_ion_shift_zero_2': comp_ion_shift_zero_mat_featurizer_2,
}


def mat_featurizer(
    composition,
    ele_order,
    featurizer_type='default',
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
    featurizer_type='default',
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
    featurizer_type='default',
    ion_order=None,
):
    # convert normalized composition array to custom features
    # Note: reactions is changed here because deepcopy is not used

    # from timebudget import timebudget
    #
    # ele_val_counter_by_mat = collections.Counter()
    # ele_val_counter_by_amt = collections.Counter()
    # ele_val_counter_by_pre = collections.Counter()
    # tar_missed_counter = collections.Counter()
    # pre_missed_counter = collections.Counter()
    # num_rnx = 0
    # rng = random.Random(7)
    # rng.shuffle(reactions)
    # with timebudget('compute valence'):
    #     for i, r in enumerate(reactions[:100]):
    #         if i % 1000 == 0:
    #             print('{} out of {}'.format(i, len(reactions)))
    #
    #         is_valid = True
    #
    #         composition = r['target_comp'][0]
    #         mat_composition = array_to_composition(
    #             comp_array=composition,
    #             elements=ele_order,
    #         )
    #         oxi_details = get_material_valence_details(mat_composition)
    #         if oxi_details is not None:
    #             for k, v in oxi_details.items():
    #                 if v == 0:
    #                     continue
    #                 ele_val_counter_by_mat[k] += 1
    #                 ele_val_counter_by_amt[k] += v
    #         else:
    #             is_valid = False
    #             tar_missed_counter[
    #                 array_to_formula(composition, ele_order)
    #             ] += 1
    #
    #         for pre in r['precursors_comp']:
    #             composition = pre[0]
    #             mat_composition = array_to_composition(
    #                 comp_array=composition,
    #                 elements=ele_order,
    #             )
    #             oxi_details = get_material_valence_details(mat_composition)
    #             if oxi_details is not None:
    #                 for k, v in oxi_details.items():
    #                     if v == 0:
    #                         continue
    #                     ele_val_counter_by_pre[k] += 1
    #             else:
    #                 is_valid = False
    #                 pre_missed_counter[
    #                     array_to_formula(composition, ele_order)
    #                 ] += 1
    #                 print('mat_composition', mat_composition)
    #                 print('oxi_details', oxi_details)
    #                 print('oxi_details', oxi_details)
    #                 print()
    #
    #         if is_valid:
    #             num_rnx += 1
    #
    # if ('X', 0) in ele_val_counter_by_mat:
    #     del ele_val_counter_by_mat[('X', 0)]
    # if ('X', 0) in ele_val_counter_by_amt:
    #     del ele_val_counter_by_amt[('X', 0)]
    # print('num_rnx', num_rnx, len(reactions))
    # print('len(ele_val_counter_by_mat)', len(ele_val_counter_by_mat))
    # print('len(ele_val_counter_by_amt)', len(ele_val_counter_by_amt))
    # print('ele_val_counter_by_mat', ele_val_counter_by_mat.most_common())
    # print('ele_val_counter_by_amt', ele_val_counter_by_amt.most_common())
    # print('tar_missed_counter', tar_missed_counter.most_common())
    # print('pre_missed_counter', pre_missed_counter.most_common())
    # pdb.set_trace()

    for r in reactions:
        r['target_comp_featurized'] = []
        for i, comp in enumerate(r['target_comp']):
            r['target_comp_featurized'].append(
                mat_featurizer(
                    composition=comp,
                    ele_order=ele_order,
                    featurizer_type=featurizer_type,
                    ion_order=ion_order,
                    feature_array=r['target_valence'][i],
                )
            )
        r['precursors_comp_featurized'] = []
        for i in range(len(r['precursors_comp'])):
            r['precursors_comp_featurized'].append([])
            for j, comp in enumerate(r['precursors_comp'][i]):
                r['precursors_comp_featurized'][-1].append(
                    mat_featurizer(
                        composition=comp,
                        ele_order=ele_order,
                        featurizer_type=featurizer_type,
                        ion_order=ion_order,
                        feature_array=r['precursors_valence'][i][j],
                    )
                )
    mat_feature_len = len(reactions[0]['target_comp_featurized'][0])

    return reactions, mat_feature_len


import collections
import numpy as np
import json
import itertools
from functools import cmp_to_key
import gensim

from SynthesisSimilarity.core import utils


def get_elements_of_material_in_raw_reactions(composition, element_substitution):
    elements = set()
    for comp in composition:
        elements.update(comp['elements'].keys())

    for ele in (elements & set(element_substitution.keys())):
        elements.add(element_substitution[ele])
        elements = elements - set(element_substitution.keys())

    return elements


def elements_to_binary_composition(ele_in_mat, all_eles):
    binary_comp = np.zeros(shape=(len(all_eles)), dtype=np.float32)
    for ele in ele_in_mat:
        binary_comp[all_eles.index(ele)] = 1.0
    return binary_comp


def get_binary_composition_of_material_in_raw_reactions(
    composition,
    element_substitution,
    all_eles
):
    ele_in_mat = get_elements_of_material_in_raw_reactions(
        composition,
        element_substitution
    )
    binary_comp = elements_to_binary_composition(
        ele_in_mat, all_eles
    )
    return binary_comp

def collect_targets_in_raw_reactions(
    train_reactions,
    all_elements,
    common_precursors_set,
    exclude_common_precursors=False,
):

    train_targets = {}
    ref_precursors_comp = {}

    for r in train_reactions:
        tar_f = r['target']['material_string']

        pre_fs = set()
        for pre in r['precursors']:
            # use material_formula for precursor but material_string for target
            # because we don't need to encode precursors with FastText
            # material_string can be name of materials sometimes
            non_hydrate_comp = pre['composition']
            if len(non_hydrate_comp) > 1:
                non_hydrate_comp = list(filter(
                    lambda x: (
                        set(x['elements'].keys()) != {'H', 'O'}
                        or x['formula'] == 'H2O2'
                    ),
                    non_hydrate_comp
                ))
            if len(non_hydrate_comp) == 1:
                try:
                    unified_pre_formula = utils.dict_to_simple_formula(
                        non_hydrate_comp[0]['elements']
                    )
                except:
                    unified_pre_formula = pre['material_formula']
            else:
                unified_pre_formula = pre['material_formula']
            pre_fs.add(unified_pre_formula)
            if unified_pre_formula not in ref_precursors_comp:
                ref_precursors_comp[unified_pre_formula] = \
                    get_binary_composition_of_material_in_raw_reactions(
                        pre['composition'],
                        r['reaction']['element_substitution'],
                        all_elements,
                    )
        pre_fs = tuple(sorted(pre_fs))
        if (
            exclude_common_precursors
            and set(pre_fs).issubset(common_precursors_set)
        ):
            continue
        if tar_f not in train_targets:
            train_targets[tar_f] = {
                'comp': get_binary_composition_of_material_in_raw_reactions(
                    r['target']['composition'],
                    r['reaction']['element_substitution'],
                    all_elements,
                ),
                'pres': collections.Counter(),
                'syn_type': collections.Counter(),
                'raw_index': set(),
                'is_common': collections.Counter(),
            }

        train_targets[tar_f]['pres'][pre_fs] += 1
        train_targets[tar_f]['raw_index'].add(r['raw_index'])

        if set(pre_fs).issubset(common_precursors_set):
            train_targets[tar_f]['is_common']['common'] += 1
        else:
            train_targets[tar_f]['is_common']['uncommon'] += 1
        if 'synthesis_type' in r:
            train_targets[tar_f]['syn_type'][r['synthesis_type']] += 1

    train_targets_formulas = list(train_targets.keys())

    return train_targets, train_targets_formulas, ref_precursors_comp


def save_ele_order(raw_reactions,
                   save_path='generated/ele_order_counter.json'):
    ele_order_counter = collections.Counter()

    for r in raw_reactions:
        for comp in r['target']['composition']:
            ele_substituion = r['reaction']['element_substitution']
            ele_pos = {
                ele: r['target']['material_formula'].find(ele)
                for ele in comp['elements']
            }
            for (e1, e2) in itertools.combinations(comp['elements'].keys(), 2):
                if ele_pos[e1] < 0:
                    continue
                if ele_pos[e2] < 0:
                    continue

                e1_subbed = ele_substituion.get(e1, e1)
                e2_subbed = ele_substituion.get(e2, e2)
                if ele_pos[e1] <= ele_pos[e2]:
                    left_ele = e1_subbed
                    right_ele = e2_subbed
                else:
                    left_ele = e2_subbed
                    right_ele = e1_subbed
                ele_order_counter[
                    '{} before {}'.format(left_ele, right_ele)
                ] += 1

    print('len(ele_order_counter)', len(ele_order_counter))
    print(ele_order_counter.most_common(100))

    with open(save_path, 'w') as fw:
        json.dump(ele_order_counter, fw, indent=2)


def sort_elements_by_stat_order(elements, stat_ele_order):
    return sorted(elements, key=cmp_to_key(
        lambda ele_a, ele_b:
        compare_elements_by_stat_order(ele_a, ele_b, stat_ele_order))
    )


def compare_elements_by_stat_order(ele_a, ele_b, stat_ele_order):
    a_before_b = stat_ele_order.get(
        '{} before {}'.format(ele_a, ele_b),
        0
    )
    b_before_a = stat_ele_order.get(
        '{} before {}'.format(ele_b, ele_a),
        0
    )
    return b_before_a - a_before_b


def composition_to_human_formula(
    composition,
    raw_reaction,
    all_elements,
    stat_ele_order
):
    human_formula = None

    all_elements_indices = {
        ele: i for (i, ele) in enumerate(all_elements)
    }

    comp_eles = set(np.array(all_elements)[composition>0])

    ele_substituion = raw_reaction['reaction']['element_substitution']

    for comp in raw_reaction['target']['composition']:
        for (ele, num) in comp['elements'].items():
            ele_subbed = ele_substituion.get(ele, ele)
            if ele_subbed not in comp_eles:
                continue
            if num.isdigit():
                if abs(float(num)) < utils.NEAR_ZERO:
                    continue
                ele_index = all_elements_indices[ele_subbed]
                composition = composition/composition[ele_index]*float(num)
    sorted_comp_eles = sort_elements_by_stat_order(comp_eles, stat_ele_order)
    human_formula = ''
    for ele in sorted_comp_eles:
        ele_index = all_elements_indices[ele]
        if composition[ele_index] == 1.0:
            human_formula += ele
        else:
            human_formula += '{}{:.3f}'.format(
                ele, composition[ele_index]
            ).rstrip('0').rstrip('.')
    return human_formula


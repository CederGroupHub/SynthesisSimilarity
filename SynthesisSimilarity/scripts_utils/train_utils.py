# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import itertools
import json
import pdb
import random
import time
from pprint import pprint
import jsonlines
from functools import cmp_to_key
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
from unidecode import unidecode
import collections
from sympy.parsing.sympy_parser import parse_expr
from copy import deepcopy
from pymatgen.core import Composition
from pymatgen.core import Element

from Synthepedia.concepts.materials.complex import GeneralComposition
from synthesis_dataset.hierarchy import (
    find_base_material,
    make_hierarchy,
    flatten_hierarchy,
)

from SynthesisSimilarity.core import utils

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

Non_Metal_Elements = set(['C', 'H', 'O', 'N', 'Cl', 'F', 'P', 'S', 'Br', 'I', 'Se'] + ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'])

def get_composition_dict(struct_list, elements_vars={}):
    """
    struct_list is from 'composition' field
    e.g. [{'amount': '1.0',
             'elements': {'Fe': '12.0',
                          'O': '24.0',
                          'Sr': '6.0'},
             'formula': 'Sr6(Fe2O4)6'}]
    """
    # goal
    combined_comp = {}
    all_comps = []

    # get all compositions from struct_list
    for tmp_struct in struct_list:
        if (set(tmp_struct['elements'].keys()) == {'H', 'O'}
                and len(struct_list) > 1):
            # not take H2O into account
            continue
        if tmp_struct.get('amount', '1.0') != '1.0':
            # multiply by coefficient if amount is not 1
            tmp_comp = {}
            for ele, num in tmp_struct['elements'].items():
                tmp_comp[ele] = '(' + num + ')*(' + tmp_struct[
                    'amount'] + ')'
            all_comps.append(tmp_comp)
        else:
            all_comps.append(tmp_struct['elements'])

    # combine all composition from struct_list
    for tmp_comp in all_comps:
        for k, v in tmp_comp.items():
            if k not in combined_comp:
                combined_comp[k] = v
            else:
                combined_comp[k] += (' + ' + v)

    # element substitution element k -> v
    for k, v in elements_vars.items():
        if k not in combined_comp:
            continue
        if v not in combined_comp:
            combined_comp[v] = combined_comp[k]
        else:
            combined_comp[v] += (' + ' + combined_comp[k])
        del combined_comp[k]

    return combined_comp


def to_GeneralMat_obj(
    composition,
    amounts_vars={},
    elements_vars={},
):
    """
    composition is either a single dict as following or a list of such dicts.
        e.g.     {'amount': '1.0',
                 'elements': {'Fe': '12.0',
                              'O': '24.0',
                              'Sr': '6.0'},
                    ... }
    """
    #     goal
    mat_obj = None
    fraction_vars = {}
    contain_vars = False

    # put composition in a list if it is a dict
    raw_composition = deepcopy(composition)
    if isinstance(raw_composition, dict):
        raw_composition['amount'] = '1.0'
        raw_composition = [raw_composition]

    #     get composition after substituting elements variables (e.g. RE -> La)
    composition = get_composition_dict(
        raw_composition,
        elements_vars=elements_vars
    )

    # get fraction_vars/amounts_vars
    # assign fraction_vars
    symbol_replace_dict = {}
    # get value of each variable
    for k, v in amounts_vars.items():
        # greek symbol is not supported by sympy, convert to alphabetical one
        new_k = unidecode(k)
        if new_k != k:
            new_k = 'greek_' + new_k
            assert new_k not in amounts_vars
            symbol_replace_dict[k] = new_k

        # get value as a list or a range
        if len(v.get('values', [])) > 0:
            fraction_vars[new_k] = v['values']
        else:
            fraction_vars[new_k] = {}
            if v.get('min_value', None) != None:
                fraction_vars[new_k]['min'] = v['min_value']
            if v.get('max_value', None) != None:
                fraction_vars[new_k]['max'] = v['max_value']
            if len(fraction_vars[new_k]) == 0:
                fraction_vars[new_k] = {'min': 0.0, 'max': 0.0}
            elif len(fraction_vars[new_k]) == 1:
                fraction_vars[new_k] = list(
                    fraction_vars[new_k].values())

                # greek symbol is not supported by sympy, convert to alphabetical one
    for k, new_k in symbol_replace_dict.items():
        for tmp_ele in composition:
            if k in composition[tmp_ele]:
                composition[tmp_ele] = composition[
                    tmp_ele].replace(k, new_k)

                # deal with extra variables
    # might from 'amount' in 'composition' field,
    # which is not in 'amounts_vars'
    # might from del_O
    all_vars = set()
    for ele in composition:
        try:
            tmp_expr = parse_expr(composition[ele])
            all_vars.update(
                set([str(x) for x in tmp_expr.free_symbols]))
        except:
            pass
    extra_vars = all_vars - set(amounts_vars.keys())

    for x in extra_vars:
        # guess del_O as 0.0
        if re.match('del.*', x):
            fraction_vars[x] = [0.0]
            # assume the amounts are represented by x, y, z (true for most cases)
        # undeclared variables other than x, y, z are not considered.
        # because they might be amount, but also possible to be errors from the text
        if x in {'x', 'y', 'z'}:
            fraction_vars[x] = [0.0]

    if len(fraction_vars):
        contain_vars = True

    # get GeneralComposition object
    try:
        mat_obj = GeneralComposition(
            composition=composition,
            contain_vars=contain_vars,
            fraction_vars=fraction_vars,
            edge_composition=[]
        )
    except Exception as e:
        pass

    # check mat_obj is correctly generated
    # because some value of variables might be improper
    try:
        # some value of variables is incredibly large and make the number of element to be negative
        # set skip_wrong_composition = True to skip those wrong values
        # for more strict critera, set skip_wrong_composition = False to skip the entire material
        edge_points = mat_obj.get_critical_compositions(
            skip_wrong_composition=True
        )
        mat_obj.overlap_with(mat_obj)
    except Exception as e:
        # print('Exception in to_GeneralMat_obj: ', e)
        mat_obj = None
    return mat_obj


def load_raw_reactions(data_file):
    if data_file.endswith('.json'):
        with open(data_file, 'r') as fr:
            reactions = json.load(fr)
        if isinstance(reactions, dict) and 'reactions' in reactions:
            reactions = reactions['reactions']
    elif data_file.endswith('.jsonl'):
        with jsonlines.open(data_file, 'r') as fr:
            reactions = list(fr)
    else:
        raise NotImplementedError
    print('len(raw_reactions): ', len(reactions))
    return reactions


def load_valid_reactions(data_file):
    export_reactions = []
    error_counter = collections.Counter()

    reactions = load_raw_reactions(data_file)

    for i, reaction in enumerate(reactions):
        if i % 5000 == 0:
            print('{}/{} reactions loaded'.format(
                i, len(reactions)
            ))
        valid = True
        # print(reaction)
        reaction['raw_index'] = i
        #     target
        #     generate Material object from entire composition for general usage
        # the raw reaction dataset already split reactions by element substitutions
        # Therefore, in the raw data, each reaction only has one set of element substitutions
        tmp_mat_obj = to_GeneralMat_obj(
            composition=reaction['target']['composition'],
            amounts_vars=reaction['target']['amounts_vars'],
            elements_vars=reaction['reaction']['element_substitution']
        )
        if not tmp_mat_obj:
            error_counter['error target'] += 1
            if error_counter['error target'] < 3:
                print('might be wrong: ', reaction['target'])
            valid = False
        reaction['target_obj'] = tmp_mat_obj
        #     precursor
        pre_objs = []
        for tmp_pre in reaction['precursors']:
            tmp_mat_obj = to_GeneralMat_obj(
                composition=tmp_pre['composition'],
                amounts_vars=tmp_pre['amounts_vars'],
                elements_vars=reaction['reaction']['element_substitution']
            )
            if not tmp_mat_obj:
                error_counter['error precursor'] += 1
                valid = False
            pre_objs.append(tmp_mat_obj)
        reaction['precursor_objs'] = pre_objs
        if valid:
            export_reactions.append(reaction)
    print('len(export_reactions): ', len(export_reactions))
    print(error_counter)
    return export_reactions


def format_as_dataset_mat(formula):
    comp = Composition(formula).as_dict()
    comp = {k: str(v) for (k, v) in comp.items()}
    comp = [{
        'amount': '1',
        'elements': comp,
        'formula': formula,
        'species': None,
        'valence': None,
    }]
    mat = {
        'additives': [],
        'amounts_vars': {},
        'composition': comp,
        'elements_vars': {},
        'is_acronym': False,
        'material_formula': formula,
        'material_name': '',
        'material_string': formula,
        'oxygen_deficiency': '',
        'phase': '',
        'thermo': [{}],
    }
    return mat


def composition_to_frac_formula(
    composition,
    raw_reaction,
    all_elements,
):
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

    frac_formula = Composition(
        utils.array_to_composition(composition, all_elements)
    ).reduced_formula

    return frac_formula


def add_prototype_info(
    reactions: list,
    all_elements: list,
    path_raw_reactions: str,
    doping_thresh: float = 0.3,
    rounding_digits: int = -1,
):
    raw_reactions = load_raw_reactions(data_file=path_raw_reactions)

    all_targets = []
    base_materials = []
    all_formulas = []
    for (i, r) in enumerate(reactions):
        for comp in r['target_comp']:
            human_formula = composition_to_frac_formula(
                comp,
                raw_reactions[r['raw_index']],
                all_elements,
            )
            tar = format_as_dataset_mat(human_formula)
            base_materials.append(
                find_base_material(
                    material=tar,
                    doping_thresh=doping_thresh,
                    rounding_digits=rounding_digits,
                )
            )
            all_targets.append(tar)
            all_formulas.append(human_formula)

    print('len(all_targets) to add_prototype_info', len(all_targets))

    mat_hierarchy = make_hierarchy(
        materials=zip(base_materials, all_targets),
        name='AllMaterials'
    )
    mat_paths = flatten_hierarchy(mat_hierarchy)
    for r in reactions:
        for _ in r['target_comp']:
            m_p = mat_paths.pop(0)
            r['prototype_path'].append(m_p[0])
    assert len(mat_paths) == 0

    return reactions


def load_synthesis_data(
    reload_path,
    data_path=None,
    max_mats_num=6,
    reload=False,
    expand_reaction=True,
):
    # initialization
    effective_eles = set()
    effective_reactions = []
    effective_ions = set()
    ion_counter = collections.Counter()

    if not reload:
        reactions = load_valid_reactions(data_path)

        ele_counter = collections.Counter()
        # get all effective_eles
        for r in reactions:
            effective_eles.update(r['target_obj'].composition.keys())

            for e in r['target_obj'].composition:
                ele_counter[e] += 1

            for p in r['precursor_objs']:
                effective_eles.update(p.composition.keys())
        effective_eles = sorted(
            effective_eles,
            key=lambda x: (Element(x).X, x)
        )
        effective_eles_numpy = np.array(effective_eles)

        print('ele_counter', ele_counter)

        # get effective_reactions
        # and convert composition dict to numpy array
        num_eff_rxn = 0
        print('reactions to effective ones', len(reactions))
        for r in reactions:
            valid = True
            # TODO: print filtered reactions
            if len(r['precursors']) + 1 > max_mats_num:
                valid = False
                continue

            tar = r['target_obj'].get_critical_compositions()
            r['target_comp'] = [
                utils.composition_to_array(
                    comp.composition, effective_eles
                )
                for comp in tar
            ]
            for comp in r['target_comp']:
                if np.sum(comp) == 0:
                    valid = False
            r['target_valence'] = []
            for comp in tar:
                oxi_details = utils.get_material_valence_details(comp.composition)
                r['target_valence'].append(oxi_details)
                if oxi_details is None:
                    valid = False

            precursors = []
            r['precursors_comp'] = []
            r['precursors_valence'] = []
            for p in r['precursor_objs']:
                pres = p.get_critical_compositions()
                # precursors in few reactions (~15) could contain variables
                # neglect these ones because extra effort is needed to determine
                # whether the variable in precursors and target should have the same value or not
                if len(pres) > 1:
                    valid = False
                precursors.append(pres)
                r['precursors_comp'].append([
                    utils.composition_to_array(
                        comp.composition, effective_eles
                    )
                    for comp in pres
                ])
                for comp in r['precursors_comp'][-1]:
                    if np.sum(comp) == 0:
                        valid = False
                r['precursors_valence'].append([])
                for comp in pres:
                    oxi_details = utils.get_material_valence_details(comp.composition)
                    r['precursors_valence'][-1].append(oxi_details)
                    if oxi_details is None:
                        valid = False

            all_pre_comps = sum(r['precursors_comp'], [])
            all_pre_comps = utils.get_composition_string(np.array(all_pre_comps))
            all_pre_comps = all_pre_comps.numpy()
            if len(set(all_pre_comps)) < len(all_pre_comps):
                # precursors should not be repeated
                valid = False

            if valid:
                num_eff_rxn += 1
                if expand_reaction:
                    flatten_r = []
                    for tar_pres_index in itertools.product(*(
                        [range(len(r['target_comp']))]
                        + [range(len(pre_comps)) for pre_comps in r['precursors_comp']])
                    ):
                        tar = r['target_comp'][tar_pres_index[0]]
                        tar_val = r['target_valence'][tar_pres_index[0]]
                        one_in_flatten_r = {
                            'target_comp': [tar],
                            'precursors_comp': [],
                            'target_valence': [tar_val],
                            'precursors_valence': [],
                            'operations': r['operations'],
                            'raw_index': r['raw_index'],
                            'synthesis_type': r['synthesis_type'],
                            'doi': r['doi'],
                            'year': r['year'],
                            'prototype_path': [],
                        }
                        tar_eles = set(effective_eles_numpy[tar>0])
                        # remove target w/o metals due to variables
                        if tar_eles.issubset(Non_Metal_Elements):
                            continue
                        # remove target w/ only one metal and O
                        if len(tar_eles - {'O',}) <= 1:
                            continue
                        pre_eles_all = set()
                        for i, pre_index in enumerate(tar_pres_index[1:]):
                            pre = r['precursors_comp'][i][pre_index]
                            pre_val = r['precursors_valence'][i][pre_index]
                            pre_eles = set(effective_eles_numpy[pre>0])
                            # precursors is not O2 H2 N2, H2O, NH3.
                            # But precursor can be NH4NO3
                            if len(pre_eles) <= 2 and pre_eles.issubset({'O', 'H', 'N'}):
                                continue
                            # precursor with extra metal elements other than target skipped
                            # Attention, this only works for solid-state synthesis
                            # TODO: For hydrothermal and precipitation, this should be modified
                            # because many targets in hydrothermal and precipitation synthesis are simple
                            if (len(pre_eles - tar_eles - Non_Metal_Elements) > 0):
                                continue
                            # Ignore few cases use precursors w/ multiple metals
                            # these cases can be useful for complex (multi-step) synthesis
                            # to study in the future
                            if len(pre_eles - Non_Metal_Elements) >= 2:
                                continue
                            pre_eles_all.update(pre_eles)
                            one_in_flatten_r['precursors_comp'].append([pre])
                            one_in_flatten_r['precursors_valence'].append([pre_val])
                        # Elements in target should be covered by elements from precursors
                        # except for 'O' and 'H'.
                        # O can from air.
                        # In very few cases, H is added after transforming precursor to intermediates.
                        if not tar_eles.issubset(pre_eles_all | {'O', 'H', }):
                            continue
                        # only investigate cases where number of precursors >= 2
                        if len(one_in_flatten_r['precursors_comp']) <= 1:
                            continue
                        flatten_r.append(one_in_flatten_r)
                    num_flatten_r = len(flatten_r)
                    for one_in_flatten_r in flatten_r:
                        one_in_flatten_r['count_weight'] = 1.0/float(num_flatten_r)
                    effective_reactions.extend(flatten_r)
                else:
                    effective_reactions.append(r)

        for i, r in enumerate(effective_reactions):
            r['id'] = i

        print('number of effective reactions', num_eff_rxn)

        # get all effective ions
        for r in effective_reactions:
            for comp in r['target_valence']:
                effective_ions.update(comp.keys())
                for ion in comp:
                    ion_counter[ion] += 1
            for pres in r['precursors_valence']:
                for comp in pres:
                    effective_ions.update(comp.keys())
                    for ion in comp:
                        ion_counter[ion] += 1
        assert ('X', 0) not in effective_ions
        assert ('X', 0) not in ion_counter
        effective_ions = sorted(
            effective_ions,
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        print('effective_ions', effective_ions)
        print('ion_counter', ion_counter)

        # valence to array
        for r in effective_reactions:
            r['target_valence'] = [
                utils.valence_to_array(
                    mat_ion=mat_ion,
                    ion_order=effective_ions,
                )
                for mat_ion in r['target_valence']
            ]
            for i in range(len(r['precursors_valence'])):
                r['precursors_valence'][i] = [
                    utils.valence_to_array(
                        mat_ion=mat_ion,
                        ion_order=effective_ions,
                    )
                    for mat_ion in r['precursors_valence'][i]
                ]
        # add prototype info
        effective_reactions = add_prototype_info(
            reactions=effective_reactions,
            all_elements=effective_eles,
            path_raw_reactions=data_path,
            doping_thresh=0.3,
            rounding_digits=1,
        )

        data_dir = os.path.dirname(reload_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        np.savez(
            reload_path,
            effective_eles=effective_eles,
            effective_reactions=effective_reactions,
            effective_ions=effective_ions,
            ion_counter=ion_counter,
        )
    else:
        assert reload_path != None
        local_data = np.load(reload_path, allow_pickle=True)
        effective_eles = list(local_data['effective_eles'])
        effective_reactions = local_data['effective_reactions']
        effective_ions = list(local_data['effective_ions'])
        effective_ions = [(x[0], int(x[1])) for x in effective_ions]
        ion_counter = local_data['ion_counter'].item()
        if expand_reaction:
            # data validation
            for r in effective_reactions:
                assert len(r['target_comp']) == 1, \
                    'Reaction not expanded'
                for x in r['precursors_comp']:
                    assert len(x) == 1, 'Reaction not expanded'
                pre_fs = set([
                    utils.array_to_formula(x[0], effective_eles)
                    for x in r['precursors_comp']
                ])
                assert len(pre_fs) == len(r['precursors_comp']), \
                    "len(pre_fs) != len(r['precursors_comp'])"
    return effective_eles, effective_reactions, effective_ions, ion_counter


def truncate_valence_array_in_reactions(
    reactions,
    all_ions,
    ion_counter,
    ion_freq_threshold=0,
    common_ion_not_feature=False,
):
    # remove rare ions
    effective_ions = list(filter(
        lambda ion: ion_counter.get(ion, 0) >= ion_freq_threshold,
        all_ions
    ))

    # remove common ions if needed
    if common_ion_not_feature:
        # get common ions
        common_ion_eles = set()
        common_ions = set()
        for (ion, freq) in ion_counter.most_common():
            (ele, val) = ion
            assert ion != ('X', 0)
            if ele == 'X':
                continue
            if ele in common_ion_eles:
                continue
            common_ion_eles.add(ele)
            common_ions.add(ion)
        # remove common ions
        effective_ions = list(filter(
            lambda ion: ion not in common_ions,
            effective_ions
        ))

    # get indices of effective ions
    mapping_indices = [
        all_ions.index(ion)
        for ion in effective_ions
    ]
    # rnx_to_remove = set()
    for k, r in enumerate(reactions):
        for i in range(len(r['target_valence'])):
            r['target_valence'][i] = r['target_valence'][i][mapping_indices]
            # if np.abs(np.sum(r['target_valence'][i])) < utils.NEAR_ZERO:
            #     rnx_to_remove.add(k)
        for i in range(len(r['precursors_valence'])):
            for j in range(len(r['precursors_valence'][i])):
                r['precursors_valence'][i][j] = r['precursors_valence'][i][j][mapping_indices]

    # print('len(reactions) before truncating ions', len(reactions))
    # reactions_truncated = [
    #     r
    #     for (k, r) in enumerate(reactions)
    #     if k not in rnx_to_remove
    # ]
    # print('len(reactions) after truncating ions', len(reactions_truncated))

    return reactions, effective_ions


def load_mp_data(data_file, ele_order, use_icsd_only=True):
    with open(data_file, 'r') as fr:
        materials = json.load(fr)
    # filter out materials with elements not in synthesis database
    eles_set = set(ele_order)
    materials = list(filter(
        lambda x: set(x['reduced_cell_formula'].keys()).issubset(eles_set),
        materials
    ))
    if use_icsd_only:
        # filter out non-icsd materials
        materials = list(filter(
            lambda x: len(x['icsd_ids']) > 0,
            materials
        ))
    print('number of materials in mp db', len(materials))

    # convert composition dict to numpy array
    effective_reactions = []
    for i, mat in enumerate(materials):
        tmp_reaction = {
            'target_comp': [
                utils.composition_to_array(
                    mat['reduced_cell_formula'], ele_order
                )
             ],
            'precursors_comp': [],
            'raw_index': i,
            'operations': [],
        }
        effective_reactions.append(tmp_reaction)

    for i, r in enumerate(effective_reactions):
        r['id'] = i

    return effective_reactions


def load_common_precursors(all_elements,
                           file_path='rsc/most_common_precursor.json',
                           as_set=True):
    # load common precursors
    with open(file_path, 'r') as fr:
        common_precursors = json.load(fr)

    for ele, x in list(common_precursors.items()):
        unified_formula = utils.array_to_formula(
            utils.formula_to_array(x, all_elements),
            all_elements
        )
        common_precursors[ele] = unified_formula

    if as_set:
        common_precursors = set(common_precursors.values())

    return common_precursors


def load_precursor_frequencies(
    all_elements,
    file_path='rsc/pre_count_normalized_by_rxn.json',
):
    # load common precursors
    with open(file_path, 'r') as fr:
        precursor_frequencies = json.load(fr)

    for ele in precursor_frequencies:
        for p_p in precursor_frequencies[ele]:
            # unify formula
            try:
                p_p['formula'] = utils.array_to_formula(
                    utils.formula_to_array(p_p['formula'], all_elements),
                    all_elements
                )
            except:
                pass

    return precursor_frequencies


def use_only_uncommon_reactions(
    reactions,
    all_elements,
    path_precursor_frequencies='rsc/pre_count_normalized_by_rxn.json',
):
    # load_common_precursors and regenerate the formulas following order of all_elements
    if isinstance(path_precursor_frequencies, str):
        precursor_frequencies = load_precursor_frequencies(
            all_elements=all_elements,
            file_path=path_precursor_frequencies,
        )
    elif (
        isinstance(path_precursor_frequencies, dict)
        and 'all' in path_precursor_frequencies
    ):
        precursor_frequencies = load_precursor_frequencies(
            all_elements=all_elements,
            file_path=path_precursor_frequencies['all'],
        )
    else:
        raise NotImplementedError
    common_precursors_set = set([
        precursor_frequencies[ele][0]['formula']
        for ele in precursor_frequencies
    ])

    effective_reactions = []
    for r in reactions:
        has_uncommon_precursors = False
        for pres_comp in r['precursors_comp']:
            if has_uncommon_precursors:
                break
            for x in pres_comp:
                x_formula = utils.array_to_formula(x, all_elements)
                if x_formula not in common_precursors_set:
                    has_uncommon_precursors = True
                    break
        if has_uncommon_precursors:
            effective_reactions.append(r)

    return effective_reactions


def get_max_temperature(temperature_dict_list):
    # goal
    max_T = 1000

    assert len(temperature_dict_list) > 0
    all_Ts = [tmp_T['max_value'] for tmp_T in temperature_dict_list]
    max_T = max(all_Ts)
    return max_T

def get_max_firing_T(reaction):
    temperature_dict_list = []
    for op in reaction['operations']:
        if (op['type'] == 'Firing'
                and op['conditions']
                and op['conditions']['heating_temperature']):
            temperature_dict_list.extend(op['conditions']['heating_temperature'])
    if temperature_dict_list:
        firing_T = get_max_temperature(temperature_dict_list)
    else:
        firing_T = 1000
    return firing_T

def prepare_dataset(
    effective_reactions,
    max_mats_num=5,
    batch_size=32,
    sampling_ratio=0.01,
    precursor_drop_n=1,
    random_seed=None,
):
    """
    generate a static dataset

    :param effective_reactions:
    :param max_mats_num:
    :param batch_size:
    :param sampling_ratio:
    :return:
    """
    # initialization
    data_dicts = []

    if random_seed is not None:
        random_gen = random.Random(random_seed)
    else:
        random_gen = random

    # get paired data
    for tmp_index, (r1, r2) in enumerate(
            itertools.combinations(
                effective_reactions, 2
            )
    ):
        if tmp_index % 100000000 == 0:
            print('prepare_dataset tmp_index', tmp_index)
        scale_factor = r1.get('count_weight', 1.0) * r2.get('count_weight', 1.0)
        if random_gen.random() > sampling_ratio*scale_factor:
            continue

        r1_target_index = random_gen.randrange(len(r1['target_comp']))
        r1_target = r1['target_comp'][r1_target_index]
        r1_target_featurized = r1['target_comp_featurized'][r1_target_index]
        r2_target_index = random_gen.randrange(len(r2['target_comp']))
        r2_target = r2['target_comp'][r2_target_index]
        r2_target_featurized = r2['target_comp_featurized'][r2_target_index]
        r1_precursors_index = [
            random_gen.randrange(len(comps))
            for comps in r1['precursors_comp']
        ]
        r1_precursors = [
            r1['precursors_comp'][i][j]
            for i, j in enumerate(r1_precursors_index)
        ]
        r1_precursors_featurized = [
            r1['precursors_comp_featurized'][i][j]
            for i, j in enumerate(r1_precursors_index)
        ]
        r2_precursors_index = [
            random_gen.randrange(len(comps))
            for comps in r2['precursors_comp']
        ]
        r2_precursors = [
            r2['precursors_comp'][i][j]
            for i, j in enumerate(r2_precursors_index)
        ]
        r2_precursors_featurized = [
            r2['precursors_comp_featurized'][i][j]
            for i, j in enumerate(r2_precursors_index)
        ]
        data_dicts.append(
            {
                'reaction_1': [r1_target] + r1_precursors,
                'reaction_2': [r2_target] + r2_precursors,
                'reaction_1_featurized': [r1_target_featurized] + r1_precursors_featurized,
                'reaction_2_featurized': [r2_target_featurized] + r2_precursors_featurized,
                'precursors_1_conditional': utils.random_drop_in_list(
                    r1_precursors,
                    drop_n=precursor_drop_n,
                    sample_shape=r1_target.shape,
                ),
                'precursors_2_conditional': utils.random_drop_in_list(
                    r2_precursors,
                    drop_n=precursor_drop_n,
                    sample_shape=r2_target.shape,
                ),
                'temperature_1': get_max_firing_T(r1),
                'temperature_2': get_max_firing_T(r2),
                'synthesis_type_1': r1['synthesis_type'],
                'synthesis_type_2': r2['synthesis_type'],
                ###########################################
                # add more features here is necessary
                ###########################################
            }
        )
    data_dicts = data_dicts[:int(len(data_dicts)/batch_size)*batch_size]
    data_dicts = data_dicts[:int(len(data_dicts) / batch_size) * batch_size]
    print('len(data_dicts)', len(data_dicts))
    random_gen.shuffle(data_dicts)

    data_type, data_shape, padded_data_shape = utils.get_input_format(
        model_type='MultiTasksOnRecipes',
        max_mats_num=max_mats_num,
    )
    data_X, data_Y = utils.dict_to_tf_dataset(
        data_dicts,
        data_type,
        data_shape,
        padded_shape=padded_data_shape,
        column_y=None,
        batch_size=batch_size,
    )
    return data_X, data_Y


def get_mat_dico(reactions, mode='all', num_reserved_ids=10, least_count=5):
    """
    mat_labels is a python list of mat_str
        label is the index in mat_labels
    mat_compositions is python list of compositions, index is
        mat_label
    mat_counts is python list of int, index is mat_label

    :param reactions: reaction is a dict
                    {
                        'target': [comp_1, comp_2, ...],
                        'precursors': [
                            [comp_1, comp_2, ...],
                            [comp_1, comp_2, ...],
                        ]
                    }
    :return:
    """

    # goal
    mat_labels = []
    mat_compositions = []
    mat_counts = []

    # get all_mats
    all_mats = []
    all_count_weights = []
    for r in reactions:
        if mode in {'all', 'target'}:
            targets = r['target_comp']
            all_mats.extend(targets)
            all_count_weights.extend([r.get('count_weight', 1.0)]*len(targets))
        if mode in {'all', 'precursor'}:
            precursors = sum(r['precursors_comp'], [])
            all_mats.extend(precursors)
            all_count_weights.extend([r.get('count_weight', 1.0)]*len(precursors))

    # mat_labels, mat_compositions, mat_counts = utils.convert_mat_to_dico(
    #     all_mats,
    #     composition_shape=reactions[0]['target_comp'][0].shape,
    #     count_weights=all_count_weights,
    #     num_reserved_ids=num_reserved_ids,
    #     least_count=least_count,
    # )

    all_mats_str = utils.get_composition_string(np.array(all_mats))
    all_mats_str = all_mats_str.numpy()
    mat_str_comp = { s: comp for (s, comp) in zip(all_mats_str, all_mats) }
    comp_shape = reactions[0]['target_comp'][0].shape

    mat_labels, mat_counts = utils.convert_list_to_dico(
        all_labels=all_mats_str,
        count_weights=all_count_weights,
        num_reserved_ids=num_reserved_ids,
        least_count=least_count,
    )
    mat_compositions = [
        mat_str_comp.get(l, np.zeros(shape=comp_shape, dtype=np.float32))
        for l in mat_labels
    ]

    return mat_labels, mat_compositions, mat_counts

def get_syn_type_dico(reactions, num_reserved_ids=10, least_count=5):
    """
    mat_labels is a python list of mat_str
        label is the index in mat_labels
    mat_compositions is python list of compositions, index is
        mat_label
    mat_counts is python list of int, index is mat_label

    :param reactions: reaction is a dict
                    {
                        'target': [comp_1, comp_2, ...],
                        'precursors': [
                            [comp_1, comp_2, ...],
                            [comp_1, comp_2, ...],
                        ]
                    }
    :return:
    """

    # goal
    syn_type_labels = []
    syn_type_counts = []

    # get all_mats
    all_syn_types = []
    all_count_weights = []
    for r in reactions:
        all_syn_types.append(r['synthesis_type'])
        all_count_weights.append(r.get('count_weight', 1.0))

    syn_type_labels, syn_type_counts = utils.convert_list_to_dico(
        all_labels=all_syn_types,
        count_weights=all_count_weights,
        num_reserved_ids=num_reserved_ids,
        least_count=least_count,
    )

    return syn_type_labels, syn_type_counts

def get_ele_counts(reactions):
    tar_labels, tar_compositions, tar_counts = get_mat_dico(
        reactions,
        mode='target',
        least_count=0,
    )
    assert len(tar_compositions) > 0
    ele_counts = np.zeros_like(tar_compositions[0])
    for i in range(len(tar_compositions)):
        comp = tar_compositions[i]
        weight = tar_counts[i]
        ele_counts += (comp>0)*weight
    return ele_counts

def get_num_reactions(reactions):
    num = 0
    for r in reactions:
        num += r.get('count_weight', 1.0)
    return num

def train_data_generator(
    reaction_pool,
    num_batch,
    max_mats_num=6,
    batch_size=32,
    precursor_drop_n=1,
):
    reaction_pool = list(reaction_pool)
    reaction_weights = [r.get('count_weight', 1.0) for r in reaction_pool]
    data_size = num_batch*batch_size
    data_type, data_shape, padded_data_shape = utils.get_input_format(
        model_type='MultiTasksOnRecipes',
        max_mats_num=max_mats_num,
    )
    # another way to generate dict is to use dataset.map
    # https://github.com/tensorflow/tensorflow/issues/28643
    def sample_data(all_reactions, weights=None):
        r1 = {}
        r2 = {}
        while r1.get('id', None) == r2.get('id', None):
            react_samples = random.choices(all_reactions, weights=weights, k=2)
            r1 = react_samples[0]
            r2 = react_samples[1]

        r1_target_index = random.randrange(len(r1['target_comp']))
        r1_target = r1['target_comp'][r1_target_index]
        r1_target_featurized = r1['target_comp_featurized'][r1_target_index]
        r2_target_index = random.randrange(len(r2['target_comp']))
        r2_target = r2['target_comp'][r2_target_index]
        r2_target_featurized = r2['target_comp_featurized'][r2_target_index]
        r1_precursors_index = [
            random.randrange(len(comps))
            for comps in r1['precursors_comp']
        ]
        r1_precursors = [
            r1['precursors_comp'][i][j]
            for i, j in enumerate(r1_precursors_index)
        ]
        r1_precursors_featurized = [
            r1['precursors_comp_featurized'][i][j]
            for i, j in enumerate(r1_precursors_index)
        ]
        r2_precursors_index = [
            random.randrange(len(comps))
            for comps in r2['precursors_comp']
        ]
        r2_precursors = [
            r2['precursors_comp'][i][j]
            for i, j in enumerate(r2_precursors_index)
        ]
        r2_precursors_featurized = [
            r2['precursors_comp_featurized'][i][j]
            for i, j in enumerate(r2_precursors_index)
        ]

        feature_dict = (
            {
                'reaction_1': [r1_target] + r1_precursors,
                'reaction_2': [r2_target] + r2_precursors,
                'reaction_1_featurized': [r1_target_featurized] + r1_precursors_featurized,
                'reaction_2_featurized': [r2_target_featurized] + r2_precursors_featurized,
                'precursors_1_conditional': utils.random_drop_in_list(
                    r1_precursors,
                    drop_n=precursor_drop_n,
                    sample_shape=r1_target.shape,
                ),
                'precursors_2_conditional': utils.random_drop_in_list(
                    r2_precursors,
                    drop_n=precursor_drop_n,
                    sample_shape=r2_target.shape,
                ),
                'temperature_1': get_max_firing_T(r1),
                'temperature_2': get_max_firing_T(r2),
                'synthesis_type_1': r1['synthesis_type'],
                'synthesis_type_2': r2['synthesis_type'],
                ###########################################
                # add more features here is necessary
                ###########################################
            }
        )
        return feature_dict

    def feature_dict_gen():
        # for _ in range(data_size):
        while(True):
            feature_dict = sample_data(reaction_pool, weights=reaction_weights)
            yield feature_dict

    def y_array_gen():
        while(True):
            yield 0.0

    dataset_x = tf.data.Dataset.from_generator(
        feature_dict_gen,
        output_types=data_type,
        output_shapes=data_shape,
    )

    dataset_y = tf.data.Dataset.from_generator(
        y_array_gen,
        output_types=tf.float32,
        output_shapes=[],
    )

    data_batch_x = dataset_x.padded_batch(
        batch_size,
        padded_shapes=padded_data_shape
    )

    data_batch_y = dataset_y.padded_batch(
        batch_size,
        padded_shapes=[]
    )
    return data_batch_x, data_batch_y

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2None(v):
    if v is None:
        return v
    if v.lower() in {'none', }:
        return None
    return v


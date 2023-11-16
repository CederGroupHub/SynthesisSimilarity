import numpy as np
from itertools import chain, combinations
from pymatgen.core import Composition, Element
from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy

from SynthesisSimilarity.core.utils import get_elements_in_formula


TOLERANCE = 1e-6  # Tolerance for determining if a particular component fraction is > 0.


def _balance_coeffs(
    reactants: List[Composition], products: List[Composition]
) -> Tuple[np.ndarray, Union[int, float], int]:
    """
    Balances the reaction and returns the new coefficient matrix

    Adapted from reaction-network
    McDermott, M. J., Dwaraknath, S. S., and Persson, K. A. (2021). A graph-based network for predicting chemical reaction pathways in solid-state materials synthesis. Nature Communications, 12(1). https://doi.org/10.1038/s41467-021-23339-x
    """
    compositions = reactants + products
    num_comp = len(compositions)

    all_elems = sorted({elem for c in compositions for elem in c.elements})
    num_elems = len(all_elems)

    comp_matrix = np.array([[c[el] for el in all_elems] for c in compositions]).T

    rank = np.linalg.matrix_rank(comp_matrix)
    diff = num_comp - rank
    num_constraints = diff if diff >= 2 else 1

    # an error = a component changing sides or disappearing
    lowest_num_errors = np.inf

    first_product_idx = len(reactants)

    # start with simplest product constraints, work to more complex constraints
    product_constraints = chain.from_iterable(
        [
            combinations(range(first_product_idx, num_comp), n_constr)
            for n_constr in range(num_constraints, 0, -1)
        ]
    )
    reactant_constraints = chain.from_iterable(
        [
            combinations(range(0, first_product_idx), n_constr)
            for n_constr in range(num_constraints, 0, -1)
        ]
    )
    best_soln = np.zeros(num_comp)

    for constraints in chain(product_constraints, reactant_constraints):
        n_constr = len(constraints)

        comp_and_constraints = np.append(
            comp_matrix, np.zeros((n_constr, num_comp)), axis=0
        )
        b = np.zeros((num_elems + n_constr, 1))
        b[-n_constr:] = 1 if min(constraints) >= first_product_idx else -1

        for num, idx in enumerate(constraints):
            comp_and_constraints[num_elems + num, idx] = 1
            # arbitrarily fix coeff to 1

        coeffs = np.matmul(np.linalg.pinv(comp_and_constraints), b)

        num_errors = 0
        if np.allclose(np.matmul(comp_matrix, coeffs), np.zeros((num_elems, 1))):
            expected_signs = np.array([-1] * len(reactants) + [+1] * len(products))
            num_errors = np.sum(np.multiply(expected_signs, coeffs.T) < TOLERANCE)
            if num_errors == 0:
                lowest_num_errors = 0
                best_soln = coeffs
                break
            if num_errors < lowest_num_errors:
                lowest_num_errors = num_errors
                best_soln = coeffs

    return np.squeeze(best_soln), lowest_num_errors, num_constraints


def balance_w_rxn_network(
    target_formula,
    precursors_formulas,
    ref_materials_comp,
):
    rxn_predict = [
        target_formula,
        {
            'left': {},
            'right': {},
        },
        None,
        '',
    ]
    #     blance reaction
    pres = []
    for x in precursors_formulas:
        if x in ref_materials_comp:
            pres.append(
                ref_materials_comp[x]['material_formula']
            )
        else:
            pres.append(x)
    tars = [target_formula] + ['H2O', 'CO2', 'NH3', 'O2', 'NO2', ]
    all_mats = pres + tars
    coeffs, lowest_num_errors, num_constraints = _balance_coeffs(
        reactants=[Composition(x) for x in pres],
        products=[Composition(x) for x in tars],
    )
    # make coeff for target as 1
    coeffs = coeffs / max(np.abs(coeffs[len(pres)]), TOLERANCE)

    coeffs = np.round(coeffs, 3)
    #     reaction should be solvable
    if np.isinf(lowest_num_errors):
        rxn_predict = None
    #     coefficients should be negative or zero for precursors
    for i, pre in enumerate(pres):
        if coeffs[i] > TOLERANCE:
            rxn_predict = None
    #     coefficients should be positive for targets
    if coeffs[len(pres)] < TOLERANCE:
        rxn_predict = None
    #     format rxn_predict as output
    if rxn_predict is not None:
        for i, coeff in enumerate(coeffs):
            if coeff < -TOLERANCE:
                rxn_predict[1]['left'][all_mats[i]] = -coeff
            elif coeff > TOLERANCE:
                rxn_predict[1]['right'][all_mats[i]] = coeff
        for mat, coeff in rxn_predict[1]['left'].items():
            rxn_predict[3] += f'{coeff} {mat} + '
        rxn_predict[3] = rxn_predict[3].strip('+ ')
        rxn_predict[3] += ' == '
        for mat, coeff in rxn_predict[1]['right'].items():
            rxn_predict[3] += f'{coeff} {mat} + '
        rxn_predict[3] = rxn_predict[3].strip('+ ')
        rxn_predict = tuple(rxn_predict)

    return rxn_predict


def are_coefficients_positive(
    target,
    precursors,
    reaction
):
    is_positive = True
    mat_coeff = {
        **reaction[1]['left'],
        **reaction[1]['right'],
    }
    for pre in precursors:
        if float(mat_coeff.get(pre, 0)) < 0 or pre in reaction[1]['right']:
            is_positive = False
    if float(mat_coeff[target]) <= 0 or target in reaction[1]['left']:
        is_positive = False
    return is_positive


def clear_zero_coeff_precursors(
    precursors,
    reaction,
):
    pres_out = []
    for pre in precursors:
        if pre in reaction[1]['left'] and float(reaction[1]['left'][pre]) > 0:
            pres_out.append(pre)
    return tuple(sorted(pres_out))

def reaction_coeff_to_float(reaction):
    reaction_out = deepcopy(reaction)
    for k in reaction_out[1]['left']:
        reaction_out[1]['left'][k] = float(reaction_out[1]['left'][k])
    for k in reaction_out[1]['right']:
        reaction_out[1]['right'][k] = float(reaction_out[1]['right'][k])
    return reaction_out
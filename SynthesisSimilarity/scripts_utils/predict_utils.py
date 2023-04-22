import random
import numpy as np

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def getAllParaCombo(paraMatrix):
    paraCombo = []
    firstKey = list(paraMatrix.keys())
    firstKey = firstKey[0]
    if len(paraMatrix) == 1:
        for tmp_para in paraMatrix[firstKey]:
            new_combo = {}
            new_combo[firstKey] = tmp_para
            paraCombo.append(new_combo)
    if len(paraMatrix) > 1:
        paraMatrixCopy = paraMatrix.copy()
        paraMatrixCopy.pop(firstKey)
        combos = getAllParaCombo(paraMatrixCopy)
        for tmp_para in paraMatrix[firstKey]:
            for tmp_combo in combos:
                new_combo = tmp_combo.copy()
                new_combo[firstKey] = tmp_para
                paraCombo.append(new_combo)
    return paraCombo

def removeParaCombo(paraCombo, paraTabu):
    paraCombo2 = paraCombo.copy()
    for tmp_papa_tabu in paraTabu:
        for tmp_para in paraCombo:
            if tmp_para not in paraCombo2:
                continue
            tmp_remove = True
            for tmp_key in tmp_papa_tabu:
                if tmp_key in tmp_para:
                    if tmp_para[tmp_key] != tmp_papa_tabu[tmp_key]:
                        tmp_remove = False
                else:
                    tmp_remove = False
            if tmp_remove == True:
                paraCombo2.remove(tmp_para)
    return paraCombo2


def get_materials_in_reaction(reaction):
    target = random.choice(reaction['target_comp'])
    precursors = [
        random.choice(comps)
        for comps in reaction['precursors_comp']
    ]
    return {
        'target': target,
        'precursors': precursors,
    }

def drop_one_dimension(inputs, original_eles, target_eles):
    index_map = [original_eles.index(e) for e in \
            target_eles]
    inputs = [x[index_map] for x in inputs]
    return inputs

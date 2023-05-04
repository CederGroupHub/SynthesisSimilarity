"""
    Recommend precurcurs for given target materials.
"""

import os
import pdb
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
from pprint import pprint
import gensim
import json
import itertools

from SynthesisSimilarity.core.utils import (
    formula_to_array,
    get_elements_in_formula,
    use_file_as_stdout,
)
from SynthesisSimilarity.scripts_utils.recommendation_utils import (
    collect_targets_in_reactions,
    add_to_sorted_list,
)
from SynthesisSimilarity.scripts_utils.precursors_recommendation_utils import (
    PrecursorsRecommendation,
)
from SynthesisSimilarity.scripts_utils.multi_processing_utils import (
    run_multiprocessing_tasks,
)
from SynthesisSimilarity.scripts_utils.MatminerSimilarity_utils import (
    MatMiner_features_for_formulas,
)
from SynthesisSimilarity.scripts_utils.train_utils import (
    load_raw_reactions,
)
from SynthesisSimilarity.scripts_utils.FastTextSimilarity_utils import (
    composition_to_human_formula,
)

__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def evaluation_prediction_precursors(
    test_targets,
    test_targets_formulas,
    all_pres_predict,
):
    # print prediction info
    assert len(test_targets_formulas) == len(
        all_pres_predict
    ), "len(test_targets_formulas) != len(all_pres_predict)"
    all_results = []
    pres_predict_result = []
    pres_predict_result_top2 = []
    pres_predict_result_top3 = []
    pres_predict_result_topn = []
    for x, pres_predict in zip(test_targets_formulas, all_pres_predict):
        pres_true_set = set(test_targets[x]["pres"].keys())

        all_results.append([])
        # print("material: ", x)
        for i, pres in enumerate(pres_predict):
            # print(i, x, test_targets[x]["pres"], pres, pres in pres_true_set)
            all_results[-1].append(len(set(pres_predict[: i + 1]) & pres_true_set) > 0)
        # print()

        pres_predict_result.append(len(set(pres_predict[:1]) & pres_true_set) > 0)
        pres_predict_result_top2.append(len(set(pres_predict[:2]) & pres_true_set) > 0)
        pres_predict_result_top3.append(len(set(pres_predict[:3]) & pres_true_set) > 0)
        pres_predict_result_topn.append(len(set(pres_predict) & pres_true_set) > 0)

    pres_predict_result = np.array(pres_predict_result, dtype=np.int64)
    print(
        "pres_predict all correct: True {num_true}/Total {num_total} = {accuracy}".format(
            num_true=sum(pres_predict_result),
            num_total=len(pres_predict_result),
            accuracy=sum(pres_predict_result) / len(pres_predict_result),
        )
    )

    pres_predict_result_top2 = np.array(pres_predict_result_top2, dtype=np.int64)
    print(
        "pres_predict top2 all correct: True {num_true}/Total {num_total} = {accuracy}".format(
            num_true=sum(pres_predict_result_top2),
            num_total=len(pres_predict_result_top2),
            accuracy=sum(pres_predict_result_top2) / len(pres_predict_result_top2),
        )
    )

    pres_predict_result_top3 = np.array(pres_predict_result_top3, dtype=np.int64)
    print(
        "pres_predict top3 all correct: True {num_true}/Total {num_total} = {accuracy}".format(
            num_true=sum(pres_predict_result_top3),
            num_total=len(pres_predict_result_top3),
            accuracy=sum(pres_predict_result_top3) / len(pres_predict_result_top3),
        )
    )

    pres_predict_result_topn = np.array(pres_predict_result_topn, dtype=np.int64)
    print(
        "pres_predict topn all correct: True {num_true}/Total {num_total} = {accuracy}".format(
            num_true=sum(pres_predict_result_topn),
            num_total=len(pres_predict_result_topn),
            accuracy=sum(pres_predict_result_topn) / len(pres_predict_result_topn),
        )
    )

    max_len = max(map(len, all_results))
    num_short_result = 0
    for i in range(len(all_results)):
        if len(all_results[i]) < max_len:
            if len(all_results[i]) > 0:
                all_results[i].extend(
                    [all_results[i][-1]] * (max_len - len(all_results[i]))
                )
            else:
                all_results[i] = [False] * max_len
            num_short_result += 1
    print("len(all_results)", len(all_results))
    print("num_short_result", num_short_result)
    all_results = np.array(all_results, dtype=np.int64)
    all_results = np.sum(all_results, axis=0) / len(all_results)
    print(list(all_results))
    print()

    return all_results


def run_recommendations():
    precursors_recommendator = PrecursorsRecommendation(
        model_dir="../models/SynthesisRecommendation",
        freq_path="../rsc/pre_count_normalized_by_rxn_ss.json",
        data_path="../rsc/data_split.npz",
    )

    data_path = "../rsc/data_split.npz"
    test_data = np.load(data_path, allow_pickle=True)
    val_reactions = test_data["val_reactions"]
    test_reactions = test_data["test_reactions"]

    (val_targets, val_targets_formulas, _,) = collect_targets_in_reactions(
        val_reactions,
        precursors_recommendator.all_elements,
        precursors_recommendator.common_precursors_set,
    )

    (test_targets, test_targets_formulas, _,) = collect_targets_in_reactions(
        test_reactions,
        precursors_recommendator.all_elements,
        precursors_recommendator.common_precursors_set,
    )

    ########################
    # recommendation through synthesis similarity
    all_pres_predict = recommend_w_SynSym(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=val_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        val_targets,
        val_targets_formulas,
        all_pres_predict,
    )

    all_pres_predict = recommend_w_SynSym(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=test_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        test_targets,
        test_targets_formulas,
        all_pres_predict,
    )

    ########################
    # recommendation through raw composition
    all_pres_predict = recommend_w_RawComp(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=val_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        val_targets,
        val_targets_formulas,
        all_pres_predict,
    )

    all_pres_predict = recommend_w_RawComp(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=test_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        test_targets,
        test_targets_formulas,
        all_pres_predict,
    )

    ########################
    # recommendation through similarity based on matminer representation
    all_pres_predict = recommend_w_MatMiner(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=val_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        val_targets,
        val_targets_formulas,
        all_pres_predict,
    )

    all_pres_predict = recommend_w_MatMiner(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=test_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        test_targets,
        test_targets_formulas,
        all_pres_predict,
    )

    ########################
    # recommendation through similarity based on fasttext representation
    all_pres_predict, fasttext_supported_val_targets_formulas = recommend_w_FastText(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=val_targets_formulas,
        test_targets=val_targets,
        top_n=10,
    )

    evaluation_prediction_precursors(
        val_targets,
        fasttext_supported_val_targets_formulas,
        all_pres_predict,
    )

    all_pres_predict, fasttext_supported_test_targets_formulas = recommend_w_FastText(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=test_targets_formulas,
        test_targets=test_targets,
        top_n=10,
    )

    evaluation_prediction_precursors(
        test_targets,
        fasttext_supported_test_targets_formulas,
        all_pres_predict,
    )

    ########################
    # recommendation through product of precursor frequencies
    all_pres_predict = recommend_w_freq(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=val_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        val_targets,
        val_targets_formulas,
        all_pres_predict,
    )

    all_pres_predict = recommend_w_freq(
        precursors_recommendator=precursors_recommendator,
        test_targets_formulas=test_targets_formulas,
        top_n=10,
    )

    evaluation_prediction_precursors(
        test_targets,
        test_targets_formulas,
        all_pres_predict,
    )


def recommend_w_SynSym(
    precursors_recommendator,
    test_targets_formulas,
    top_n=10,
):
    all_pres_predict = precursors_recommendator.recommend_precursors(
        target_formula=test_targets_formulas,
        top_n=top_n,
        validate_first_attempt=False,
        recommendation_strategy="SynSim_conditional",
    )

    return all_pres_predict


def recommend_w_RawComp(
    precursors_recommendator,
    test_targets_formulas,
    top_n=10,
):
    train_targets_compositions = [
        precursors_recommendator.train_targets[formula]["comp"]
        for formula in precursors_recommendator.train_targets_formulas
    ]
    test_targets_compositions = [
        formula_to_array(formula, precursors_recommendator.all_elements)
        for formula in test_targets_formulas
    ]
    train_targets_vecs = np.array(train_targets_compositions)
    test_targets_vecs = np.array(test_targets_compositions)

    train_targets_vecs = train_targets_vecs / (
        np.linalg.norm(
            train_targets_vecs,
            axis=-1,
            keepdims=True,
        )
    )
    test_targets_vecs = test_targets_vecs / (
        np.linalg.norm(
            test_targets_vecs,
            axis=-1,
            keepdims=True,
        )
    )
    all_distance = test_targets_vecs @ train_targets_vecs.T

    all_pres_predict, _ = precursors_recommendator.recommend_precursors_by_similarity(
        test_targets_formulas=test_targets_formulas,
        train_targets_recipes=precursors_recommendator.train_targets_recipes,
        all_distance=all_distance,
        top_n=top_n,
        strategy="naive_common",
    )

    return all_pres_predict


def recommend_w_MatMiner(
    precursors_recommendator,
    test_targets_formulas,
    top_n=10,
):
    path_to_imputer = "../other_rsc/matminer/mp_imputer_preset_v1.0.2.pkl"
    path_to_scaler = "../other_rsc/matminer/mp_scaler_preset_v1.0.2.pkl"
    train_targets_features = run_multiprocessing_tasks(
        tasks=precursors_recommendator.train_targets_formulas,
        thread_func=MatMiner_features_for_formulas,
        func_args=(
            path_to_imputer,
            path_to_scaler,
        ),
        num_cores=4,
        join_results=True,
        use_threading=False,
        mp_context=None,
    )
    test_targets_features = run_multiprocessing_tasks(
        tasks=test_targets_formulas,
        thread_func=MatMiner_features_for_formulas,
        func_args=(
            path_to_imputer,
            path_to_scaler,
        ),
        num_cores=4,
        join_results=True,
        use_threading=False,
        mp_context=None,
    )
    train_targets_vecs = np.array(train_targets_features)
    test_targets_vecs = np.array(test_targets_features)

    train_targets_vecs = train_targets_vecs / (
        np.linalg.norm(
            train_targets_vecs,
            axis=-1,
            keepdims=True,
        )
    )
    test_targets_vecs = test_targets_vecs / (
        np.linalg.norm(
            test_targets_vecs,
            axis=-1,
            keepdims=True,
        )
    )
    all_distance = test_targets_vecs @ train_targets_vecs.T

    all_pres_predict, _ = precursors_recommendator.recommend_precursors_by_similarity(
        test_targets_formulas=test_targets_formulas,
        train_targets_recipes=precursors_recommendator.train_targets_recipes,
        all_distance=all_distance,
        top_n=top_n,
        strategy="naive_common",
    )

    return all_pres_predict


def recommend_w_FastText(
    precursors_recommendator,
    test_targets_formulas,
    test_targets,
    top_n=10,
):
    fasttext = gensim.models.keyedvectors.KeyedVectors.load(
        "../other_rsc/fasttext_pretrained_matsci/fasttext_embeddings-MINIFIED.model"
    )
    # Need to set this when loading from saved file
    fasttext.bucket = 2000000

    # load ele_order by statistics from text
    with open("../rsc/ele_order_counter.json", "r") as fr:
        stat_ele_order = json.load(fr)

    path_raw_reactions = "../rsc/reactions_v20_20210820_ss.jsonl"
    raw_reactions = load_raw_reactions(data_file=path_raw_reactions)
    print("len(raw_reactions)", len(raw_reactions))

    # encode with fasttext
    train_targets_features = []
    test_targets_features = []
    fasttext_supported_train_targets_formulas = []
    fasttext_supported_test_targets_formulas = []
    for i, x in enumerate(precursors_recommendator.train_targets_formulas):
        try:
            human_formula = composition_to_human_formula(
                precursors_recommendator.train_targets[x]["comp"],
                raw_reactions[
                    list(precursors_recommendator.train_targets[x]["raw_index"])[0]
                ],
                precursors_recommendator.all_elements,
                stat_ele_order,
            )
        except:
            print(
                "error in guess formula",
                x,
                list(precursors_recommendator.train_targets[x]["raw_index"]),
            )

        try:
            train_targets_features.append(fasttext[human_formula.lower()])
            fasttext_supported_train_targets_formulas.append(x)
        except:
            print("fasttext wrong train x skipped", x, human_formula)

    for i, x in enumerate(test_targets_formulas):
        try:
            human_formula = composition_to_human_formula(
                formula_to_array(x, precursors_recommendator.all_elements),
                raw_reactions[list(test_targets[x]["raw_index"])[0]],
                precursors_recommendator.all_elements,
                stat_ele_order,
            )
        except:
            print(
                "error in guess formula",
                x,
                list(test_targets[x]["raw_index"]),
            )

        try:
            test_targets_features.append(fasttext[human_formula.lower()])
            fasttext_supported_test_targets_formulas.append(x)
        except:
            print("fasttext wrong test x skipped", x, human_formula)

    train_targets_formulas = fasttext_supported_train_targets_formulas
    test_targets_formulas = fasttext_supported_test_targets_formulas
    train_targets_vecs = np.array(train_targets_features)
    test_targets_vecs = np.array(test_targets_features)
    assert len(train_targets_formulas) == len(train_targets_vecs)
    assert len(test_targets_formulas) == len(test_targets_vecs)

    train_targets_vecs = train_targets_vecs / (
        np.linalg.norm(
            train_targets_vecs,
            axis=-1,
            keepdims=True,
        )
    )
    test_targets_vecs = test_targets_vecs / (
        np.linalg.norm(
            test_targets_vecs,
            axis=-1,
            keepdims=True,
        )
    )
    all_distance = test_targets_vecs @ train_targets_vecs.T

    train_targets_recipes = [
        precursors_recommendator.train_targets[x] for x in train_targets_formulas
    ]

    all_pres_predict, _ = precursors_recommendator.recommend_precursors_by_similarity(
        test_targets_formulas=test_targets_formulas,
        train_targets_recipes=train_targets_recipes,
        all_distance=all_distance,
        top_n=top_n,
        strategy="naive_common",
    )

    return all_pres_predict, test_targets_formulas


def recommend_w_freq(
    precursors_recommendator,
    test_targets_formulas,
    top_n=10,
):
    all_pres_predict = []
    precursor_frequencies = precursors_recommendator.precursor_frequencies
    common_eles = set(["C", "H", "O", "N"])
    nonvolatile_nonmetal_eles = {
        "P",
        "S",
        "Se",
    }

    for i, x in enumerate(test_targets_formulas):
        # prediction for precursors
        eles_x = set(get_elements_in_formula(x))
        effective_eles_x = eles_x & set(precursor_frequencies.keys())

        pres_multi_predicts = []
        pres_candidates_by_ele = [
            precursor_frequencies[ele] for ele in effective_eles_x
        ]
        if len(effective_eles_x & nonvolatile_nonmetal_eles) > 0:
            pres_candidates_by_ele_wo_nonmetal = [
                precursor_frequencies[ele]
                for ele in (effective_eles_x - nonvolatile_nonmetal_eles)
            ]
        else:
            # no need to repeat iteration based on
            # pres_candidates_by_ele if no extra non-metal element
            pres_candidates_by_ele_wo_nonmetal = []
        front_p_min = 10
        front_p_max = front_p_min
        for p_by_e in pres_candidates_by_ele:
            if len(p_by_e) > front_p_max:
                front_p_max = len(p_by_e)
        # front_p_min = front_p_max
        for front_p in range(front_p_min, front_p_max + 1):
            # get candidates in front first to reduce computational cost
            pres_candidates = []
            pres_probabilities = []
            for comb_i, pre_comb in enumerate(
                itertools.chain(
                    itertools.product(
                        *[p_by_e[:front_p] for p_by_e in pres_candidates_by_ele]
                    ),
                    itertools.product(
                        *[
                            p_by_e[:front_p]
                            for p_by_e in pres_candidates_by_ele_wo_nonmetal
                        ]
                    ),
                )
            ):
                # It is safe to presume the first term (two precursors,
                # one for metal, one for nonmetal, comb_i==0)
                # has the largest frequency because the precursor
                # with both metal and nonmetal has low frequency,
                # which is always lower than the product of two
                # common precursors of the metal and the nonmetal
                # precursors recommended or not

                # make sure no duplication
                pres_formulas = tuple(sorted(set([x["formula"] for x in pre_comb])))
                if pres_formulas in {x["precursors"] for x in pres_candidates}:
                    # is_recommended = True
                    continue

                # element matched or first attempt using all common precursors
                pres_eles = set(sum([x["elements"] for x in pre_comb], []))
                if (
                    pres_eles.issubset(eles_x | common_eles)
                    and eles_x.issubset(
                        pres_eles
                        | {
                            "O",
                            "H",
                        }
                    )
                ) or (comb_i == 0):
                    pres_prob = np.prod([x["frequency"] for x in pre_comb])
                    pres_candidates, pres_probabilities = add_to_sorted_list(
                        items=pres_candidates,
                        values=pres_probabilities,
                        new_item={
                            "precursors": pres_formulas,
                            "probability": pres_prob,
                            "elements": pres_eles,
                        },
                        new_value=pres_prob,
                    )
                    if len(pres_candidates) > top_n:
                        # sorting in add_to_sorted_list is from low to high
                        pres_candidates.pop(0)
                        pres_probabilities.pop(0)

            # check if candidates are sufficient
            if len(pres_candidates) >= top_n or front_p >= front_p_max:
                # sort candidates by probability
                pres_candidates = sorted(
                    pres_candidates,
                    key=lambda x: x["probability"],
                    reverse=True,
                )
                pres_multi_predicts = [
                    tuple(sorted(pres_cand["precursors"]))
                    for pres_cand in pres_candidates[:top_n]
                ]
                break

        all_pres_predict.append(pres_multi_predicts)

    return all_pres_predict


if __name__ == "__main__":
    use_file_as_stdout("../generated/output.txt")
    run_recommendations()

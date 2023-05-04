import json
from pathlib import Path
import numpy as np
from pprint import pprint
import os
from typing import (
    Union,
    List,
    Tuple,
    Dict,
    Optional,
)

from pymatgen.core import Composition, Element

# TODO: add reaction completer later
# import reaction_completer

from SynthesisSimilarity.core import model_utils
from SynthesisSimilarity.core import callbacks
from SynthesisSimilarity.scripts_utils.train_utils import load_precursor_frequencies

# from SynthesisPredictorAPI.PrecursorsPredictor.load_utils import load_precursor_frequencies
from SynthesisSimilarity.core.utils import formula_to_array
from SynthesisSimilarity.core.utils import array_to_formula
from SynthesisSimilarity.core.utils import get_elements_in_formula

# TODO: move parts from utils to mat_utils
# from SynthesisPredictorAPI.PrecursorsPredictor.mat_utils import formula_to_array
# from SynthesisPredictorAPI.PrecursorsPredictor.mat_utils import get_elements_in_formula
from SynthesisSimilarity.core.mat_featurization import featurize_list_of_composition

# from SynthesisPredictorAPI.PrecursorsPredictor.mat_utils import featurize_list_of_composition
from SynthesisSimilarity.scripts_utils.recommendation_utils import (
    collect_targets_in_reactions,
)

import reaction_completer

import pdb


all_eles = set(Element.__members__.keys())
all_extended_metal_eles = set(
    filter(
        lambda x: (Element(x).is_metal or Element(x).is_metalloid),
        all_eles,
    )
)


class PrecursorsRecommendation(object):
    def __init__(
        self,
        model_dir: str,
        freq_path: str,
        data_path: str,
        all_to_knowledge_base: str = False,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.freq_path = freq_path
        self.data_path = data_path
        self.load_model()
        self.load_pre_freq()
        self.load_data(all_to_knowledge_base=all_to_knowledge_base)

    def load_model(self):
        # # Important to allow_gpu_growth for multiple processors
        # utils.allow_gpu_growth()
        self.framework_model, self.model_config = model_utils.load_framework_model(
            self.model_dir
        )

        self.all_elements = self.model_config["all_eles"]
        self.mat_feature_len = self.model_config["mat_feature_len"]
        self.featurizer_type = self.model_config["featurizer_type"]
        self.max_mats_num = self.model_config["max_mats_num"]

        self.predict_precursor_callback = callbacks.PredictPrecursorsCallback(
            all_elements=self.all_elements,
            # all_ions=all_ions,
            mat_feature_len=self.mat_feature_len,
            test_data=None,
            output_thresh=0.0,
            featurizer_type=self.featurizer_type,
        )

    def load_data(
        self,
        uncommon_reactions_only: bool = False,
        all_to_knowledge_base: str = False,
    ):
        local_data = np.load(self.data_path, allow_pickle=True)
        print("local_data.keys()", list(local_data.keys()))
        if all_to_knowledge_base:
            self.train_reactions = (
                list(local_data["train_reactions"])
                + list(local_data["val_reactions"])
                + list(local_data["test_reactions"])
            )
        else:
            self.train_reactions = list(local_data["train_reactions"])

        # TODO: remove uncommon_reactions_only
        (
            self.train_targets,
            self.train_targets_formulas,
            self.train_targets_features,
        ) = collect_targets_in_reactions(
            self.train_reactions,
            self.all_elements,
            self.common_precursors_set,
            exclude_common_precursors=uncommon_reactions_only,
        )

        self.train_targets_recipes = [
            self.train_targets[x] for x in self.train_targets_formulas
        ]

        # TP similarity
        # train_targets_features is pre-transformed features
        # TODO: need to convert self.train_targets_features in collect_targets_in_reactions
        self.train_targets_vecs = self.framework_model.get_mat_vector(
            np.array(self.train_targets_features)
        ).numpy()

        self.train_targets_vecs = self.train_targets_vecs / (
            np.linalg.norm(self.train_targets_vecs, axis=-1, keepdims=True)
        )

    def load_pre_freq(self):
        # TODO: we just need common precursor here?
        # load_common_precursors and regenerate the formulas following order of all_elements
        self.precursor_frequencies = load_precursor_frequencies(
            all_elements=self.all_elements,
            file_path=self.freq_path,
        )
        self.common_precursors = {
            ele: self.precursor_frequencies[ele][0]
            for ele in self.precursor_frequencies
        }
        self.common_precursors_set = set(
            [
                self.precursor_frequencies[ele][0]["formula"]
                for ele in self.precursor_frequencies
            ]
        )

    def recommend_precursors(
        self,
        target_formula: Union[str, List[str]],
        top_n: int = 1,
        validate_first_attempt: bool = True,
        # TODO: add this to reaction_utils
        # validate_reaction: bool = True,
        recommendation_strategy="SynSim_conditional",
        precursors_not_available: Optional[set] = None,
    ):
        if isinstance(target_formula, str):
            test_targets_formulas = [target_formula]
        else:
            test_targets_formulas = target_formula

        # get target_candidate_normal_vecs
        # TODO: should this test_targets_compositions be ndarray?
        test_targets_compositions = [
            formula_to_array(formula, self.all_elements)
            for formula in test_targets_formulas
        ]
        test_targets_features = featurize_list_of_composition(
            comps=test_targets_compositions,
            ele_order=self.all_elements,
            featurizer_type=self.featurizer_type,
        )

        # TP similarity
        # train_targets_features is pre-transformed features
        # TODO: convert test_targets_features to np in advance
        test_targets_vecs = self.framework_model.get_mat_vector(
            np.array(test_targets_features)
        ).numpy()

        test_targets_vecs = test_targets_vecs / (
            np.linalg.norm(test_targets_vecs, axis=-1, keepdims=True)
        )

        all_distance = test_targets_vecs @ self.train_targets_vecs.T
        # TODO: use dict instead of list later
        all_distance_by_formula = {
            test_targets_formulas[i]: all_distance[i]
            for i in range(len(test_targets_formulas))
        }

        if recommendation_strategy == "SynSim_conditional":
            (
                all_pres_predict,
                all_rxns_predict,
            ) = self.recommend_precursors_by_similarity(
                test_targets_formulas=test_targets_formulas,
                test_targets_compositions=test_targets_compositions,
                test_targets_features=test_targets_features,
                train_targets_recipes=self.train_targets_recipes,
                all_distance=all_distance,
                top_n=top_n,
                validate_first_attempt=validate_first_attempt,
                # validate_reaction=validate_reaction,
                strategy="conditional",
                precursors_not_available=precursors_not_available,
            )
        elif recommendation_strategy == "SynSim_naive":
            (
                all_pres_predict,
                all_rxns_predict,
            ) = self.recommend_precursors_by_similarity(
                test_targets_formulas=test_targets_formulas,
                test_targets_compositions=test_targets_compositions,
                test_targets_features=test_targets_features,
                train_targets_recipes=self.train_targets_recipes,
                all_distance=all_distance,
                top_n=top_n,
                validate_first_attempt=validate_first_attempt,
                # validate_reaction=validate_reaction,
                strategy="naive_common",
            )
        else:
            raise NotImplementedError

        # TODO: better to output as a dict {'target', 'precursors', ...}

        return all_pres_predict

    def recommend_precursors_by_similarity(
        self,
        test_targets_formulas: List[str],
        train_targets_recipes: List[dict],
        all_distance: np.ndarray,
        test_targets_compositions: Optional[np.ndarray] = None,
        test_targets_features: Optional[np.ndarray] = None,
        top_n: int = 1,
        validate_first_attempt: bool = False,
        path_log="../generated/distance/dist_reaction.txt",
        # validate_reaction: bool = True,
        common_eles: tuple = ("C", "H", "O", "N"),
        strategy="conditional",
        precursors_not_available: Optional[set] = None,
    ):
        # TODO: allow and test both naive and conditional here
        #   and remove naive later
        # TODO: add reaction validation later
        all_pres_predict = []
        all_rxns_predict = []
        ref_precursors_comp = {}
        ref_materials_comp = {}
        common_eles = set(common_eles)
        if precursors_not_available is None:
            precursors_not_available = set()

        # # logs
        # all_logs = []
        # dir_all_logs = os.path.join(
        #     *Path(path_log).parts[0:-2],
        #     'all_logs'
        # )
        # if not os.path.exists(dir_all_logs):
        #     os.makedirs(dir_all_logs)

        # # Important to allow_gpu_growth for multiple processors
        # utils.allow_gpu_growth()

        # for x in random.choices(test_targets_formulas, k=1000):
        print("len(test_targets_formulas)", len(test_targets_formulas))
        print("top_n", top_n)
        for x_index, x in enumerate(test_targets_formulas):
            if x_index % 10 == 0:
                print(
                    "x_index: {} out of {}".format(x_index, len(test_targets_formulas))
                )
            most_similar_y_index = np.argsort(all_distance[x_index, :])[::-1]

            # all_logs.append(
            #     {
            #         'target_formula': x,
            #         'similar_materials': [],
            #         'precursors_predicts': [],
            #     }
            # )

            # prediction for precursors
            eles_x = set(get_elements_in_formula(x))
            pres_candidates = []
            zero_composition = np.zeros(
                shape=(len(self.all_elements),),
                dtype=np.float32,
            )
            for y_index in most_similar_y_index[: 300 * top_n]:
                # 30*top_n is sufficient to get recommendations
                # no need to check all ref targets
                # all recipes for each target used by freq here
                pres_candidates.extend(
                    [
                        item[0]
                        for item in train_targets_recipes[y_index]["pres"].most_common()
                    ]
                )
                # all_logs[-1]['similar_materials'].append(
                #     {
                #         'y_index': int(y_index),
                #         'similarity': float(all_distance[x_index, y_index]),
                #         # 'precursors_dict': train_targets[y]['pres'],
                #         'formula': array_to_formula(train_targets_recipes[y_index]['comp'], self.all_elements),
                #         'precursors': [item[0] for item in train_targets_recipes[y_index]['pres'].most_common()],
                #         'raw_index': list(train_targets_recipes[y_index]['raw_index']),
                #         'pres_raw_index': list(train_targets_recipes[y_index]['pres_raw_index'].items()),
                #     }
                # )

            # reformat pres_candidates (formula -> eles)
            pres_candidates, ref_precursors_comp = self._reformat_precursors(
                pres_candidates=pres_candidates,
                ref_precursors_comp=ref_precursors_comp,
            )

            pres_multi_predicts = []

            # add common precursors
            if top_n > 1:
                pres_predict = self.common_precursors_recommendation(
                    eles_target=eles_x,
                    common_precursors=self.common_precursors,
                    common_eles=common_eles,
                    validate_first_attempt=validate_first_attempt,
                    # validate_reaction=validate_reaction,
                    target_formula=x,
                    ref_materials_comp=ref_materials_comp,
                )
                if pres_predict is not None:
                    pres_multi_predicts.append(pres_predict)
                    # all_logs[-1]['precursors_predicts'].append(
                    #     {
                    #         'i_pres_candidates': None,
                    #         'pres': None,
                    #         'pres_predict': pres_predict,
                    #     }
                    # )

            pres_conditional_tried = set()
            for i in range(len(pres_candidates)):
                pres_predict = set()
                eles_covered = {
                    "O",
                    "H",
                }
                precursors_conditional = []

                pres = pres_candidates[i]
                for p in pres.values():
                    if p["formula"] in pres_predict:
                        continue
                    if p["elements"].issubset(eles_x | common_eles):
                        pres_predict.add(p["formula"])
                        eles_covered |= p["elements"]
                        precursors_conditional.append(p["composition"])

                # print('pres_predict 1', pres_predict)
                if not eles_x.issubset(eles_covered):
                    if tuple(sorted(pres_predict)) in pres_conditional_tried:
                        continue
                    pres_conditional_tried.add(tuple(sorted(pres_predict)))
                    if strategy == "conditional":
                        for _ in range(len(pres_predict), self.max_mats_num - 1):
                            precursors_conditional.append(zero_composition)
                        (
                            pre_lists_pred,
                            pre_str_lists_pred,
                        ) = self.predict_precursor_callback.predict_precursors(
                            self.framework_model,
                            target_compositions=np.expand_dims(
                                test_targets_compositions[x_index], axis=0
                            ),
                            target_features=np.expand_dims(
                                test_targets_features[x_index], axis=0
                            ),
                            precursors_conditional=np.expand_dims(
                                precursors_conditional, axis=0
                            ),
                            to_print=False,
                        )
                        for ele in eles_x - eles_covered:
                            if eles_x.issubset(eles_covered):
                                # done
                                break
                            for (p_comp_prob, p_f_prob) in zip(
                                pre_lists_pred[0], pre_str_lists_pred[0]
                            ):
                                p_comp = p_comp_prob["composition"]
                                p_f = p_f_prob[0]
                                p_eles = set(np.array(self.all_elements)[p_comp > 0])
                                if p_f in pres_predict:
                                    continue
                                if p_f in precursors_not_available:
                                    continue
                                if ele in p_eles and p_eles.issubset(
                                    eles_x | common_eles
                                ):
                                    pres_predict.add(p_f)
                                    eles_covered |= p_eles
                                    break
                    elif strategy == "naive_common":
                        # add metal/metalloid sources first
                        for ele in (eles_x - eles_covered) & all_extended_metal_eles:
                            if ele in self.common_precursors:
                                pres_predict.add(self.common_precursors[ele]["formula"])
                                eles_covered |= set(
                                    self.common_precursors[ele]["elements"]
                                )
                        # add nonvolatile nonmetal elements if necessary
                        for ele in eles_x - eles_covered:
                            if ele in self.common_precursors:
                                pres_predict.add(self.common_precursors[ele]["formula"])
                                eles_covered |= set(
                                    self.common_precursors[ele]["elements"]
                                )

                if not eles_x.issubset(eles_covered):
                    continue

                pres_predict = tuple(sorted(pres_predict))
                # precursors recommended or not
                if pres_predict in pres_multi_predicts:
                    # is_recommended = True
                    continue
                pres_multi_predicts.append(pres_predict)
                # all_logs[-1]['precursors_predicts'].append(
                #     {
                #         'i_pres_candidates': i,
                #         'pres': [p['formula'] for p in pres.values()],
                #         'pres_predict': pres_predict,
                #     }
                # )

                if len(pres_multi_predicts) >= top_n:
                    break

            all_pres_predict.append(pres_multi_predicts)
            # all_logs[-1]['has_at_least_one_reported'] = (
            #     len(set(pres_multi_predicts) & set(all_logs[-1]['precursors_true'])) > 0
            # )

        # with open(
        #     os.path.join(
        #         dir_all_logs,
        #         'all_logs_precursor_recommendation_{pid}.json'.format(
        #             pid=os.getpid()
        #         )
        #     ), 'w'
        # ) as fw:
        #     json.dump(all_logs, fw, indent=2)

        return all_pres_predict, all_rxns_predict

    def complete_precursors_conditional(
        self,
        test_targets_formulas: List[str],
        train_targets_recipes: List[dict],
        all_distance: np.ndarray,
        common_precursors: dict,
        top_n: int = 1,
        validate_first_attempt: bool = True,
        validate_reaction: bool = True,
        common_eles: tuple = ("C", "H", "O", "N"),
    ):
        ...

    def naive_recommendation(
        self,
        test_targets_formulas: List[str],
        train_targets_recipes: List[dict],
        all_distance: np.ndarray,
        common_precursors: dict,
        top_n: int = 1,
        validate_first_attempt: bool = True,
        validate_reaction: bool = True,
        common_eles: tuple = ("C", "H", "O", "N"),
    ):
        all_pres_predict = []
        all_rxns_predict = []
        ref_precursors_comp = {}
        ref_materials_comp = {}
        common_eles = set(common_eles)

        # for x in random.choices(test_targets_formulas, k=1000):
        print("len(test_targets_formulas)", len(test_targets_formulas))
        print("top_n", top_n)
        for x_index, x in enumerate(test_targets_formulas):
            if x_index % 10 == 0:
                print(
                    "x_index: {} out of {}".format(x_index, len(test_targets_formulas))
                )
            most_similar_y_index = np.argsort(all_distance[x_index, :])[::-1]

            # prediction for precursors
            eles_x = set(get_elements_in_formula(x))
            pres_candidates = []
            for y_index in most_similar_y_index[: 30 * top_n]:
                # 30*top_n is sufficient to get recommendations
                # no need to check all ref targets
                # all recipes for each target used by freq here
                pres_candidates.extend(
                    [
                        item[0]
                        for item in train_targets_recipes[y_index]["pres"].most_common()
                    ]
                )

            # reformat pres_candidates (formula -> eles)
            pres_candidates, ref_precursors_comp = self._reformat_precursors(
                pres_candidates=pres_candidates,
                ref_precursors_comp=ref_precursors_comp,
            )

            pres_multi_predicts = []
            rxns_multi_predicts = []

            # add common precursors
            if top_n > 1:
                pres_predict, rxn_predict = self.common_precursors_recommendation(
                    eles_target=eles_x,
                    common_precursors=common_precursors,
                    common_eles=common_eles,
                    validate_first_attempt=validate_first_attempt,
                    validate_reaction=validate_reaction,
                    target_formula=x,
                    ref_materials_comp=ref_materials_comp,
                )
                if pres_predict is not None:
                    pres_multi_predicts.append(pres_predict)
                    rxns_multi_predicts.append(rxn_predict)

            for i in range(len(pres_candidates)):
                pres_predict = set()
                eles_covered = set()
                # find pres in rest of candidates
                for j, pres in enumerate(pres_candidates[i:]):
                    if eles_x.issubset(eles_covered):
                        # done
                        break
                    # collect precursors w/ corresponding elements
                    for ele in eles_x - eles_covered:
                        if eles_x.issubset(eles_covered):
                            # done
                            break
                        for p in pres.values():
                            if p["formula"] in pres_predict:
                                continue
                            if ele in p["elements"] and p["elements"].issubset(
                                eles_x | common_eles
                            ):
                                pres_predict.add(p["formula"])
                                eles_covered |= p["elements"]

                if not eles_x.issubset(eles_covered):
                    continue

                pres_predict = tuple(sorted(pres_predict))
                # precursors recommended or not
                if pres_predict in pres_multi_predicts:
                    # is_recommended = True
                    continue

                if validate_reaction:
                    rxn_predict = self.get_reaction(
                        target_formula=x,
                        precursors_formulas=pres_predict,
                        ref_materials_comp=ref_materials_comp,
                    )
                    if rxn_predict is None:
                        continue
                else:
                    rxn_predict = None

                pres_multi_predicts.append(pres_predict)
                rxns_multi_predicts.append(rxn_predict)
                if len(pres_multi_predicts) >= top_n:
                    break

            all_pres_predict.append(pres_multi_predicts)
            all_rxns_predict.append(rxns_multi_predicts)

        return all_pres_predict, all_rxns_predict

    def _reformat_precursors(
        self,
        pres_candidates: List[List[str]],
        ref_precursors_comp: Dict[str, set],
    ):
        reformated_pres_candidates = []
        for i in range(len(pres_candidates)):
            pres = pres_candidates[i]
            pres_info = {}
            for p in pres:
                if p in ref_precursors_comp:
                    p_comp = ref_precursors_comp[p]
                else:
                    p_comp = formula_to_array(p, self.all_elements)
                    ref_precursors_comp[p] = p_comp
                # TODO: can i also make self.all_elements as ndarray?
                p_eles = set(np.array(self.all_elements)[p_comp > 0])
                pres_info[p] = {
                    "formula": p,
                    "composition": p_comp,
                    "elements": p_eles,
                }
            reformated_pres_candidates.append(pres_info)

        return reformated_pres_candidates, ref_precursors_comp

    def common_precursors_recommendation(
        self,
        eles_target: set,
        common_precursors: dict,
        common_eles: set,
        validate_first_attempt: bool = True,
        validate_reaction: bool = True,
        target_formula: Optional[str] = None,
        ref_materials_comp: Optional[Dict[str, dict]] = None,
    ):
        common_pres = []
        for ele in eles_target:
            if ele in common_precursors:
                common_pres.append(common_precursors[ele])
        pres_eles = set(sum([x["elements"] for x in common_pres], []))
        pres_formulas = tuple(sorted(set([x["formula"] for x in common_pres])))

        if validate_first_attempt:
            if not (
                pres_eles.issubset(eles_target | common_eles)
                and eles_target.issubset(
                    pres_eles
                    | {
                        "O",
                        "H",
                    }
                )
            ):
                pres_formulas = None

        # TODO: add validate_reaction later
        # if validate_reaction and (pres_formulas is not None):
        #     rxn_predict = self.get_reaction(
        #         target_formula=target_formula,
        #         precursors_formulas=pres_formulas,
        #         ref_materials_comp=ref_materials_comp,
        #     )
        #     if rxn_predict is None:
        #         pres_formulas = None
        # else:
        #     rxn_predict = None

        return pres_formulas

    def get_reaction(
        self,
        target_formula: str,
        precursors_formulas: Union[List[str], Tuple[str]],
        ref_materials_comp: Dict[str, dict],
    ):
        """

        :param target_formula:
        :param precursors_formulas:
        :param ref_materials_comp: will be updated inside function
        :return:
        """
        target_comp = self._reformat_material(
            mat_formula=target_formula,
            ref_materials_comp=ref_materials_comp,
        )
        targets_comps = [
            {
                **target_comp,
                "elements_vars": {},
                "additives": [],
            }
        ]

        precursors_comps = []
        for precursor_formula in precursors_formulas:
            precursor_comp = self._reformat_material(
                mat_formula=precursor_formula,
                ref_materials_comp=ref_materials_comp,
            )
            precursors_comps.append(precursor_comp)

        reactions = reaction_completer.balance_recipe(
            precursors=precursors_comps,
            targets=targets_comps,
        )
        if len(reactions) == 0:
            return None
        else:
            assert len(reactions) == 1
            return reactions[0]

    def _reformat_material(
        self,
        mat_formula: str,
        ref_materials_comp: Dict[str, dict],
    ):
        """

        :param mat_formula:
        :param ref_materials_comp: will be updated inside function
        :return:
        """
        if mat_formula in ref_materials_comp:
            mat_comp = ref_materials_comp[mat_formula]
        else:
            mat_comp = self._material_as_dataset_format(mat_formula=mat_formula)
            ref_materials_comp[mat_formula] = mat_comp
        return mat_comp

    def _material_as_dataset_format(
        self,
        mat_formula: str,
    ):
        elements = Composition(mat_formula).as_dict()
        mat_comp = {
            "material_formula": mat_formula,
            "material_string": mat_formula,
            "composition": [
                {"formula": mat_formula, "elements": elements, "amount": "1.0"}
            ],
        }
        return mat_comp


if __name__ == "__main__":
    print("-------Loading models for precursors prediction-------")
    precursors_recommendator = PrecursorsRecommendation(
        model_dir="../models/SynthesisRecommendation",
        freq_path="../rsc/pre_count_normalized_by_rxn_ss.json",
        data_path="../rsc/data_split.npz",
    )
    print("-------Finished loading models for precursors prediction-------")

    targets_formulas = [
        "LiFePO4",
        "LiNi0.333Mn0.333Co0.333O2",
    ]
    top_n = 10
    all_pres_predict = precursors_recommendator.recommend_precursors(
        target_formula=targets_formulas,
        top_n=top_n,
    )
    print("all_pres_predict")
    pprint(all_pres_predict)

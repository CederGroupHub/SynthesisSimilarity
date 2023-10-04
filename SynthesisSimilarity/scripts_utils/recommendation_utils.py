import collections
import bisect

from SynthesisSimilarity.core import utils


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def collect_targets_in_reactions(
    train_reactions,
    all_elements,
    common_precursors_set,
    exclude_common_precursors=False,
):
    # TODO: clean the name of train_xx
    raw_indices_train = set()
    train_targets = {}
    for r in train_reactions:
        tar_f = utils.array_to_formula(r["target_comp"][0], all_elements)
        if len(r["target_comp"]) > 1:
            print("len(r['target_comp'])", len(r["target_comp"]))
        assert len(r["target_comp"]) == 1, "Reaction not expanded"
        for x in r["precursors_comp"]:
            assert len(x) == 1, "Reaction not expanded"
        pre_fs = set(
            [utils.array_to_formula(x[0], all_elements) for x in r["precursors_comp"]]
        )
        assert len(pre_fs) == len(
            r["precursors_comp"]
        ), "len(pre_fs) != len(r['precursors_comp'])"
        pre_fs = tuple(sorted(pre_fs))
        if exclude_common_precursors and set(pre_fs).issubset(common_precursors_set):
            continue
        if tar_f not in train_targets:
            train_targets[tar_f] = {
                "comp": r["target_comp"][0],
                "comp_fea": r["target_comp_featurized"][0],
                "pres": collections.Counter(),
                "syn_type": collections.Counter(),
                "syn_type_pres": collections.Counter(),
                "raw_index": set(),
                "is_common": collections.Counter(),
                "pres_raw_index": {},
            }
        train_targets[tar_f]["pres"][pre_fs] += 1
        train_targets[tar_f]["raw_index"].add(r["raw_index"])
        if pre_fs not in train_targets[tar_f]["pres_raw_index"]:
            train_targets[tar_f]["pres_raw_index"][pre_fs] = []
        train_targets[tar_f]["pres_raw_index"][pre_fs].append(r["raw_index"])
        raw_indices_train.add(r["raw_index"])
        if set(pre_fs).issubset(common_precursors_set):
            train_targets[tar_f]["is_common"]["common"] += 1
        else:
            train_targets[tar_f]["is_common"]["uncommon"] += 1
        if "synthesis_type" in r:
            train_targets[tar_f]["syn_type"][r["synthesis_type"]] += 1
            train_targets[tar_f]["syn_type_pres"][(r["synthesis_type"],) + pre_fs] += 1

    train_targets_formulas = list(train_targets.keys())
    # TODO: shall we make this np.ndarray?
    train_targets_features = [
        train_targets[x]["comp_fea"] for x in train_targets_formulas
    ]
    print("len(train_targets)", len(train_targets))
    return train_targets, train_targets_formulas, train_targets_features


def add_to_sorted_list(items, values, new_item, new_value):
    new_idx = bisect.bisect_left(values, new_value)
    items.insert(new_idx, new_item)
    values.insert(new_idx, new_value)
    return items, values

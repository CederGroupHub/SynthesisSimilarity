"""
    Explore the relationship of Tar_1 - Pre_1 vs Tar_2 - Pre_2,
    similar to king - man ~ queen - woman from word2vec.
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


import numpy as np
from pprint import pprint
from typing import List

from SynthesisSimilarity.core import model_utils
from SynthesisSimilarity.core import callbacks
from sklearn import decomposition
import matplotlib.pyplot as plt


__author__ = "Tanjin He"
__maintainer__ = "Tanjin He"
__email__ = "tanjin_he@berkeley.edu"


def explore_reaction_relationship(
    vec_cb,
    framework_model,
):
    target_formulas = [
        "InCuO2",
        "YCuO2",
        "Al2CuO4",
        "FeCuO2",
        "BaIn2O4",
        "Ba3Y4O9",
        "BaAl2O4",
        "BaFeO3",
        "TiIn2O5",
        "Ti3Y2O9",
        "Ti3Al2O9",
        "Ti3Fe2O9",
    ]
    precursor_formulas = [
        "In2O3",
        "Y2O3",
        "Al2O3",
        "Fe2O3",
        "In2O3",
        "Y2O3",
        "Al2O3",
        "Fe2O3",
        "In2O3",
        "Y2O3",
        "Al2O3",
        "Fe2O3",
    ]

    decoder = vec_cb.get_decoder(framework_model)
    project_w_attention = False
    if decoder.predict_precursor_under_mask:
        project_w_attention = True
    pre_vec_mapping = vec_cb.get_pre_vec_mapping(
        decoder=decoder,
    )
    precursor_vecs = vec_cb.get_pre_vecs_from_mapping(
        precursor_formulas=precursor_formulas,
        pre_vec_mapping=pre_vec_mapping,
    )
    tar_vec_mapping, _ = vec_cb.get_tar_vec_mapping(
        target_formulas=target_formulas,
        model=framework_model,
        project_w_attention=project_w_attention,
        decoder=decoder,
        max_mats_num=vec_cb.max_mats_num,
    )
    diff_vecs = []
    for i in range(len(target_formulas)):
        diff_vecs.append(tar_vec_mapping[target_formulas[i]] - precursor_vecs[i])

    diff_vecs = np.array(diff_vecs)
    pca = decomposition.PCA(
        n_components=2,
        svd_solver="arpack",
    )
    pca.fit(diff_vecs)
    diff_vecs_2d = pca.transform(diff_vecs)

    print("diff_vecs_2d")
    pprint(diff_vecs_2d)

    plot_relation_shift(
        target_formulas=target_formulas,
        precursor_formulas=precursor_formulas,
        diff_vecs=diff_vecs_2d,
    )


def plot_relation_shift(
    target_formulas: List[str],
    precursor_formulas: List[str],
    diff_vecs: np.ndarray,
):

    jitter_scale = 0.05
    target_text = {
        "InCuO2": "InCuO$_2$",
        "YCuO2": "YCuO$_2$",
        "Al2CuO4": "Al$_2$CuO$_4$",
        "FeCuO2": "FeCuO$_2$",
        "BaIn2O4": "BaIn$_2$O$_4$",
        "Ba3Y4O9": "Ba$_3$Y$_4$O$_9$",
        "BaAl2O4": "BaAl$_2$O$_4$",
        "BaFeO3": "BaFeO$_3$",
        "TiIn2O5": "TiIn$_2$O$_5$",
        "Ti3Y2O9": "Ti$_3$Y$_2$O$_9$",
        "Ti3Al2O9": "Ti$_3$Al$_2$O$_9$",
        "Ti3Fe2O9": "Ti$_3$Fe$_2$O$_9$",
    }
    target_text_jitter = {
        "InCuO2": np.array([-2.2, 0.0]) * jitter_scale,
        "YCuO2": np.array([-2.5, 0]) * jitter_scale,
        "Al2CuO4": np.array([-3.5, -1]) * jitter_scale,
        "FeCuO2": np.array([-4, -5]) * jitter_scale,
        "BaIn2O4": np.array([-1, -3.0]) * jitter_scale,
        "Ba3Y4O9": np.array([-1, -3.0]) * jitter_scale,
        "BaAl2O4": np.array([-3.5, 0.0]) * jitter_scale,
        "BaFeO3": np.array([-1.5, -2.5]) * jitter_scale,
        "TiIn2O5": np.array([-3.5, 0.5]) * jitter_scale,
        "Ti3Y2O9": np.array([-5, -3]) * jitter_scale,
        "Ti3Al2O9": np.array([-3.2, 0]) * jitter_scale,
        "Ti3Fe2O9": np.array([-3, 0]) * jitter_scale,
    }
    target_text_angles = {
        "InCuO2": -85,
        "YCuO2": -90,
        "Al2CuO4": -80,
        "FeCuO2": -70,
        "BaIn2O4": 32,
        "Ba3Y4O9": 40,
        "BaAl2O4": 32,
        "BaFeO3": 18,
        "TiIn2O5": -10,
        "Ti3Y2O9": -15,
        "Ti3Al2O9": -15,
        "Ti3Fe2O9": -10,
    }
    precursor_init_points = {
        "Fe2O3": np.array([-3, 3]) * jitter_scale,
        "Al2O3": np.array([-1.5, 1.5]) * jitter_scale,
        "In2O3": np.array([0, 0]) * jitter_scale,
        "Y2O3": np.array([1.5, -1.5]) * jitter_scale,
    }
    precursor_text = {
        "In2O3": "In$_2$O$_3$",
        "Fe2O3": "Fe$_2$O$_3$",
        "Al2O3": "Al$_2$O$_3$",
        "Y2O3": "Y$_2$O$_3$",
    }
    precursor_text_jitter = {
        "In2O3": np.array([1.5, -0.0]) * jitter_scale,
        "Fe2O3": np.array([-5.2, 1.2]) * jitter_scale,
        "Al2O3": np.array([-6.0, -1.7]) * jitter_scale,
        "Y2O3": np.array([-6.2, -1.0]) * jitter_scale,
    }
    category_text = {
        "TiO2": "React w/ TiO$_2$",
        "CuO": "React w/ CuO",
        "BaCO3": "React w/ BaCO$_3$",
    }
    category_text_points = {
        "TiO2": np.array([2.3, -9.2]) * jitter_scale,
        "CuO": np.array([2, 4.0]) * jitter_scale,
        "BaCO3": np.array([-24, -3.2]) * jitter_scale,
    }
    category_text_angles = {
        "TiO2": -20,
        "CuO": -90,
        "BaCO3": 18,
    }
    category_colors = {
        "BaCO3": "slateblue",
        "CuO": "orchid",
        "TiO2": "seagreen",
    }
    target_colors = {
        "InCuO2": category_colors["CuO"],
        "YCuO2": category_colors["CuO"],
        "Al2CuO4": category_colors["CuO"],
        "FeCuO2": category_colors["CuO"],
        "BaIn2O4": category_colors["BaCO3"],
        "Ba3Y4O9": category_colors["BaCO3"],
        "BaAl2O4": category_colors["BaCO3"],
        "BaFeO3": category_colors["BaCO3"],
        "TiIn2O5": category_colors["TiO2"],
        "Ti3Y2O9": category_colors["TiO2"],
        "Ti3Al2O9": category_colors["TiO2"],
        "Ti3Fe2O9": category_colors["TiO2"],
    }
    precursor_colors = {
        "Fe2O3": "tab:blue",
        "Al2O3": "tab:green",
        "In2O3": "tab:purple",
        "Y2O3": "tab:orange",
    }
    precursor_markers = {
        "Fe2O3": "^",
        "Al2O3": "s",
        "In2O3": "p",
        "Y2O3": "o",
    }

    target_indices_to_plot = [target_formulas.index(x) for x in target_formulas]
    precursors_to_plot = list(
        set([precursor_formulas[t_i] for t_i in target_indices_to_plot])
    )

    fig = plt.figure(
        figsize=(12, 10),
        # constrained_layout=True,
    )
    ax = fig.add_subplot(111)

    # sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)

    for t_i in target_indices_to_plot:
        pre = precursor_formulas[t_i]
        pre_xy = precursor_init_points[pre]
        tar = target_formulas[t_i]
        tar_dxy = diff_vecs[t_i]
        tar_xy = tar_dxy + pre_xy
        tar_text = target_text[tar]
        tar_text_xy = tar_xy + target_text_jitter[tar]
        tar_text_angle = target_text_angles[tar]
        tar_color = target_colors[tar]
        ax.arrow(
            pre_xy[0],
            pre_xy[1],
            tar_dxy[0],
            tar_dxy[1],
            head_width=0.04,
            head_length=0.05,
            linewidth=2,
            facecolor=tar_color,
            edgecolor=tar_color,
            alpha=0.75,
            head_starts_at_zero=False,
            length_includes_head=True,
        )
        ax.text(
            tar_text_xy[0],
            tar_text_xy[1],
            tar_text,
            rotation=tar_text_angle,
            fontsize=28,
            color=tar_color,
        )

    for pre in precursors_to_plot:
        pre_xy = precursor_init_points[pre]
        pre_text = precursor_text[pre]
        pre_text_xy = pre_xy + precursor_text_jitter[pre]
        pre_color = precursor_colors[pre]
        pre_marker = precursor_markers[pre]
        ax.plot(
            pre_xy[0],
            pre_xy[1],
            marker=pre_marker,
            markersize=20,
            color=pre_color,
        )
        ax.text(
            pre_text_xy[0],
            pre_text_xy[1],
            pre_text,
            fontsize=28,
            color=pre_color,
        )

    for cat in category_text_points:
        cat_text_xy = category_text_points[cat]
        cat_text_angle = category_text_angles[cat]
        cat_color = category_colors[cat]
        cat_text = category_text[cat]
        ax.text(
            cat_text_xy[0],
            cat_text_xy[1],
            cat_text,
            fontsize=32,
            color=cat_color,
            rotation=cat_text_angle,
        )

    plt.xlabel("First principal component", size=36)
    plt.ylabel("Second principal component", size=36)
    ax.tick_params(axis="x", which="major", labelsize=26)
    ax.tick_params(axis="y", which="major", labelsize=26)
    plt.xlim(-1.22, 1.27)
    plt.ylim(-0.88, 1.2)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    path_save = "../generated/plots/relationship_shift.png"
    if not os.path.exists(os.path.dirname(path_save)):
        os.makedirs(os.path.dirname(path_save))
    plt.savefig(path_save, dpi=300)
    print("Figure saved to {}".format(os.path.dirname(path_save)))
    # plt.show()


if __name__ == "__main__":

    model_dir = "../models/SynthesisRecommendation"

    framework_model, model_config = model_utils.load_framework_model(model_dir)
    all_elements = model_config["all_eles"]
    max_mats_num = model_config["max_mats_num"]
    featurizer_type = model_config["featurizer_type"]

    # using callbacks
    vec_test = callbacks.VectorMathCallback(
        all_elements=all_elements,
        featurizer_type=featurizer_type,
        max_mats_num=max_mats_num,
        top_n=10,
        test_data=None,
    )

    # plot relationship
    explore_reaction_relationship(
        vec_cb=vec_test,
        framework_model=framework_model,
    )

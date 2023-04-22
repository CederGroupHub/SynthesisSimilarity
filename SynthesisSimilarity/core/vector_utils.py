import numpy as np
from typing import List

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def most_similar_by_vector(
    target_vec: np.ndarray,
    target_candidate_formulas: List[str],
    target_candidate_normal_vecs: np.ndarray,
    positive_vecs: List[np.ndarray],
    negative_vecs: List[np.ndarray],
    top_n=10,
):
    sum_vec = np.sum(
        [target_vec]
        + positive_vecs
        +[-v for v in negative_vecs],
        axis=0,
    )
    sum_vec = sum_vec/np.linalg.norm(sum_vec)
    all_similarity = sum_vec @ target_candidate_normal_vecs.T

    mat_idx_sort = np.argsort(all_similarity)[::-1]

    most_similar_mats = [
        target_candidate_formulas[idx]
        for idx in mat_idx_sort[:top_n]
    ]
    most_similar_scores = list(all_similarity[mat_idx_sort[:top_n]])

    return list(zip(most_similar_mats, most_similar_scores))

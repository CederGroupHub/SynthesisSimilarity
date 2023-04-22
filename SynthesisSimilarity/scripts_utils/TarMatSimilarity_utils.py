import collections
import numpy as np
import os
from pymatgen.core import Composition
from SynthesisSimilarity.core import model_utils
from SynthesisSimilarity.core import mat_featurization
from SynthesisSimilarity.scripts_utils import similarity_utils

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


class TarMatSimilarity(object):
    def __init__(self, model_dir):
        """
        Args:
            model_dir (str) - path to dir where the model is saved
        """
        (self.model, self.model_config) = similarity_utils.load_encoding_model(model_dir)

    def feature_vector(self, formula):
        """
        Args:
            formula (str)

        Returns:
            1d array of feature values (float)
        """
        mat_vector = self.get_mat_vector_by_formula(formula)
        return mat_vector

    def compare(self, formula1, formula2):
        """
        Args:
            formula1 (str)
            formula2 (str)

        Returns:
            similarity by cosine distance of two vectors
        """
        m1 = self.feature_vector(formula1)
        m2 = self.feature_vector(formula2)
        return self.cos_distance(m1, m2)

    def get_mat_vector_by_formula(self, formula):
        return self.get_mat_vector_by_formulas([formula])[0]

    def get_mat_vector_by_formulas(self, formulas):
        compositions = []
        for f in formulas:
            mat = Composition(f).as_dict()
            compositions.append(mat)
        return self.get_mat_vector_by_compositions(compositions)

    def get_mat_vector_by_compositions(self, compositions):
        comps = []
        for x in compositions:
            comps.append(similarity_utils.composition_to_array(x, self.model_config['all_eles']))
        comps = mat_featurization.featurize_list_of_composition(
            comps=comps,
            ele_order=self.model_config['all_eles'],
            featurizer_type=self.model_config['featurizer_type'],
            ion_order=self.model_config['all_ions'],
        )
        comps = np.array(comps)
        return self.model(comps)

    def cos_distance(self, vec_1, vec_2):
        dot = np.dot(vec_1, vec_2)
        norm_1 = np.linalg.norm(vec_1)
        norm_2 = np.linalg.norm(vec_2)
        return dot / (norm_1 * norm_2)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:07:28 2019

@author: chrisbartel
"""
import pdb

import pkgutil

if pkgutil.find_loader("matminer"):
    from matminer.featurizers.composition import ElementProperty
else:
    print(
        "Magpie encoding needs the package matminer and scikit-learn==1.0.2. "
        "You may want to install them with 'pip install matminer scikit-learn==1.0.2'."
    )


from pymatgen.core import Composition
import os
import numpy as np
import json
import multiprocessing as mp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


def _dist(m1, m2):
    """
    Args:
        m1 (1d-array) - feature vector 1
        m2 (1d-array) - feature vector 2

    Returns:
        Euclidean distance between vectors (float)
    """

    return np.sqrt(np.sum([(m1[i] - m2[i]) ** 2 for i in range(len(m1))]))


def _similarity(m1, m2):
    """
    Args:
        m1 (1d-array) - feature vector 1
        m2 (1d-array) - feature vector 2

    Returns:
        inverse distance (similarity) between two vectors (float)
    """

    return 1 / _dist(m1, m2)


class MatminerSimilarity(object):
    def __init__(
        self,
        path_to_imputer,
        path_to_scaler,
        data_source="magpie",
    ):
        """
        Args:
            path_to_imputer (str) - path to .pkl with SimpleImputer fit to MP
            path_to_scaler (str) - path to .pkl with StandardScaler fit to MP
            features (str) - 'magpie' ('pymatgen', etc. not implemented)
            data_source (str) - 'magpie' ('pymatgen', etc. not implemented)
            stats (list) - list of statistics (str) to manipulate features with

        Returns:
            list of features (str)
            loaded imputer and scaler

        """

        if data_source == "magpie":
            self.data_source = "magpie"
        else:
            raise NotImplementedError

        if os.path.exists(path_to_imputer):
            self.imputer = joblib.load(path_to_imputer)
        else:
            self.imputer = None
        if os.path.exists(path_to_scaler):
            self.scaler = joblib.load(path_to_scaler)
        else:
            self.scaler = None

    def feature_vector(self, formula):
        vector = ElementProperty.from_preset(self.data_source).featurize(
            Composition(formula)
        )
        vector = np.array(vector)
        return vector

    def feature_vector_normalized(self, formula):
        """
        Args:
            formula (str)

        Returns:
            1d array of feature values (float or NaN)
        """
        vector = self.feature_vector(formula)
        imp, sc = self.imputer, self.scaler
        vector = np.expand_dims(vector, axis=0)
        vector = imp.transform(vector)
        vector = sc.transform(vector)
        vector = vector.squeeze(axis=0)
        return vector

    def compare(self, formula1, formula2):
        """
        Args:
            formula1 (str)
            formula2 (str)

        Returns:
            inverse distance representation of similarity (float)
        """
        m1 = self.feature_vector_normalized(formula1)
        m2 = self.feature_vector_normalized(formula2)
        return _similarity(m1, m2)


def _get_feature_vector(obj, formula):
    return obj.feature_vector(formula)


class RegenerateScalerImputer(object):
    """
    For re-generating Scaler and Imputer when you e.g., change the feature or stats bases
    """

    def __init__(
        self,
        path_to_formulas,
        inputs={
            "data_source": "magpie",
            "path_to_scaler": "mp_scaler.pkl",
            "path_to_imputer": "mp_imputer.pkl",
        },
    ):
        with open(path_to_formulas) as f:
            d = json.load(f)
        self.formulas = d["formulas"]
        self.inputs = inputs

    @property
    def MatminerSimilarityObject(self):
        inputs = self.inputs
        obj = MatminerSimilarity(
            path_to_imputer=inputs["path_to_imputer"],
            path_to_scaler=inputs["path_to_scaler"],
            data_source=inputs["data_source"],
        )
        return obj

    @property
    def X(self):
        # THIS TAKES SEVERAL MINUTES ON MY NICE MAC
        obj = self.MatminerSimilarityObject
        formulas = self.formulas
        pool = mp.Pool(processes=mp.cpu_count() - 1)
        return [
            r
            for r in pool.starmap(
                _get_feature_vector, [(obj, formula) for formula in formulas]
            )
        ]

    @property
    def fit_imputer_and_scaler(self):
        X = self.X
        imp = SimpleImputer()
        imp.fit(X)
        X = imp.transform(X)
        sc = StandardScaler()
        sc.fit(X)

        fimp = self.inputs["path_to_imputer"]
        fsc = self.inputs["path_to_scaler"]

        joblib.dump(imp, fimp)
        joblib.dump(sc, fsc)
        print("done")


def MatMiner_features_for_formulas(
    formulas,
    path_to_imputer,
    path_to_scaler,
):
    all_features = []
    obj = MatminerSimilarity(
        path_to_imputer=path_to_imputer,
        path_to_scaler=path_to_scaler,
    )
    for x in formulas:
        all_features.append(
            obj.feature_vector_normalized(x),
        )
    return all_features

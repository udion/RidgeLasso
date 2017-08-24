#!/bin/python3

import numpy as np
import pandas as pd
import pickle

"""
NOTE: All functions mentioned below MUST be implemented
      All functions must be reproducible, i.e., repeated function calls with the
      same parameters must result in the same output. Look into numpy RandomState
      to achieve this.
"""

def get_feature_matrix(file_path):
    """
    reads the csv from the file_path and returns the numpy array of data
    """
    df = pd.read_csv(file_path, header=0)
    res = df.values
    return res



def get_output(file_path):
    """
    file_path: path to a file in the same format as in the Kaggle competition

    Return: an n x 1 numpy array where n is the number of examples in the file.
            The array must contain the Output column values of the file
    """
	#assuming the last column is the output column
    df = pd.read_csv(file_path,header=0)
    res = np.array(df.iloc[:,len(df.columns)-1])
    return res


def get_weight_vector(feature_matrix, output, lambda_reg, p):
    """
    feature_matrix: an n x m 2-D numpy array where n is the number of samples
                    and m the feature size.
    output: an n x 1 numpy array reprsenting the outputs for the n samples
    lambda_reg: regularization parameter
    p: p-norm for the regularized regression

    Return: an m x 1 numpy array weight vector obtained through stochastic gradient descent
            using the provided function parameters such that the matrix product
            of the feature_matrix matrix with this vector will give you the
            n x 1 regression outputs
    """


def get_my_best_weight_vector():
    """
    Return: your best m x 1 numpy array weight vector used to predict the output for the
            kaggle competition.

            The matrix product of the feature_matrix, obtained from get_feature_matrix()
            call with file as test_features.csv, with this weight vector should
            result in you best prediction for the test dataset.

    NOTE: For your final submission you are expected to provide an output.csv containing
          your predictions for the Kaggle test set and the weight vector returned here
          must help us to EXACTLY reproduce that output.csv

          We will also be using this weight to evaluate on a separate hold out test set

          We expect this function to return fast. So you are encouraged to return a pickeled
          file after all your experiments with various values of p and lambda_reg.
    """

#!/bin/python3

import numpy as np
import pickle
import pandas as pd

def get_feature_matrix(file_path):
	df = pd.read_csv(file_path)
	n_data = len(df.index)
	n_feature = len(df.columns)

	res = np.zeros((n_data, n_feature))

	for i in range(0,n_feature):
		res[:,i] = df[i]
	res


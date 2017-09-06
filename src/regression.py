#!/bin/python3

import numpy as np
import pandas as pd
import pickle


def get_feature_matrix(file_path):
	df = pd.read_csv(file_path, header=0)
	#the total of 24 valueable features
	#the first 1 is not helpful 
	#however and datetime data can be used, <----tried various experiment, doesn't really help
	c = len(df.iloc[0,:])
	df_ = df.iloc[:,2:c]

	if len(df_.iloc[0,:])==25:
		print('loading training data...')
		res = df_.iloc[:,0:-1].values
		res = engineerdFeatures(res)
		res = normalizeMydata(res, np.mean(res, axis=0), np.std(res, axis=0))
	else:
		print('loading testing data...')
		res = df_.values
		res = engineerdFeatures(res)
		res = normalizeMydata(res, trainMean, trainStd)
		res = np.append(res, np.ones([len(res[:,0]),1]), axis=1)    
	return res


#assuming the last column is the output column
def get_output(file_path):
	df = pd.read_csv(file_path,header=0)
	res = np.array(df.iloc[:,len(df.columns)-1])
	return res


def get_weight_vector(feature_matrix, output, lambda_reg, p):
	X = feature_matrix #the date has been taken care of
	Y = output
	alpha = 0.01
	result = subgrad(X, Y, lambda_reg, p, alpha)
	return result[0] #weights


def get_my_best_weight_vector():
	fObj = open('../results/simpleLinear_cubedFeatures_noYClipping_final','rb')
	b = pickle.load(fObj)
	return b[2]





################## Some of my own functions ############################

def get_cost(W,X,Y):
	n = len(X[0,:])
	m = len(Y)
	cost = 0
	for i in range(0,m):
	    cost += (sum(W*X[i,:]) - Y[i])**2
	cost = np.sqrt(cost/(m))
	return cost


def subgrad(X,Y,lam,p,alpha):
	#iterable range
	maxX = np.max(X, axis=0)
	n_features = len(maxX) #number of features
	n_samples = len(Y) #number of samples

	#initialisation of weights to perform grad descent
	W = 0.01*np.ones(n_features+1)
	W_ = np.zeros(n_features+1)
	#vars for descent
	eps = 0.0001
	#stepLimit = 500
	costs = []
	steps = 0
	norm_grad = 100
	print('size of X[0,:]={0} and size of W={1}'.format(len(X[0,:]), len(W)))

	#the optimization appending ones in X_ will help simplify calc
	X_1 = np.append(X, np.ones([len(X[:,0]),1]), axis=1)
 
	while(norm_grad > eps):
		steps += 1
		grad = np.zeros(n_features+1)

		#cooler way to do things
		hypo = np.dot(X_1,W)
		loss = hypo - Y
		cost = get_cost(W, X_1,Y)
		grad = np.dot(X_1.transpose(),loss)/n_samples
		#to make it general
		for i in range(0,n_features):
			if(abs(W[i]) >0):
				grad[i] += lam*p*(abs(W[i])**p)/(W[i])

		W_ = W - alpha*grad

		#switching old and new W
		temp = W
		W = W_
		W_ = temp

		costs.append(get_cost(W,X_1,Y))
		norm_grad = np.linalg.norm(grad)

		if(steps > 3):
			alpha= updateStep(alpha,costs[-1], costs[-2])
		# 	if(alpha_new < alpha):
		# 		alpha = alpha_new
		# 		W = W_
		# 		print('retrying...')
		# 	else:
		# 		alpha = alpha_new

		print('step={0}, cost={1}, norm_grad={2}'.format(steps, costs[-1], norm_grad))
		#if(steps%100 == 0):
		#	print('step={0}, cost={1}, norm_grad={2}'.format(steps, costs[-1], norm_grad))
		#if(steps > stepLimit):
			#break
		#now the loop finshes
		#print('step={0}, cost={1}, norm_grad={2}'.format(steps, costs[-1], norm_grad))
	return(W, costs, steps)


def engineerdFeatures(mat):
	M = mat
	M = np.append(mat, np.power(mat,2), axis=1)
	M = np.append(M, np.power(mat,3), axis=1)
	return M


#for updating the stepsize
def updateStep(alpha, new_cost, prev_cost):
	if(new_cost - prev_cost < 0):
		alpha_ = alpha+0.05*alpha
	else:
		print('overshot the optimum point, readjusting...')
		alpha_ = alpha*(1/5)
		print(alpha_, alpha)
	return alpha_


#to avoid repetition
def storeResultsFor(file_name, lambda_reg, p, test_file_path):
	resList = []
	resList.append(lambda_reg)
	resList.append(p)
	W = get_weight_vector(inputXFeatures, outputY, lambda_reg, p)
	resList.append(W)

	#to do the prediction
	test_X = get_feature_matrix(test_file_path)

	print('loaded testing data, initiating prediction...')

	#test_X = normalizeMydata(test_data, trainMean, trainStd)
	test_X_1 = np.append(test_X, np.ones([len(test_X[:,0]),1]), axis=1)
	predicted_Y = np.dot(test_X_1, W)
	#making a csv and storing the results
	n_res = len(predicted_Y)
	index = range(1,n_res+1)
	resdf = pd.DataFrame()
	resdf['id'] = index
	resdf['output'] = predicted_Y
	resdf.to_csv('../results/'+file_name+'.csv', header = 1, index=0)

	#store
	fObj = open('../results/'+file_name, 'wb')
	pickle.dump(resList, fObj)
	fObj.close()

	print('results saved, check '+'../results/')

#function to convert the datetime data appropriately after loading
def setMydateData(data):
	npdates = [np.datetime64(x) for x in data[:,0]]
	data[:,0] = (npdates - np.min(npdates))/np.timedelta64(1,'D')
	return data

#function to normalize the data, using the training data
# def normalizeMydata(data, training_data):
# 	data = np.array(data, dtype=np.float32)
# 	training_data = np.array(training_data, dtype=np.float32)

# 	meanX = np.mean(training_data, axis=0)
# 	stdX = np.std(training_data, axis=0)

# 	data = (data-meanX)/stdX
# 	return data


def normalizeMydata(data, trainMean, trainStdd):
	data_ = np.array(data, dtype=np.float32)
	data_ = (data-trainMean)/trainStdd
	return data_



################################## The part to start the program #########################################
inputXFeatures = get_feature_matrix('../data/train.csv')
#inputXFeatures = setMydateData(inputXFeatures)
inputXFeatures = np.array(inputXFeatures, dtype = np.float32)
outputY = get_output('../data/train.csv')
outputY = np.array(outputY, dtype = np.float32)
######## some train constants derived through experiment
Xt = pd.read_csv('../data/train.csv', header = 0)
c = len(Xt.iloc[0,:])
Xt = Xt.iloc[:,2:c-1].values
Xt = engineerdFeatures(Xt)
trainMean = np.mean(Xt, axis=0)
trainStd = np.std(Xt, axis=0)

######## kindly change the myLamb for lambda and myP for p, the below function is a wrapper to
#### If you want to load the already train weights, please comment things below and write your code
#### please DO NOT TOUCH any thing above
myLamb = 0
myP = 1
storeResultsFor('simpleLinear_cubedFeatures_noYClipping_final', 0,1,'../data/test_features.csv')

##### UNCOMMENT ME BELOW ############
# prediction = np.dot(get_feature_matrix('../data/test_features.csv'), get_my_best_weight_vector())
# print(prediction)
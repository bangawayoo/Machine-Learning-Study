import numpy as np
import numpy.linalg as lin
from sklearn.preprocessing import StandardScaler
import pandas as pd 

class mypca(object):
	'''
	k : component 수
	n : 원래 차원
	components : 고유벡터 저장소 shape (k,n)
	explain_values : 고유값 shape (k,)
	'''
	k = None
	components = None
	explain_values= None

	def __init__(self, k=None, X_train=None):
		'''
		k의 값이 initial에 없으면 None으로 유지
		'''
		if k is not None :
			self.k = k
		if X_train is not None:
			self.fit(X_train)

	def fit(self,X_train=None):
		if X_train is None:
			print('Input is nothing!')
			return
		if self.k is None:
			self.k = min(X_train.shape[0],X_train.shape[1])
			
		#############################################
		# TO DO                                     #
		# 인풋 데이터의 공분산행렬을 이용해         #
		# components와 explain_values 완성          #
		#############################################
		X_train = pd.DataFrame(X_train)
		covariance_mtx = X_train.cov()
		self.components = lin.eig(covariance_mtx)[1][:,:self.k].T
		self.explain_values = lin.eig(covariance_mtx)[0][:self.k]
		#############################################
		# END CODE                                  #
		#############################################
		return

	def transform(self,X=None):
		if X is None:
			print('Input is nothing!')
			return

		result = None
		'''
		N : X의 행 수
		result의 shape : (N, k)
		'''
		#############################################
		# TO DO                                     #
		# components를 이용해 변환결과인            #
		# result 계산                               #
		#############################################
		N = X.shape[0]
		result = np.empty([N, self.k])
		for pc in range(self.k):
			result[:,pc] = self.components[pc,:].dot(X.T)
			
		#############################################
		# END CODE                                  #
		#############################################
		return result

	def fit_transform(self,X=None):
		if X is None:
			print('Input is nothing!')
			return
		self.fit(X)
		return self.transform(X)
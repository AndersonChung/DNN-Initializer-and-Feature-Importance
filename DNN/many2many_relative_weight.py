import numpy as np
from sklearn.preprocessing import StandardScaler as ss
from sklearn.cross_decomposition import CCA as ccc
import pandas as pd
import scipy as sc
from keras.models import Sequential, load_model
from keras import backend as K

def unitnorm(x,y):
	x=ss().fit_transform(x)
	y=ss().fit_transform(y)
	x=x/np.power(x.shape[0],(1/2))
	y=y/np.power(x.shape[0],(1/2))
	return x,y

def cca(x,y):
	if x.T.dot(x).size==1 and y.T.dot(y).size==1:
		ccc=x.T.dot(y)
		ur=x
		vr=y
	elif x.T.dot(x).size==1 and y.T.dot(y).size!=1:
		beta=y.T.dot(x)
		ccc=np.power(np.sum(np.power(beta,2),0),1/2)
		ur=x
		vr=y.dot(beta)
	elif x.T.dot(x).size!=1 and y.T.dot(y).size==1:
		beta=x.T.dot(y)
		ccc=np.power(np.sum(np.power(beta,2),0),1/2)
		ur=x.dot(beta)
		vr=y
	elif x.T.dot(x).size!=1 and y.T.dot(y).size!=1:
		svdxu,svdxs,svdxv=np.linalg.svd(x.T.dot(x),full_matrices=False)
		svdyu,svdys,svdyv=np.linalg.svd(y.T.dot(y),full_matrices=False)
		sxx=svdxu.dot(np.dot(np.diag(np.power(svdxs,-1/2)),svdxv))
		syy=svdyu.dot(np.dot(np.diag(np.power(svdys,-1/2)),svdyv))
		br,ccc,art=np.linalg.svd(syy.dot(np.dot(y.T.dot(x),sxx)),full_matrices=False)
		ur=x.dot(np.dot(sxx,art.T))
		vr=y.dot(np.dot(syy,br))
	return ccc,ur,vr

def rwn(x,y):
	svdxu,svdxs,svdxv=np.linalg.svd(x,full_matrices=False)
	svdyu,svdys,svdyv=np.linalg.svd(y,full_matrices=False)
	zx=svdxu.dot(svdxv)
	zy=svdyu.dot(svdyv)
	ccc,ur,vr=cca(zx,zy)
	rwx=np.dot(np.power(zx.T.dot(x),2).T,np.power(zx.T.dot(vr),2))
	rwy=np.dot(np.power(zy.T.dot(y),2).T,np.power(zy.T.dot(vr),2))
	return ccc,rwx,rwy




def many2manyrw (x,y):
	if x.shape[0]!=y.shape[0] or type(x) is not np.ndarray or type(y) is not np.ndarray:
		return 0 #only for numpy matrix using
	x,y=unitnorm(x,y)
	ccc,rwx,rwy=rwn(x,y)
	return ccc,rwx,rwy



if __name__=='__main__':
	train_x=np.load("data/train_x.npy")
	train_y=np.load("data/train_y.npy")
	model_acc=load_model("model/model_acc1.h5")
	get_input = K.function([model_acc.layers[0].input],[model_acc.layers[2].input])
	get_output = K.function([model_acc.layers[0].input],[model_acc.layers[2].output])
	inputs=get_input([train_x])[0]
	outputs=get_output([train_x])[0]
	inputs=inputs.astype('float64')
	outputs=outputs.astype('float64')
	#print(np.linalg.matrix_rank(inputs),np.linalg.matrix_rank(outputs))
	ccc,rwx,rwy=many2manyrw(inputs,outputs)
	print(ccc)	


import numpy as np
from sklearn.preprocessing import StandardScaler as ss
from sklearn.cross_decomposition import CCA as ccc
import pandas as pd
import scipy as sc
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten,BatchNormalization
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
from keras.constraints import max_norm
from keras.callbacks import Callback
from keras.models import Sequential, load_model
import random
from keras import backend as K
from many2many_relative_weight import many2manyrw

test_x=np.load("data/test_x.npy")
train_x=np.load("data/train_x.npy")
test_y=np.load("data/test_y.npy")
train_y=np.load("data/train_y.npy")
model_acc=load_model("model/model_acc1.h5")


def nnrw(x,y,model,actv=True):
	rw=[]
	if actv:
		num=int(len(model.layers)/2)
	else:
		num=int((len(model.layers)+1)/2)
	for i in range(num+1):
		if i==num:
			get_input = K.function([model.layers[0].input],[model.layers[-1].output])
			inputs=get_input([x])[0]
			outputs=np.reshape(y,(len(y),1))
			inputs=inputs.astype('float64')
			outputs=outputs.astype('float64')
			ccc,rwx,rwy=many2manyrw(inputs,outputs)
			rw.append(rwx)
		else:
			get_input = K.function([model.layers[0].input],[model.layers[2*i].input])
			get_output = K.function([model.layers[0].input],[model.layers[2*i].output])
			inputs=get_input([x])[0]
			outputs=get_output([x])[0]
			inputs=inputs.astype('float64')
			outputs=outputs.astype('float64')
			ccc,rwx,rwy=many2manyrw(inputs,outputs)
			rw.append(np.dot(rwx,rwy.T))
	for i in range(len(rw)-1):
		if i==0:
			rwv=np.dot(rw[0],rw[1])
		else:
			rwv=np.dot(rwv,rw[i+1])
	return rwv,rw[-1]


for i in range(5):
	path="model3/model_loss1-"+str(i)+".h5"
	model_loss=load_model(path)
	a,b=nnrw(train_x,train_y,model_loss,actv=False)
	print(i)
	if i==0:
		out=a
	else:
		out=np.concatenate((out,a),axis=1)
np.savetxt("mse2.csv", out, delimiter=",")
	





def nnmw(x,y,model,actv=True):
	mw=[]
	if actv:
		num=int(len(model.layers)/2)
	else:
		num=int((len(model.layers)+1)/2)
	for i in range(num):
		mw.append(np.absolute(model.layers[2*i].get_weights()[0]))
	for i in reversed(range(len(mw)-1)):
		if i==(len(mw)-2):
			mwv=np.dot(mw[i],mw[i+1])
		elif i==0:
			for j in range(len(mwv)):
				mw[i][:,j]=mw[i][:,j]*mwv[j]
			mwv=mw[i]
		else:
			mwv=np.dot(mw[i],mwv)
	for i in range(mwv.shape[1]):
		mwv[:,i]=mwv[:,i]/sum(mwv[:,i])
	return mwv


#train_y=preprocessing.scale(train_y)
#for i in range(5):
#	path="model3/model_loss1-"+str(i)+".h5"
#	model_loss=load_model(path)
#	a=np.sum(nnmw(train_x,train_y,model_loss,actv=False),1)
#	a=np.reshape(a,(len(a),1))
#	print(i)
#	if i==0:
#		out=a
#	else:
#		out=np.concatenate((out,a),axis=1)
#np.savetxt("mse2_mw.csv", out, delimiter=",")





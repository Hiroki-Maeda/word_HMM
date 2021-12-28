#all_filmed_data : not hmm train data



import numpy as np
import pandas
import os, sys
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
from scipy import signal
import collections
import random
from sklearn.model_selection import train_test_split

#filtaring method
def lowpass(x, samplerate, fp, fs, gpass, gstop):
	fn = samplerate/2
	wp = fp/fn
	ws = fs/fn
	N,Wn = signal.buttord(wp,ws,gpass,gstop)
	b,a = signal.butter(N,Wn,"low")
	y = signal.filtfilt(b,a,x)
	return y


#read data method
def read_data(Data_Path,words,ch_names,one_data_len,one_state_num,b_num,samplerate=1000,fp=30,fs=50,gpass=3,gstop=40):	
	all_filted_data = []
	all_label = np.empty(0,dtype=int)
	word_num = len(words)
	print("read end")
	for j in range(len(words)):
		one_word_data = []
		for k in range(len(ch_names)):
	
			data = pd.read_csv(os.path.join(Data_Path,"data_"  + words[j] +ch_names[k] +".csv")).values

			filted_data = np.empty((0,data.shape[1]-100),float)
			
			for i in range(data.shape[0]):
				filted_data = np.vstack([filted_data,lowpass(data[i,:],samplerate,fp,fs,gpass,gstop)[50:-50]])	
				

			data = filted_data	
				
			if k == 0:
				one_word_data = np.empty((data.shape[0],b_num,data.shape[1]))
		
			one_word_data[:,k,:] =data 

			if j == 0 and k == 0:
				#(data_num, data_length) = filted_data.shape	
				(data_num, data_length) = data.shape	
				all_filted_data = np.empty((0,b_num,data_length))
				
		all_filted_data = np.concatenate([all_filted_data,one_word_data],0)
		all_label = np.concatenate([all_label,np.full(one_word_data.shape[0],j,dtype=int)],0)
		
	print("read end")	

	return all_filted_data,all_label

def normalize_func(all_data):
	return  (all_data-all_data.mean(axis=1).reshape(-1,1))/all_data.std(axis=1).reshape(-1,1)

def reject_artifact(all_data,all_label,threshold):
	(data_num,b_num,data_length) = all_data.shape
	applied_data = np.empty((0,b_num,data_length))
	applied_label = np.empty(0)
	for i in range(data_num):
		for j in range(b_num):
			if max(abs(all_data[i,j,:])) >threshold:
				break
		
			elif j == b_num-1:

				temp = normalize_func(all_data[i])
				applied_data = np.append(applied_data,np.array([temp]),axis=0)
				applied_label = np.append(applied_label,all_label[i])

	return applied_data,applied_label	

#config
#Data_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","1_word_HMM","data","vec_data")
Data_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","word_HMM","data","div_data")
Save_Path =  os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","word_HMM","picture")


color = ["#000000","#44ffff","#88ffff","#bbffff","#eeffff","#ff44ff","#ff88ff","#ffbbff","#ffeeff","#ffff44","#ffff88","#ffffbb","#ffffee","#444444","#888888","#bbbbbb","#eeeeee","#44ff44","#88ff88","#bbffbb","#eeffee"]
#words = ["a","i","u","e","o"]
words = ["0","1","2","3","4"]

ch_names = [" 2-REF"," 6-REF"," 4-REF"]
#ch_names = ["_F7-T7","_T7-Cz","_Cz-F7"]
fig_flag = 0
b_num = len(ch_names)
word_num = len(words)
all_data = np.empty((b_num,0))
one_data_len= 924
one_state_num = 4
one_state_len = round(one_data_len/one_state_num)
state_num = 1+len(words)*one_state_num
data_div_state =[[[[] for i in range(one_state_num)] for j in range(b_num)] for k in range(word_num)]
data_div_state_0 = [[]for i in range(b_num)]
sum_data = np.empty((b_num,0))
set_init = True
hmm_trainnum_rate = 0.3
#classification_modelnum_rate = 0.4
test_num_rate = 0.7
Restrict = 1


#read data
all_data , all_label= read_data(Data_Path,words,ch_names,one_data_len,one_state_num,b_num,samplerate=500,fp=30,fs=50,gpass=3,gstop=40)
(data_num,_,data_length) = all_data.shape
print(all_data.shape)
print(all_label.shape)
print(np.sum(all_label == 0))
print(np.sum(all_label == 1))
print(np.sum(all_label == 2))
print(np.sum(all_label == 3))
print(np.sum(all_label == 4))
rj_applied_data , rj_applied_label = reject_artifact(all_data,all_label,100)
print(rj_applied_data.shape)
print(rj_applied_label.shape)
print(np.sum(rj_applied_label == 0))
print(np.sum(rj_applied_label == 1))
print(np.sum(rj_applied_label == 2))
print(np.sum(rj_applied_label == 3))
print(np.sum(rj_applied_label == 4))
(data_num,_,data_length) = rj_applied_data.shape

#divide into train data or test data
train_num = round(data_num*hmm_trainnum_rate)
test_num = round(data_num*test_num_rate)
print(test_num)
print(data_num)

train_shuffle = random.sample(range(data_num),data_num) 
"""
train_data = np.zeros([word_num*b_num,train_num,data_length])
test_data = np.zeros([word_num*b_num,test_num,data_length])

train_data = all_filted_data[:,train_shuffle[0:train_num],:]
test_data = all_filted_data[:,train_shuffle[train_num:data_num],:]
#hmm_model = hmm.GaussianHMM(n_components=state_num,covariance_type="full",init_params="")

"""
data_number = np.linspace(0,data_num-1,data_num,dtype=int)
x_train , x_test, y_train, y_test = train_test_split(rj_applied_data,rj_applied_label,test_size=1-hmm_trainnum_rate,stratify=rj_applied_label)

y_test = y_test.astype(np.int64)
y_train = y_train.astype(np.int64)

#train_data = np.zeros((len(x_train_num),b_num,data_length))

#sys.exit()

#trian_data = np.zeros([train_num,b_num,data_length])

if set_init == True:
	model = [hmm.GaussianHMM(n_components=one_state_num,covariance_type="full",init_params="") for i in range(word_num)]
else:
	model = [hmm.GaussianHMM(n_components=one_state_num,covariance_type="full") for i in range(word_num)]

#init init_prob
init_prob = []
temp = []
for i in range(one_state_num):
	temp.extend([np.random.rand()])

temp[0] += 1	
init_prob.extend((temp/np.sum(temp)).tolist())


ans = []
print("train start")
for i in range(word_num):
	train_data = x_train[y_train==i,:]
	print("train_data_num : {}".format(train_data.shape[0]))
	if set_init == True:
		#init a : transfer mat
		a = []

		for s_j in range(one_state_num):
			temp = []
			for s_i in range(one_state_num):	
				temp.append(np.random.rand())	
				
			a.append(temp)

		a = np.array(a)

		a[0,:] = 1.0/state_num	
		for j in range(one_state_num-1):
			a[j+1,j+1] += 1

			
		for s_j in range(one_state_num):
			temp = 0
			for s_i in range(one_state_num):
				temp += a[s_j,s_i]	
			a[s_j,:] = a[s_j,:]/temp 

		a = a.tolist()


		#init mu

		train_mu = np.zeros((one_state_num,b_num))
		for s_i in range(one_state_num):
			for s_j in range(b_num):
				train_mu[s_i,s_j] = train_data[:,s_j,s_i*one_state_len:(1+s_i)*one_state_len].mean()
		#init cov
		train_cov = np.tile(np.identity(b_num),(one_state_num,1,1))
		for s_i in range(one_state_num):
			train_cov[s_i] = np.cov(train_data[:,0:b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].reshape((2,-1)),train_data[:,b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].reshape((1,-1)),rowvar=1)
		
		#init params
		model[i].startprob_ = init_prob
		model[i].transmat_ = a
		model[i].means_ = train_mu
		model[i].covars_ = train_cov

	temp = np.zeros((0,data_length))
	temp = np.sum(train_data,axis=0)/train_data.shape[0]
		
	model[i].fit(temp.T)
print("train end")
print("classification start")
prob_ans = np.zeros(word_num)
ans_count = 0
confusion_matrix = np.zeros((word_num,word_num))
print("test_num:{}".format(test_num))
print("test_num(len):{}".format(x_test.shape[0]))

#for i in range(word_num):	
#	print("classification : ",i)
for j in range(test_num):
	temp = np.zeros((0,data_length))
	for k in range(word_num):
		prob_ans[k] = model[k].score(x_test[j,:,:].T)
	if j==0:
		print(prob_ans)
	ans.append(np.argmax(prob_ans))		
	confusion_matrix[y_test[j],ans[-1]] += 1
	if ans[-1]==y_test[j]:
		ans_count +=1
print("classification end")
print("ans : ",ans)
print("ans rate  : ",ans_count/(test_num))
print("confusion matrix")
print(confusion_matrix)

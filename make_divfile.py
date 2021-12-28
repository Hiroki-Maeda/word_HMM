import numpy as np
import pandas as pd
import os
import sys
#DataPath = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","data_tukiyama","SSBCI","SSBCI","recorded_EEGs","maeda","CSV")

#data = pd.read_csv(os.path.join(DataPath,"data_1","overt_1.CSV"))

#print(data.head())

#StimulusPath = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","data_tukiyama","SSBCI","SSBCI","stimulus")
StimulusPath = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","data_word","stimuli_1207_csv")
#Save_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","covert","div_data")
Save_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","word_HMM","data","div_data")

#stimulusData = pd.read_excel(os.path.join(StimulusPath,"stimulus_1.xlsx"))

#print(stimulusData.head())
#print(stimulusData["covert"])
#data = data.iloc[:,1:]

count = 0
flag = 0
data_size = 0
start = 0
finish = 0
segment_size = 1024 #500hz, 2s
Dataset_num = 60
sub_names = ["morita","yanagibashi","uemura"]
#ch_names = [" F7-REF"," T7-REF"," Cz-REF"]


#ch_names = [" F7-REF"," T7-REF"," P7-REF"," O1-REF"," O2-REF"," P8-REF"," T8-REF"," F8-REF"," Fp2-REF"," Fp1-REF"," F3-REF"," C5-REF"," P3-REF"," P4-REF"," C6-REF"," F4-REF"," Fz-REF"," Cz-REF"," Pz-REF"]
ch_names = [" 1-REF"," 2-REF"," 3-REF"," 4-REF"," 5-REF"," 6-REF"]
read_cols = [" 1-REF"," 2-REF"," 3-REF"," 4-REF"," 5-REF"," 6-REF"," T1"]


for ch_name in ch_names:
	 
	data_0 = np.empty((0,segment_size),float)
	data_1 = np.empty((0,segment_size),float)
	data_2 = np.empty((0,segment_size),float) 
	data_3 = np.empty((0,segment_size),float)
	data_4 = np.empty((0,segment_size),float)

	for set_num in range(Dataset_num):
		stimulusData = pd.read_csv(os.path.join(StimulusPath,"stimuli_"+str(set_num)+".csv"),header=None).values[0]
		for sub_name in sub_names:
			#print("sub_name : {}".format(sub_name))
			#DataPath = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","data_tukiyama","SSBCI","SSBCI","recorded_EEGs",sub_name,"CSV")
			DataPath = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","data_word",sub_name)

		 
			data = pd.read_csv(os.path.join(DataPath,"covert_"+str(set_num)+".CSV"),header=3,usecols=read_cols)
		
			div_data = np.empty((0,segment_size),float)
			finish = 0
			for i in range(len(data)-1):
				if ( i-finish)>500:
				
					if(data.loc[i," T1"]>1000 and flag ==0 ):
						
						flag=1
						count+=1  
						start=i

					elif(flag==1 and data.loc[i," T1"]<1000):
						flag=0
						finish = i
					
						div_data= np.vstack([div_data ,data.loc[start:start+segment_size-1,ch_name].values])

			if not (div_data.shape[0] == 10):
				print(len(data))
				print(div_data.shape)
				print("data cut error",file=sys.stderr)
				sys.exit()	
			print(count)

			stimulus_num = stimulusData

			for i in range(len(stimulusData)):
				if(stimulus_num[i]==1):
					data_0 = np.vstack([data_0,div_data[i]]) 
				elif(stimulus_num[i]==2):
					data_1 = np.vstack([data_1,div_data[i]]) 
				elif(stimulus_num[i]==3):
					data_2 = np.vstack([data_2,div_data[i]]) 
				elif(stimulus_num[i]==4):
					data_3 = np.vstack([data_3,div_data[i]]) 
				elif(stimulus_num[i]==5):
					data_4 = np.vstack([data_4,div_data[i]]) 

	print(data_0.shape)
	print(data_1.shape)
	print(data_2.shape)
	print(data_3.shape)
	print(data_4.shape)
	np.savetxt(os.path.join(Save_Path,"data_0"+str(ch_name)+".csv"),data_0,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_1"+str(ch_name)+".csv"),data_1,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_2"+str(ch_name)+".csv"),data_2,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_3"+str(ch_name)+".csv"),data_3,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_4"+str(ch_name)+".csv"),data_4,delimiter=',')


print(data_0)

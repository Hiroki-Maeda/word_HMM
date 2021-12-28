import numpy as np
import pandas as pd
import os

Data_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","word_HMM","data","div_data")
Save_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","word_HMM","data","vec_data")
words = ["0","1","2","3","4"]
ch_names = [" F7-REF"," T7-REF"," Cz-REF"]
vec_names = ["F7-T7","T7-Cz","Cz-F7"]
ch_names = [" 2-REF"," 6-REF"," 4-REF"]
vec_names = ["2-6","6-4","4-2"]
for word in words:
	for i in range(len(ch_names)):
		data_1 = pd.read_csv(os.path.join(Data_Path,"data_" + word + ch_names[i%len(ch_names)] + ".csv")).values
		data_2 = pd.read_csv(os.path.join(Data_Path,"data_" + word + ch_names[(i+1)%len(ch_names)] + ".csv")).values		
		vec_data = data_1-data_2

		np.savetxt(os.path.join(Save_Path,"data_"+word +"_"+ vec_names[i] + ".csv") , vec_data ,delimiter=',')

		print( "save : data_" + word + "_" + vec_names[i] + ".csv")
	



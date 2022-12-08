import numpy as np
import datetime
import pandas as pd 
import random
from Settings import*
from sklearn_extra.cluster import KMedoids
from Data_process import df_aFRR_scen,df_mFRR_scen,df_FCR_DE_scen,DA_list_scen
import matplotlib.pyplot as plt
random.seed(123)

#Input data   Set period in Settings.py
DA = DA_list_scen
aFRR_up = df_aFRR_scen['aFRR Upp Pris (EUR/MW)'].tolist()
aFRR_down = df_aFRR_scen['aFRR Ned Pris (EUR/MW)'].tolist()
mFRR = df_mFRR_scen['mFRR_UpPriceEUR'].tolist()
FCR = df_FCR_DE_scen['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 


## Find Representive weeks for a year ## 


Data = [DA,FCR,aFRR_up,aFRR_down,mFRR]
Data_names = ['DA','FCR','aFRR Up','aFRR Down','mFRR']


blocksize = 24*7
weeks = 52
#Sample length


### Multi ### 
df = pd.DataFrame({'DA':  np.array(Data[0]), 'aFRR Up:':  np.array(Data[2]), 'aFRR Down': np.array(Data[3]), 'mFRR':  np.array(Data[4])})

data = df.values.tolist() #Acces element by  data[0][0]

n = len(data)

#Split sample in blocks of length blocksize

blocks = [data[i:i + blocksize ] for i in range (0,n,blocksize)]#Acces element by blocks[0][0][0]

#Delete last block if length differs from blocksize 
if len(blocks[-1]) != blocksize:
    del blocks[-1]


## Average ## 
Avg_scenarios = np.zeros((weeks,blocksize))
len(Avg_scenarios)
for i in range(0,weeks):
    for j in range(0,blocksize):
        Avg_scenarios[i][j] = np.mean(blocks[i][j])


## Clustering ## 

n_clusters = 10 ## Rep weeks 

Red_Scen = []   

Prob = np.zeros(n_clusters) # Prob for each week   

kmedoids = KMedoids(n_clusters=n_clusters,metric='euclidean').fit(Avg_scenarios)


## Calculating scenario probability ## 

Red_Scen.append(kmedoids.cluster_centers_) 

for j in range(0,n_clusters):
    Prob[j] = np.count_nonzero(kmedoids.labels_ == j)/len(kmedoids.labels_)


## Find indexes of weeks ## 



#RedAvg_scenarios =[[19999.8, 22222. ,  2222.2,  4444.4,  2222.2,  4444.4, 11111. ,13333.2], [15555.4, 17777.6, 11111. , 13333.2, 15555.4, 17777.6,  6666.6,8888.8]]
true = 0
sample_length = len(Red_Scen[0][0])
index = []


#true = true+1

#len(blocks[0][1])

for j in range(0,len(Red_Scen[0])):   
    for x in range(0,weeks):     
        for i in range(0,sample_length):
        
            if Red_Scen[0][j][i] == np.mean(blocks[x][i]):
                true = true+1
                
                if true == sample_length: # true is a variable not bool True
                    index.append(x)
                    
            if Red_Scen[0][j][i] != np.mean(blocks[x][i]):
                true = 0
    else:
        continue


Rep_weeks = []  

for i in index:
    
    d = "2021-W" + str(i)
    r = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
    r.strftime("%m-%d-%Y %H:%M")
    Rep_weeks.append(r)


dfRepWeeks = pd.DataFrame()



dfRepWeeks['Rep Weeks for 2021'] = Rep_weeks
dfRepWeeks['π for Rep weeks 2021'] =  Prob

#save to Excel 
dfRepWeeks.to_excel("Result_files/RepWeeks2021.xlsx")

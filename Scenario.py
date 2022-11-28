import numpy as np
import pandas as pd 
import random
from Settings import*
from sklearn_extra.cluster import KMedoids
from Data_process import df_aFRR,df_mFRR,df_FCR_DE,DA_list
import matplotlib.pyplot as plt
random.seed(123)

#Input data   Set period in Settings.py
DA = DA_list
aFRR_up = df_aFRR['aFRR Upp Pris (EUR/MW)'].tolist()
aFRR_down = df_aFRR['aFRR Ned Pris (EUR/MW)'].tolist()
mFRR = df_mFRR['mFRR_UpPriceEUR'].tolist()
FCR = df_FCR_DE['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 

Data = [DA,FCR,aFRR_up,aFRR_down,mFRR]
Data_names = ['DA','FCR','aFRR Up','aFRR Down','mFRR']

## Scenario generation ## 

Type = 'single'   # 'single' or 'combined'

n_samples = 1000 #Number of samples to be made  

blocksize = 24 # 7days = 168 hours
sample_length = blocksize*3 # sampling 52 weeks blocks 

def Bootsrap(Type,Data,Data_names,n_samples,blocksize,sample_length):


    if Type == 'single':

        for x in range(0,len(Data)):
            #Sample length
            n = len(Data[x])

            #Split sample in blocks of length blocksize

            blocks = [Data[x][i:i + blocksize ] for i in range (0,n,blocksize)]

            #Delete last block if length differs from blocksize 
            if len(blocks[-1]) != blocksize:
                del blocks[-1]


            samples = np.zeros((n_samples,sample_length))

            for i in range(0,n_samples):
                t = 0
                while t < sample_length:

                    r = random.randrange(0,len(blocks))
                    #print(r)

                    for j in range(0,blocksize):
                        samples[i,t+j] = blocks[r][j]

                    t = t + blocksize
        
            if Data_names[x] == 'DA':
                DA_block = samples
            if Data_names[x] == 'FCR':
                FCR_block = samples
            if Data_names[x] == 'aFRR Up':
                aFRR_up_block = samples
            if Data_names[x] == 'aFRR Down':
                aFRR_down_block = samples
            if Data_names[x] == 'mFRR':
                mFRR_block = samples


                
    if Type == 'combined':

        ########## Multi Blocks ######## 
        ### Multi ### 
        df = pd.DataFrame({'DA':  np.array(Data[0]), 'FCR':  np.array(Data[1]) , 'aFRR Up:':  np.array(Data[2]), 'aFRR Down': np.array(Data[3]), 'mFRR':  np.array(Data[4])})

        data = df.values.tolist() #Acces element by  data[0][0]

        n = len(data)

        #Split sample in blocks of length blocksize

        blocks = [data[i:i + blocksize ] for i in range (0,n,blocksize)]#Acces element by blocks[0][0][0]

        #Delete last block if length differs from blocksize 
        if len(blocks[-1]) != blocksize:
            del blocks[-1]

        len_element = len(blocks[1][1])

        ## creating an array with zeros and same dimensions as blocks 
        samples = np.zeros((n_samples,sample_length,len_element))

        for i in range(0,n_samples):
            t = 0
            while t < sample_length:

                r = random.randrange(0,len(blocks))
                print(r)

                for j in range(0,blocksize):
                
                    samples[i,t+j] = blocks[r][j]

                t = t + blocksize

        Combined_blocks = samples

        
    if Type == 'single':
        return  DA_block,FCR_block,aFRR_up_block,aFRR_down_block,mFRR_block
    if Type == 'combined': 
        return Combined_blocks

scenarios = Bootsrap(Type,Data,Data_names,n_samples,blocksize,sample_length)

## Scenario reduction ## 
#Specify number of clusters
n_clusters = 5

def K_Medoids(scenarios,n_clusters):

    Red_Scen = []   ## Red_Scen[0] = DA scenarios, Red_Scen[1] = FCR scenarios, Red_Scen[2] = aFRR_up scenarios, Red_Scen[3] = aFRR_Down scenarios, Red_Scen[4] = mFRR scenarios 

    Prob = np.zeros((n_clusters,len(scenarios))) # Prob scenario 1 in DA = Prob[0,0], Prob scenario 2 in DA = Prob[1,0] osv... Prob scenario 1 FCR = Prob[0,1] .....   



    for i in range(0,len(scenarios)):

        kmedoids = KMedoids(n_clusters=n_clusters,metric='euclidean').fit(scenarios[i])


        ## Calculating scenario probability ## 

        Red_Scen.append(kmedoids.cluster_centers_) 

        for j in range(0,n_clusters):
            Prob[j,i] = np.count_nonzero(kmedoids.labels_ == j)/len(kmedoids.labels_)


        

    return Red_Scen,Prob 

Rep_scen,Prob = K_Medoids(scenarios,n_clusters)     

# Rep_scen[0] = DA scenarios, Rep_scen[1] = FCR scenarios, Rep_scen[2] = aFRR_up scenarios, Rep_scen[3] = aFRR_Down scenarios, Rep_scen[4] = mFRR scenarios 
Rep_scen[1][0][0]   #Market , Omega, time 

Prob # Prob scenario 1 in DA = Prob[0,0], Prob scenario 2 in DA = Prob[1,0] osv... Prob scenario 1 FCR = Prob[0,1] ..... 























""" 



from statsmodels.tsa.stattools import adfuller


def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")


check_stationarity(df_data['FCR'])


import statsmodels.api as sm

start_date = '2020-01-01 00:00'
end_date = '2021-12-31 23:59'


df_data[(start_date <= df_data.index) & (df_data.index <= end_date)].plot(grid='on')
plt.show()


decomposition = sm.tsa.seasonal_decompose(df_data['DA'], model = 'additive')
seasonal = decomposition.seasonal
trend = decomposition.trend
trend.plot()
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
plt.show()




trend = decomposition.trend
#seasonal = decompose_result_mult.seasonal
#residual = decompose_result_mult.resid





 """
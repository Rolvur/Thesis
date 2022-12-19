import numpy as np
import datetime
import pandas as pd 
import random
from Settings import*
from sklearn_extra.cluster import KMedoids
from Data_process import PV_scenPower,df_aFRR_scen,df_mFRR_scen,df_FCR_DE_scen,DA_list_scen
import matplotlib.pyplot as plt
random.seed(123)

#Input data   Set period in Settings.py
DA = DA_list_scen
aFRR_up = df_aFRR_scen['aFRR Upp Pris (EUR/MW)'].tolist()
aFRR_down = df_aFRR_scen['aFRR Ned Pris (EUR/MW)'].tolist()
mFRR = df_mFRR_scen['mFRR_UpPriceEUR'].tolist()
FCR = df_FCR_DE_scen['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 

Data = [DA,FCR,aFRR_up,aFRR_down,mFRR]
Data_names = ['DA','FCR','aFRR Up','aFRR Down','mFRR']

Data_comb = [DA,aFRR_up,aFRR_down,mFRR]
Data_comb_names = ['DA','aFRR Up','aFRR Down','mFRR']

PV = PV_scenPower


#Defining functions
def Bootsrap(Type,Data,Data_names,n_samples,blocksize,sample_length):


    if Type == 'single':
        DA_block = []
        FCR_block = []
        aFRR_up_block = []
        aFRR_down_block = []
        mFRR_block = []


        for x in range(0,len(Data)):
            #Sample length
            n = len(Data[x])

            #Split sample in blocks of length blocksize

            blocks = [Data[x][i:i + blocksize] for i in range (0,n,blocksize)]

            #Delete last block if length differs from blocksize 
            if len(blocks[-1]) != blocksize:
                del blocks[-1]


            samples = np.zeros((n_samples,sample_length))

            for i in range(0,n_samples):
                t = 0
                while t < sample_length:

                    r = random.randrange(0,len(blocks))
                   

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
        df = pd.DataFrame({'DA':  np.array(Data[0]) , 'aFRR Up:':  np.array(Data[1]), 'aFRR Down': np.array(Data[2]), 'mFRR':  np.array(Data[3])})

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
                

                for j in range(0,blocksize):
                
                    samples[i,t+j] = blocks[r][j]

                t = t + blocksize

        Combined_blocks = samples

        
    if Type == 'single':
        
            return  DA_block,FCR_block,aFRR_up_block,aFRR_down_block,mFRR_block

    if Type == 'combined': 
        return Combined_blocks

def GenAverage(scenarios,n_samples,sample_length):
    Avg_scenarios = np.zeros((n_samples,sample_length))

    for i in range(0,n_samples):
        for j in range(0,sample_length):
            Avg_scenarios[i][j] = scenarios[i][j].mean()
    return Avg_scenarios

def AvgKmedReduction(Avg_scenarios,scenarios,n_clusters,n_samples,sample_length):

    Red_Scen = []   ## Red_Scen[0] = DA scenarios, Red_Scen[1] = FCR scenarios, Red_Scen[2] = aFRR_up scenarios, Red_Scen[3] = aFRR_Down scenarios, Red_Scen[4] = mFRR scenarios 

    Prob = np.zeros(n_clusters) # Prob scenario 1 in DA = Prob[0,0], Prob scenario 2 in DA = Prob[1,0] osv... Prob scenario 1 FCR = Prob[0,1] .....   

    kmedoids = KMedoids(n_clusters=n_clusters,metric='euclidean').fit(Avg_scenarios)


    ## Calculating scenario probability ## 

    Red_Scen.append(kmedoids.cluster_centers_) 

    for j in range(0,n_clusters):
        Prob[j] = np.count_nonzero(kmedoids.labels_ == j)/len(kmedoids.labels_)


            


    #Rep_scen,Prob = K_Medoids(scenarios,n_clusters)     


    


    #RedAvg_scenarios =[[19999.8, 22222. ,  2222.2,  4444.4,  2222.2,  4444.4, 11111. ,13333.2], [15555.4, 17777.6, 11111. , 13333.2, 15555.4, 17777.6,  6666.6,8888.8]]
    true = 0
    index = []


    #RedAvg_scenarios[0][0] == scenarios[0][0].mean()
    #true = true+1

    for j in range(0,len(Red_Scen[0])):   
        for x in range(0,n_samples):     
            for i in range(0,sample_length):
            
                if Red_Scen[0][j][i] == scenarios[x][i].mean():
                    true = true+1
                    
                    if true == sample_length:
                        index.append(x)
                        
                if Red_Scen[0][j][i] != scenarios[x][i].mean():
                    true = 0
        else:
            continue

        
    rep_senc1 = []

    index
    for i in index:
        rep_senc1.append(scenarios[i]) 

    # Rep_scen[0] = DA scenarios, Rep_scen[1] = FCR scenarios, Rep_scen[2] = aFRR_up scenarios, Rep_scen[3] = aFRR_Down scenarios, Rep_scen[4] = mFRR scenarios 
    #Rep_scen[1][0][0]   #Market , Omega, time 

    #rep_senc1[1][0].mean()

    #rep_senc1[1][:,0]

    #len(rep_senc1[0][0])

    Rep_scen1 = np.zeros((len(rep_senc1[0][0]),n_clusters,sample_length))




    for i in range(0,len(rep_senc1[0][0])): # nr markets
        for j in range(0,n_clusters):
            Rep_scen1[i][j] = rep_senc1[j][:,i]

    
            
            
            
    return Rep_scen1, Prob

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

def SingleInputData(Rep_scen,Prob):

    x = len(Rep_scen[0])
    hours = len(Rep_scen[0][0])
    Ω = x**4
    Φ = x
    c_FCRs = {}
    c_aFRR_ups = {}
    c_aFRR_downs = {}
    c_mFRR_ups = {}
    π_r = {}

    for a in range(1,x+1):
        for b in range(1,x+1):
            for c in range(1,x+1):
                for d in range(1,x+1):
                
                    w = (a-1)*x**3 + (b-1)*x**2 + (c-1)*x + d
                    π_r[w] = Prob[a-1,1] * Prob[b-1,2] * Prob[c-1,3] * Prob[d-1,4] 
                    
                    for t in range(1,hours+1):

                        c_FCRs[(w,t)] = Rep_scen[1][a-1][t-1]
                        c_aFRR_ups[(w,t)] = Rep_scen[2][b-1][t-1]
                        c_aFRR_downs[(w,t)] = Rep_scen[3][c-1][t-1]
                        c_mFRR_ups[(w,t)] = Rep_scen[4][d-1][t-1]
    

    c_DAs = {}
    π_DA = {}
    for i in range(1,x+1):
        π_DA[(i)] = Prob[i-1,0] 
        for t in range(1,hours+1):
            c_DAs[(i,t)] = Rep_scen[0][i-1][t-1]





    return Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA

def CombInputData(Rep_scen_comb,Prob_comb):

    x = len(Rep_scen_comb[0])
    hours = len(Rep_scen_comb[0][0])
    Ω = x**4
    Φ = x
    c_FCRs = {}
    c_aFRR_ups = {}
    c_aFRR_downs = {}
    c_mFRR_ups = {}
    π_r = {}

    for a in range(1,x+1):
        for b in range(1,x+1):
            for c in range(1,x+1):
                for d in range(1,x+1):
                
                    w = (a-1)*x**3 + (b-1)*x**2 + (c-1)*x + d
                    π_r[w] = Prob_comb[a-1]*Prob_comb[b-1]*Prob_comb[c-1]*Prob_comb[d-1]
                    
                    for t in range(1,hours+1):

                        c_FCRs[(w,t)] = Rep_scen_comb[1][a-1][t-1]
                        c_aFRR_ups[(w,t)] = Rep_scen_comb[2][b-1][t-1]
                        c_aFRR_downs[(w,t)] = Rep_scen_comb[3][c-1][t-1]
                        c_mFRR_ups[(w,t)] = Rep_scen_comb[4][d-1][t-1]


    c_DAs = {}
    π_DA = {}
    for i in range(1,x+1):
        π_DA[(i)] = Prob_comb[i-1] 
        for t in range(1,hours+1):
            c_DAs[(i,t)] = Rep_scen_comb[0][i-1][t-1]


    return Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA

def PV_Blocks(PV,weeks,blocksize_PV):

    weeks = weeks
    #Sample length
    n = weeks*blocksize_PV

    #Split PV in blocks of length blocksize

    PV_blocks = [PV[i:i + blocksize_PV ] for i in range (0,n,blocksize_PV)]

    #Delete last block if length differs from blocksize 
    if len(PV_blocks[-1]) != blocksize_PV:
        del PV_blocks[-1]

    return PV_blocks


## Plot functions  
def ScenariosPlots(Rep_scen,scenarios,sample_length,n_samples):


    x = np.arange(0,sample_length)

    fig, ax = plt.subplots(nrows=1,ncols=1)

    #ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
    for i in range(0,n_samples):
        ax.plot(x, scenarios[0][i], color='lightgrey',linestyle = '-', linewidth=1)

        
    for i in range(0,len(Rep_scen[0])):
        ax.plot(x, Rep_scen[0][i],linestyle = '-', label =f"ω{i+1}", linewidth=2)    

        
    ax.set_ylabel('€/MW')
    ax.set_xlabel('Hours')
    #ax.set_ylim([-60, 170])
    ax.legend(loc='upper left')
    #ax.set_title('Day-Ahead Price')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()

def PlotPV_Rep(PV_block,PV_rep):

    x = np.arange(0,len(PV_block[0]))

    fig, ax = plt.subplots(nrows=1,ncols=1)

    #ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
    for i in range(0,len(PV_block)):
        ax.plot(x, PV_block[i], color='lightgrey',linestyle = '-', linewidth=1)

        

    ax.plot(x, PV_rep[0],linestyle = '-',color='red' ,label =f"PV Representation", linewidth=2)    

        
    ax.set_ylabel('MW')
    ax.set_xlabel('Hours')
    #ax.set_ylim([-60, 170])
    ax.legend(loc='upper left')
    #ax.set_title('Day-Ahead Price')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()



if Type == 'single':
    scenarios = Bootsrap(Type,Data,Data_names,n_samples,blocksize,sample_length)
    Rep_scen,Prob = K_Medoids(scenarios,n_clusters)   
    Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA = SingleInputData(Rep_scen,Prob)



if Type == 'combined':
    ## Generate Average Price for all markets for each time (Only for "Combined scenario generation"!!) ##

    ## For DA, aFRR_up & down and mFRR 
    scenarios = Bootsrap(Type,Data_comb,Data_comb_names,n_samples,blocksize,sample_length)
    Avg_scenarios = GenAverage(scenarios,n_samples,sample_length)
    Rep_scen_comb, Prob_comb = AvgKmedReduction(Avg_scenarios,scenarios,n_clusters,n_samples,sample_length) 

    Data_FCR = [FCR]
    Data_names_FCR = ['FCR']
    Type = 'single'
    scenarios_DA = Bootsrap(Type,Data_FCR,Data_names_FCR,n_samples,blocksize,sample_length)
    kmedoids = KMedoids(n_clusters=n_clusters,metric='euclidean').fit(scenarios_DA[1])

    FCR_red_scen = kmedoids.cluster_centers_


    Prob_FCR = np.zeros((n_clusters))

    for j in range(0,n_clusters):
                Prob_FCR[j] = np.count_nonzero(kmedoids.labels_ == j)/len(kmedoids.labels_)

    #Rep_scen_comb[market][scen][time]

    Rep_scen_combALL = np.zeros((5,n_clusters,len(Rep_scen_comb[0][0])))
    Rep_scen_combALL[0] = Rep_scen_comb[0]  ## Day ahead 
    Rep_scen_combALL[1] = FCR_red_scen  ## FCR
    Rep_scen_combALL[2] = Rep_scen_comb[1]  ## aFRR up 
    Rep_scen_combALL[3] = Rep_scen_comb[2]  ## aFRR down 
    Rep_scen_combALL[4] = Rep_scen_comb[3]  ## mFRR  


    Prob_comb_all = np.zeros((n_clusters,len(Rep_scen_combALL)))

    for i in range(0,len(Rep_scen_combALL)):
        for j in range(0,n_clusters):
            if i != 1:
                Prob_comb_all[j][i] = Prob_comb[j]
            if i == 1:
                Prob_comb_all[j][i] = Prob_FCR[j]


        Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA = SingleInputData(Rep_scen_combALL,Prob_comb_all)

#Rep_scen_comb[0]   ### markets , scenario , time 


if PV_Cluster == 'True': 

    PV_block = PV_Blocks(PV,weeks,blocksize_PV)
    kmedoids = KMedoids(n_clusters=n_clusters_PV,metric='euclidean').fit(PV_block)
    PV_rep = kmedoids.cluster_centers_
    


    #Model input
    P_PV_max = {}
    
    
    for t in range(1,len(PV_rep[0])+1):
        P_PV_max[t] = PV_rep[0][t-1]













#ScenariosPlots(Rep_scen,scenarios,sample_length,n_samples)
#PlotPV_Rep(PV_block,PV_rep)


## Scenario reduction ## 

## Scenario reduction (Single!!!) ## 
#Specify number of clusters(scenarios)




# Rep_scen[0] = DA scenarios, Rep_scen[1] = FCR scenarios, Rep_scen[2] = aFRR_up scenarios, Rep_scen[3] = aFRR_Down scenarios, Rep_scen[4] = mFRR scenarios 
#Rep_scen[1][0][0]   #Market , Omega, time 

#Prob # Prob scenario 1 in DA = Prob[0,0], Prob scenario 2 in DA = Prob[1,0] osv... Prob scenario 1 FCR = Prob[0,1] ..... 



## Scenario reduction (Combined!!!) ##
#Specify number of clusters(scenarios)



#Rep_scen_comb[market][scenario][hour]






### Model Input ### 

#Rep_scen_comb[market][scenario][hour]

# Rep_scen[0] = DA scenarios, Rep_scen[1] = FCR scenarios, Rep_scen[2] = aFRR_up scenarios, Rep_scen[3] = aFRR_Down scenarios, Rep_scen[4] = mFRR scenarios 
#Rep_scen[1][0][0]   #Market , Omega, time 

#Prob # Prob scenario 1 in DA = Prob[0,0], Prob scenario 2 in DA = Prob[1,0] osv... Prob scenario 1 FCR = Prob[0,1] ..... 

#len(Rep_scen[1][1])

#Rep_scen[0][0][0]






#Rep_scen[0]
#hours = len(Rep_scen[0][0])













a = np.array([10,20,30,40,50])
b = np.array([1,2,3,4,5])

a - b



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
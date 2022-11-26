import datetime
import numpy as np
import pandas as pd 
from pathlib import Path
from Data_process import df_aFRR,df_mFRR,df_FCR_DE,DA_list
import random
from Settings import*
random.seed(123)


DA = DA_list
aFRR_up = df_aFRR['aFRR Upp Pris (EUR/MW)'].tolist()
aFRR_down = df_aFRR['aFRR Ned Pris (EUR/MW)'].tolist()
mFRR = df_mFRR['mFRR_UpPriceEUR'].tolist()
FCR = df_FCR_DE['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 

Data = [DA,FCR,aFRR_up,aFRR_down,mFRR]
Data_names = ['DA','FCR','aFRR_up','aFRR_down','mFRR']

Type = 'combined'   # 'single' or 'combined'

n_samples = 3 #Number of samples to be made  

blocksize = 24 # 7days = 168 hours
sample_length = blocksize*5 # sampling 52 weeks blocks

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
                    print(r)

                    for j in range(0,blocksize):
                        samples[i,t+j] = blocks[r][j]

                    t = t + blocksize
        
            if Data_names[x] == 'DA':
                DA_block = samples
            if Data_names[x] == 'FCR':
                FCR_block = samples
            if Data_names[x] == 'aFRR_up':
                aFRR_up_block = samples
            if Data_names[x] == 'aFRR_down':
                aFRR_down_block = samples
            if Data_names[x] == 'mFRR':
                mFRR_block = samples


                
    if Type == 'combined':

        ########## Multi Blocks ######## 
        ### Multi ### 
        df = pd.DataFrame({'DA':  np.array(Data[0]), 'FCR':  np.array(Data[1]) , 'aFRR_up:':  np.array(Data[2]), 'aFRR_down': np.array(Data[3]), 'mFRR':  np.array(Data[4])})

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






### TEST ### 
data1 = [1,2,3,4,5,6,7,8,9,10]
data2 = [10,20,30,40,50,60,70,80,90,100]
data3 = [100,200,300,400,500,600,700,800,900,1000]
data4 = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
data5 = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
df_test = pd.DataFrame({'Data1':data1,'Data2':data2,'Data3':data3,'Data4':data4,'Data5':data5})

#df_test.corr()

DA = df_test['Data1'].tolist()
FCR =  df_test['Data2'].tolist() 
aFRR_up =  df_test['Data3'].tolist()
aFRR_down =  df_test['Data4'].tolist()
mFRR =  df_test['Data5'].tolist()


Data = [DA,FCR,aFRR_up,aFRR_down,mFRR]
Data_names = ['DA','FCR','aFRR_up','aFRR_down','mFRR']

Type = 'combined'   # 'single' or 'combined'

n_samples = 5 #Number of samples to be made  

blocksize = 2 # 7days = 168 hours
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
                    print(r)

                    for j in range(0,blocksize):
                        samples[i,t+j] = blocks[r][j]

                    t = t + blocksize
        
            if Data_names[x] == 'DA':
                DA_block = samples
            if Data_names[x] == 'FCR':
                FCR_block = samples
            if Data_names[x] == 'aFRR_up':
                aFRR_up_block = samples
            if Data_names[x] == 'aFRR_down':
                aFRR_down_block = samples
            if Data_names[x] == 'mFRR':
                mFRR_block = samples


                
    if Type == 'combined':

        ########## Multi Blocks ######## 
        ### Multi ### 
        df = pd.DataFrame({'DA':  np.array(Data[0]), 'FCR':  np.array(Data[1]) , 'aFRR_up:':  np.array(Data[2]), 'aFRR_down': np.array(Data[3]), 'mFRR':  np.array(Data[4])})

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










### Multi ### 

data1 = np.array([1,2,3,4,5,6,7,8,9,10])
data2 = np.array([100,200,300,400,500,600,700,800,900,1000])
data3 = np.array([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
data4 = np.array([10000,20000,30000,40000,50000,60000,70000,80000,90000,100000])

df = pd.DataFrame({'Data 1': data1, 'Data 2': data2,'Data 3': data3, 'Data 4':data4})

data = df.values.tolist() #Acces element by  data[0][0]

n = len(data)

#Block size
blocksize = 3   # 7days = 168 hours

#Split sample in blocks of length blocksize

blocks = [data[i:i + blocksize ] for i in range (0,n,blocksize)]#Acces element by blocks[0][0][0]

#Delete last block if length differs from blocksize 
if len(blocks[-1]) != blocksize:
    del blocks[-1]

len_element = len(blocks[1][1])
n_samples = 3 
sample_length = blocksize*3 # sampling 52 weeks blocks



## creating an array with zeros and same dimensions as blocks 
samples = np.zeros((n_samples,sample_length,len_element))



samples[1,1] = blocks[1][2]

for i in range(0,n_samples):
    t = 0
    while t < sample_length:

        r = random.randrange(0,len(blocks))
        print(r)

        for j in range(0,blocksize):
        
            samples[i,t+j] = blocks[r][j]

        t = t + blocksize


print(np.size(samples,0)) #x axis
print(np.size(samples,1)) #y axis


for j in range(0,2):
    print(j)

samples[0]







# size of samples

print(np.size(samples,0)) #x axis
print(np.size(samples,1)) #y axis




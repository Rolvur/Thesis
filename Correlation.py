from turtle import width
from Data_process import df_SEafrr2021_raw,df_SEafrr2020_raw,df_SEafrr2022_raw
#from Opt_Model_V2 import df_results
from Opt_Constants import *
import scipy
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as md
from statistics import mean
from pathlib import Path
import datetime
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from IPython.display import display

##################### aFRR #######################
file_to_open = Path("Data/") / "SE_AFRR_2020.csv"
df_SEafrr2020_raw = pd.read_csv(file_to_open,sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
file_to_open = Path("Data/") / "SE_AFRR_2021.csv"
df_SEafrr2021_raw = pd.read_csv(file_to_open,sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
file_to_openn = Path("Data/") / "SE_AFRR_2022.xlsx"
df_SEafrr2022_raw = pd.read_excel(file_to_openn)

file_to_open = Path("Data/") / "FI_AFFR_2020.xlsx"
df_FIafrr2020_raw = pd.read_excel(file_to_open, index_col=2)
file_to_open = Path("Data/") / "FI_AFFR_2021.xlsx"
df_FIafrr2021_raw = pd.read_excel(file_to_open, index_col=2)
file_to_open = Path("Data/") / "FI_AFFR_2022.xlsx"
df_FIafrr2022_raw = pd.read_excel(file_to_open, index_col=2)

df_FIafrr_raw = pd.concat([df_FIafrr2020_raw,df_FIafrr2021_raw,df_FIafrr2022_raw])




df_SEafrr2020_raw.drop([8808], axis=0, inplace = True) #Dropping summation 
df_SEafrr2021_raw.drop([8784], axis=0, inplace = True) #Dropping summation 

df_SEafrr2020_raw['Period'] =  df_SEafrr2020_raw['Period'].apply(pd.to_datetime)
df_SEafrr2021_raw['Period'] =  df_SEafrr2021_raw['Period'].apply(pd.to_datetime)
df_SEafrr2022_raw['Period'] =  df_SEafrr2022_raw['Period'].apply(pd.to_datetime)


df_SEafrr2020_raw['aFRR Upp Pris (EUR/MW)'].mean()
df_SEafrr2021_raw['aFRR Upp Pris (EUR/MW)'].mean()
df_SEafrr2022_raw['aFRR Upp Pris (EUR/MW)'].mean()

df_SEafrr2020_raw['aFRR Ned Pris (EUR/MW)'].mean()
df_SEafrr2021_raw['aFRR Ned Pris (EUR/MW)'].mean()
df_SEafrr2022_raw['aFRR Ned Pris (EUR/MW)'].mean()


df_FIafrr2020_raw['Automatic Frequency Restoration Reserve, price, up'].mean()
df_FIafrr2021_raw['Automatic Frequency Restoration Reserve, price, up'].mean()
df_FIafrr2022_raw['Automatic Frequency Restoration Reserve, price, up'].mean()

df_FIafrr2020_raw['Automatic Frequency Restoration Reserve, price, down'].mean()
df_FIafrr2021_raw['Automatic Frequency Restoration Reserve, price, down'].mean()
df_FIafrr2022_raw['Automatic Frequency Restoration Reserve, price, down'].mean()


df_FIaFRR = pd.concat([df_FIafrr2020_raw,df_FIafrr2021_raw,df_FIafrr2022_raw])
df_FIaFRR = df_FIaFRR[['Start time UTC+02:00','Automatic Frequency Restoration Reserve, price, up','Automatic Frequency Restoration Reserve, price, down']]


df_SEaFRR = pd.concat([df_SEafrr2020_raw,df_SEafrr2021_raw,df_SEafrr2022_raw])

df_SEaFRR = df_SEaFRR[['Period','aFRR Upp Pris (EUR/MW)','aFRR Ned Pris (EUR/MW)']]




#swap points and rebounds columns



fig, ax = plt.subplots(nrows=1,ncols=1)

ax.plot(df_SEaFRR['Period'], df_SEaFRR['aFRR Upp Pris (EUR/MW)'], color = 'navy', label = 'Sweden Up' )
ax.plot(df_SEaFRR['Period'], df_SEaFRR['aFRR Ned Pris (EUR/MW)'], color = 'maroon', label = 'Sweden Down')


ax.set_ylabel('[€/MW]')
ax.legend(loc='upper left')



ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show() 




##################### FCR #######################
file_to_open = Path("Data/") / "RESULT_CAPACITY_2019_FCR .csv"
df_FCRR2019_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "RESULT_CAPACITY_2020_FCR.csv"
df_FCRR2020_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "RESULT_CAPACITY_2021_FCR.csv"
df_FCRR2021_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "RESULT_CAPACITY_2022_FCR.csv"
df_FCRR2022_raw = pd.read_csv(file_to_open,sep=',')

#### Cleaning up FCR data

#Merging the two datasets
df_FCRR_raw = pd.concat([df_FCRR2020_raw, df_FCRR2021_raw], ignore_index=True, sort=False)
del df_FCRR_raw['Column1']
del df_FCRR_raw['Unnamed: 31']
del df_FCRR_raw['Unnamed: 32']

#deleteting any row where column 'TENDER_NUMBER' is not '1'
df_FCR = df_FCRR_raw[df_FCRR_raw.TENDER_NUMBER == 1]
#resetting index
df_FCR.reset_index(drop=True)
#creating a dictionary linking the PRODUCTNAME to the amount of hours in block
dic_block = {'NEGPOS_00_24': 24, 'NEGPOS_00_04': 4, 'NEGPOS_04_08': 4, 'NEGPOS_08_12': 4, 'NEGPOS_12_16': 4, 'NEGPOS_16_20': 4, 'NEGPOS_20_24': 4}

#writing new column in FCR data using dictionary
df_FCR.insert(len(df_FCR.columns),'BLOCK_LENGTH',df_FCR.iloc[:,4].map(dic_block))
#df_FCR.iloc[:,30] = df_FCR.iloc[:,4].map(dic_block)
df_FCR = df_FCR.loc[df_FCR.index.repeat(df_FCR.BLOCK_LENGTH)].reset_index(drop=True)


#writing hour to all data
for j in range(0,int(len(df_FCR['DATE_FROM'])/24)):
    i = j*24
    df_FCR.iloc[i,0] = df_FCR.iloc[i,0] + " 00:00"
    df_FCR.iloc[i+1,0] = df_FCR.iloc[i+1,0] + " 01:00" 
    df_FCR.iloc[i+2,0] = df_FCR.iloc[i+2,0] + " 02:00"
    df_FCR.iloc[i+3,0] = df_FCR.iloc[i+3,0] + " 03:00" 
    df_FCR.iloc[i+4,0] = df_FCR.iloc[i+4,0] + " 04:00"
    df_FCR.iloc[i+5,0] = df_FCR.iloc[i+5,0] + " 05:00" 
    df_FCR.iloc[i+6,0] = df_FCR.iloc[i+6,0] + " 06:00"
    df_FCR.iloc[i+7,0] = df_FCR.iloc[i+7,0] + " 07:00" 
    df_FCR.iloc[i+8,0] = df_FCR.iloc[i+8,0] + " 08:00"
    df_FCR.iloc[i+9,0] = df_FCR.iloc[i+9,0] + " 09:00" 
    df_FCR.iloc[i+10,0] = df_FCR.iloc[i+10,0] + " 10:00"
    df_FCR.iloc[i+11,0] = df_FCR.iloc[i+11,0] + " 11:00" 
    df_FCR.iloc[i+12,0] = df_FCR.iloc[i+12,0] + " 12:00"
    df_FCR.iloc[i+13,0] = df_FCR.iloc[i+13,0] + " 13:00" 
    df_FCR.iloc[i+14,0] = df_FCR.iloc[i+14,0] + " 14:00"
    df_FCR.iloc[i+15,0] = df_FCR.iloc[i+15,0] + " 15:00" 
    df_FCR.iloc[i+16,0] = df_FCR.iloc[i+16,0] + " 16:00"
    df_FCR.iloc[i+17,0] = df_FCR.iloc[i+17,0] + " 17:00" 
    df_FCR.iloc[i+18,0] = df_FCR.iloc[i+18,0] + " 18:00"
    df_FCR.iloc[i+19,0] = df_FCR.iloc[i+19,0] + " 19:00" 
    df_FCR.iloc[i+20,0] = df_FCR.iloc[i+20,0] + " 20:00"
    df_FCR.iloc[i+21,0] = df_FCR.iloc[i+21,0] + " 21:00" 
    df_FCR.iloc[i+22,0] = df_FCR.iloc[i+22,0] + " 22:00"
    df_FCR.iloc[i+23,0] = df_FCR.iloc[i+23,0] + " 23:00"
    
  

#df_FCR['DATE_FROM'] = df_FCR['DATE_FROM'].astype(str) #probably not necessary, at the field is already a string
#Converting time to datetime
for i in range(0,len(df_FCR['DATE_FROM'])):
    df_FCR.iloc[i,0] = datetime.datetime.strptime(df_FCR.iloc[i,0], '%d/%m/%Y %H:%M')
    df_FCR.iloc[i,0] = df_FCR.iloc[i,0].strftime('%Y-%m-%d %H:%M')


FCR_2020 = (df_FCR['DATE_FROM'] >= '2020-01-01 00:00') & (df_FCR['DATE_FROM']  <= '2020-12-31 23:59')
FCR_2021 = (df_FCR['DATE_FROM'] >= '2021-01-01 00:00') & (df_FCR['DATE_FROM']  <= '2021-12-31 23:59')
df_FCR_2020 = df_FCR[FCR_2020]
df_FCR_2021 = df_FCR[FCR_2021]

#Deleting rows where a series has missing data


columns2020 = ['AT_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','BE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','CH_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]',
            'FR_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','NL_SETTLEMENTCAPACITY_PRICE_[EUR/MW]']



columns2021 = ['AT_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','BE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','CH_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]',
            'FR_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','NL_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','SI_SETTLEMENTCAPACITY_PRICE_[EUR/MW]',
                'DK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]']


for i in columns2020:
    a = df_FCRR2019_raw.index[df_FCRR2019_raw[i] == '-']
    df_FCRR2019_raw.drop(a, axis=0, inplace = True)

for i in columns2020:
    a = df_FCR_2020.index[df_FCR_2020[i] == '-']
    df_FCR_2020.drop(a, axis=0, inplace = True)

for i in columns2021:


    b = df_FCR_2021.index[df_FCR_2021[i] == '-']

    df_FCR_2021.drop(b, axis=0, inplace = True)

for i in columns2021:


    b = df_FCRR2022_raw.index[df_FCRR2022_raw[i] == '-']

    df_FCRR2022_raw.drop(b, axis=0, inplace = True)




FCR2019 = df_FCRR2019_raw[columns2020].astype(float)
FCR2020 = df_FCR_2020[columns2020].astype(float)
FCR2021 = df_FCR_2021[columns2021].astype(float)
FCR2022 = df_FCRR2022_raw[columns2021].astype(float)

# Renaming the columns 

FCR2019.columns = ['Austira', 'Belgium', 'Czech', 'Germany', 'France', 'Netherlands']
FCR2020.columns = ['Austira', 'Belgium', 'Czech', 'Germany', 'France', 'Netherlands']
FCR2021.columns =  ['Austira', 'Belgium', 'Czech', 'Germany', 'France', 'Netherlands', 'Switzerland', 'Denmark']
FCR2022 = ['Austira', 'Belgium', 'Czech', 'Germany', 'France', 'Netherlands', 'Switzerland', 'Denmark']



Cor2019 = FCR2019.corr()
Cor2020 = FCR2020.corr()
Cor2021 = FCR2021.corr()
Cor2022 = FCR2021.corr()
 

sn.heatmap(Cor2022, annot=True)
sn.color_palette("husl", 8)
plt.tight_layout()
plt.show()

##Average correlation 

Avg_Corr = pd.DataFrame({'2019' : Cor2019.mean(),'2020' : Cor2020.mean(),'2021' : Cor2021.mean(),'2022' : Cor2022.mean() })
display(Avg_Corr)

sn.heatmap(Avg_Corr, annot=True)
sn.color_palette("husl", 8)
plt.tight_layout()
plt.show()



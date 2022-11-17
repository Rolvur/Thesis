
import datetime
import numpy as np
import pandas as pd 
#from Settings import Start_date, End_date, Demand_pattern
#from Opt_Constants import k_d
#from pyparsing import line
import warnings
import openpyxl
from pathlib import Path


##################### FCR #######################
file_to_open = Path("Data/") / "RESULT_CAPACITY_2019_FCR .csv"
df_FCRR2019_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "ANONYM_CAPACITY_2020_BIDS_FCR.csv"
df_FCRA2020_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "DEMAND_CAPACITY_2020_FCR.csv"
df_FCRD2020_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "RESULT_CAPACITY_2020_FCR.csv"
df_FCRR2020_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "ANONYM_CAPACITY_2021_BIDS_FCR.csv"
df_FCRA2021_raw = pd.read_csv(file_to_open,sep=',')
file_to_open = Path("Data/") / "DEMAND_CAPACITY_2021_FCR.csv"
df_FCRD2021_raw = pd.read_csv(file_to_open,sep=',')
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



#check if any date is appearing more than once:
#for i in range (0,len(df_FCR['DATE_FROM'])-24):
#    if df_FCR['DATE_FROM'][i] == df_FCR['DATE_FROM'][i+24]:
#        print(i) 

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


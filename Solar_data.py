import pandas as pd 
import numpy as np
import datetime
from pathlib import Path

from os import listdir
from os.path import isfile, join
from datetime import datetime
from datetime import timedelta

#Getting names of all files in folder 
file_names = [f for f in listdir('Solar_data') if isfile(join('Solar_data', f))]


df_results = pd.DataFrame()

def GetHourlyAverage(file_name):

    #Reading file 
    file_to_open = Path("Solar_data/") / file_name
    df = pd.read_csv(file_to_open)
    #From ms to s
    df['TIMESTAMP'] = (df['TIMESTAMP']/1000)-60*60 
    #Converting to dates 
    df['TIMESTAMP'] = [datetime.fromtimestamp(x) for x in df['TIMESTAMP']]


    #Taking hourly average values 
    df_avg = df.groupby(pd.PeriodIndex(df['TIMESTAMP'], freq='H'))['INSOLATION_irrad1[kW/m2]'].mean()


    return df_avg


for i in file_names:

   df_hourly = GetHourlyAverage(i)
   df_results = pd.concat([df_results,df_hourly])










df_results.columns=['Irradiance']


df_avg_results = df_results.groupby(pd.PeriodIndex(df_results.index, freq='H'))['Irradiance'].mean()


df_avg_results[df_avg_results<0] = 0

df_avg_results = df_avg_results[:-1]

### Dealing with missing values 

#from 4 jan kl 11 to 15 jan kl 11 
#There are missing values from jan 9 kl 11 to jan 10 kl 11
#Take average from 3 days back and 3 days forward and use that where missing values are (Did that in excel)


## Reading the solar data 
file_to_open = Path("Data/") / "DTU_solar_data.xlsx"
df_solar = pd.read_excel(file_to_open)


## Number of solar panels
n_pv = 300*10**6/600 


df_solar['Power [MW]'] = ((df_solar['Irradiance']/0.8)*454.6*n_pv)/10**6
 
df_solar.to_excel('PV_data.xlsx')




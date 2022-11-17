import datetime
import numpy as np
import pandas as pd 
from Settings import Start_date, End_date, Demand_pattern
from Opt_Constants import k_d
#from pyparsing import line
import warnings
import openpyxl
from pathlib import Path
##################### Solar #######################


#Input for model
file_to_open = Path("Data/") / "PV_data.xlsx"
df_solar_prod = pd.read_excel(file_to_open)

TimeRangePV = (df_solar_prod['Unnamed: 0'] >= Start_date) & (df_solar_prod['Unnamed: 0']  <= End_date)
df_solar_prod_time = df_solar_prod[TimeRangePV]

PV = df_solar_prod_time['Power [MW]'].tolist() #Convert from pandas data series to list

P_PV_max = dict(zip(np.arange(1,len(PV)+1),PV))

##################### Day ahead #######################
file_to_open = Path("Data/") / "Elspotprices_RAW.csv"
df_DKDA_raw = pd.read_csv(file_to_open,sep=';',decimal=',')

#Converting to datetime
df_DKDA_raw[['HourUTC','HourDK']] = df_DKDA_raw[['HourUTC','HourDK']].apply(pd.to_datetime)
#save to Excel 
#df_DKDA_raw.to_excel("Result_files/DA.xlsx")


#Input for model
#Using year 2020
TimeRange2020DA = (df_DKDA_raw['HourDK'] >= Start_date) & (df_DKDA_raw['HourDK']  <= End_date)
df_DKDA_raw2020 = df_DKDA_raw[TimeRange2020DA]
DA = df_DKDA_raw2020['SpotPriceEUR,,'].tolist()
DA = dict(zip(np.arange(1,len(DA)+1),DA))
#print(DA,Start_date,End_date)

#Getting time range
DateRange = df_DKDA_raw2020['HourDK']


#df_FCR_DE['DATE_FROM'] = df_FCR['DATE_FROM']
file_to_open = Path("Data/") / "df_FCR_DE.csv"

df_FCR_DE = pd.read_csv(file_to_open,sep=',',low_memory=False)
df_FCR_DE['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'] = df_FCR_DE['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].astype(float)

#Input for model
TimeRange_FCR = (df_FCR_DE['DATE_FROM'] >= Start_date) & (df_FCR_DE['DATE_FROM']  <= End_date)
df_FCR_DE = df_FCR_DE[TimeRange_FCR]


#Convert from pandas data series to list
list_FCR = df_FCR_DE['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
c_FCR = dict(zip(np.arange(1,len(list_FCR)+1),list_FCR))


##################### mFRR #######################
file_to_open = Path("Data/") / "MfrrReservesDK1.csv"
df_DKmFRR_raw = pd.read_csv(file_to_open,sep=';', decimal=',')


#Converting to datetime
df_DKmFRR_raw[['HourUTC','HourDK']] =  df_DKmFRR_raw[['HourUTC','HourDK']].apply(pd.to_datetime)
df_mFRR = df_DKmFRR_raw.iloc[0:24095,:]
df_mFRR = df_mFRR[::-1]
sum(df_mFRR['mFRR_UpPriceEUR'])


TimeRange_mFRR = (df_mFRR['HourDK'] >= Start_date) & (df_mFRR['HourDK']  <= End_date)
df_mFRR = df_mFRR[TimeRange_mFRR]
#convert to list
list_mFRR_up = df_mFRR['mFRR_UpPriceDKK'].tolist() #Convert from pandas data series to list

#convert to dict
c_mFRR_up = dict(zip(np.arange(1,len(list_mFRR_up)+1),list_mFRR_up))




##################### aFRR #######################

file_to_open = Path("Data/") / "df_aFRR.xlsx"
df_aFRR = pd.read_excel(file_to_open)

#reduce data point to the chosen time period
TimeRange_aFRR = (df_aFRR['Period'] >= Start_date) & (df_aFRR['Period']  <= End_date)
df_aFRR = df_aFRR[TimeRange_aFRR]
#convert to list
list_aFRR_up = df_aFRR['aFRR Upp Pris (EUR/MW)'].tolist() #Convert from pandas data series to list
list_aFRR_down = df_aFRR['aFRR Ned Pris (EUR/MW)'].tolist() #Convert from pandas data series to list

#convert to dict
c_aFRR_up = dict(zip(np.arange(1,len(list_aFRR_up)+1),list_aFRR_up))
c_aFRR_down = dict(zip(np.arange(1,len(list_aFRR_down)+1),list_aFRR_down))

######################PEM efficiency table###################

file_to_openn = Path("Data/") / "PEM_efficiency_curve.xlsx"
PEM_efficiency_raw = pd.read_excel(file_to_openn, index_col=2)
pem_setpoint = PEM_efficiency_raw['p_pem'].tolist() 
hydrogen_mass_flow = PEM_efficiency_raw['m'].tolist()

##################### Methanol Demand #######################


#Input for model



Demand = list(0 for i in range(0,len(PV)))

if Demand_pattern == 'Hourly':
    for i in range(0,len(Demand),1):
        Demand[i] = k_d

if Demand_pattern == 'Daily':
    for i in range(0,len(Demand),24):
        Demand[i+23] = k_d*24
        
if Demand_pattern == 'Weekly':
    for i in range(1,1+int(len(PV)/(24*7))):
        Demand[i*24*7-1] = k_d*24*7

    dw = len(PV)/(24*7) - int(len(PV)/(24*7))
    if dw > 0:
        Demand[len(PV)-1] = dw*7*24*k_d


            
Demand = dict(zip(np.arange(1,len(Demand)+1),Demand))





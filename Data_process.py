import datetime
import numpy as np
import pandas as pd 
from Settings import Start_date, End_date, Demand_pattern,Start_date_scen,End_date_scen
from Opt_Constants import k_d
#from pyparsing import line
import warnings
import openpyxl
from pathlib import Path
##################### Solar #######################


#Input for model
file_to_open = Path("Data/") / "PV_data.xlsx"
df_solar_prod = pd.read_excel(file_to_open)

#Rep_week2020byAvg = PV_data[(PV_data['Hour UTC'] >= '2020-08-24 00:00') & (PV_data['Hour UTC']  <= '2020-08-30 23:59')]
#Rep_week2021byAvg = PV_data[(PV_data['Hour UTC'] >= '2021-03-15 00:00') & (PV_data['Hour UTC']  <= '2021-03-21 23:59')]



TimeRangePV = (df_solar_prod['Hour UTC'] >= '2021-03-15 00:00') & (df_solar_prod['Hour UTC']  <= '2021-03-21 23:59')
TimeRangePV_scen = (df_solar_prod['Hour UTC'] >= Start_date_scen) & (df_solar_prod['Hour UTC']  <= End_date_scen)

df_solar_prod_time = df_solar_prod[TimeRangePV]
PV_scen = df_solar_prod[TimeRangePV_scen]

PV_scenPower = PV_scen['Power [MW]'].tolist() #Convert from pandas data series to list

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
TimeRange2020DA = (df_DKDA_raw['HourDK'] >= Start_date) & (df_DKDA_raw['HourDK']  <= End_date)
TimeRangeScenarioDA = (df_DKDA_raw['HourDK'] >= Start_date_scen) & (df_DKDA_raw['HourDK']  <= End_date_scen)

df_DKDA_raw2020 = df_DKDA_raw[TimeRange2020DA]
df_DKDA_rawScen = df_DKDA_raw[TimeRangeScenarioDA]

DA_list = df_DKDA_raw2020['SpotPriceEUR,,'].tolist()
DA_list_scen = df_DKDA_rawScen['SpotPriceEUR,,'].tolist()


DA = dict(zip(np.arange(1,len(DA_list)+1),DA_list))
#print(DA,Start_date,End_date)

#Getting time range
DateRange = df_DKDA_raw2020['HourDK']


#df_FCR_DE['DATE_FROM'] = df_FCR['DATE_FROM']
file_to_open = Path("Data/") / "df_FCR_DE.csv"

df_FCR_DE_raw = pd.read_csv(file_to_open,sep=',',low_memory=False)
df_FCR_DE_raw['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'] = df_FCR_DE_raw['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].astype(float)


#Input for model
TimeRange_FCR = (df_FCR_DE_raw['DATE_FROM'] >= Start_date) & (df_FCR_DE_raw['DATE_FROM']  <= End_date)
TimeRangeFCR_Scen = (df_FCR_DE_raw['DATE_FROM'] >= Start_date_scen) & (df_FCR_DE_raw['DATE_FROM']  <= End_date_scen)

df_FCR_DE = df_FCR_DE_raw[TimeRange_FCR]
df_FCR_DE_scen = df_FCR_DE_raw[TimeRangeFCR_Scen]

#df_FCR_DE.to_csv("Data/df_FCR.csv")


#Convert from pandas data series to list
list_FCR = df_FCR_DE['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
c_FCR = dict(zip(np.arange(1,len(list_FCR)+1),list_FCR))


##################### mFRR #######################
file_to_open = Path("Data/") / "MfrrReservesDK1.csv"
df_DKmFRR_raw = pd.read_csv(file_to_open,sep=';', decimal=',')


#Converting to datetime
df_DKmFRR_raw[['HourUTC','HourDK']] =  df_DKmFRR_raw[['HourUTC','HourDK']].apply(pd.to_datetime)
df_mFRR = df_DKmFRR_raw.iloc[0:24095,:]
df_mFRR_raw = df_mFRR[::-1]
sum(df_mFRR['mFRR_UpPriceEUR'])

TimeRange_mFRR = (df_mFRR_raw['HourDK'] >= Start_date) & (df_mFRR_raw['HourDK']  <= End_date)
TimeRange_mFRR_Scen = (df_mFRR_raw['HourDK'] >= Start_date_scen) & (df_mFRR_raw['HourDK']  <= End_date_scen)

#df_mFRR_raw2020and21 = df_mFRR_raw[TimeRange_mFRR]

#df_mFRR_raw2020and21.to_csv("Data/df_mFRR.csv")


df_mFRR = df_mFRR_raw[TimeRange_mFRR]
df_mFRR_scen = df_mFRR_raw[TimeRange_mFRR_Scen]

#convert to list
list_mFRR_up = df_mFRR['mFRR_UpPriceEUR'].tolist() #Convert from pandas data series to list

#convert to dict
c_mFRR_up = dict(zip(np.arange(1,len(list_mFRR_up)+1),list_mFRR_up))




##################### aFRR #######################

file_to_open = Path("Data/") / "df_aFRR.xlsx"
df_aFRR_raw = pd.read_excel(file_to_open)

#reduce data point to the chosen time period
TimeRange_aFRR = (df_aFRR_raw['Period'] >= Start_date) & (df_aFRR_raw['Period']  <= End_date)
TimeRange_aFRR_Scen = (df_aFRR_raw['Period'] >= Start_date_scen) & (df_aFRR_raw['Period']  <= End_date_scen)

df_aFRR = df_aFRR_raw[TimeRange_aFRR]
df_aFRR_scen = df_aFRR_raw[TimeRange_aFRR_Scen]

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



######Creating test scenarios for reserve market variables#######
?? = 2
?? = 2
c_DA = {}
c_FCRs = {}
c_aFRR_ups = {}
c_aFRR_downs = {}
c_mFRR_ups = {}
??_r = {}
??_DA = {}
for i in range(1,??+1):
    ??_r[i] = 0.5
    ??_DA[i] = 0.5
    for j in range(1,len(c_FCR)+1):
        c_FCRs[(i,j)] = c_FCR[j]*i
        c_aFRR_ups[(i,j)] = c_aFRR_up[j]*(1-0.5*(i-1))
        c_aFRR_downs[(i,j)] = c_aFRR_down[j]*i
        c_mFRR_ups[(i,j)] = c_mFRR_up[j]*i
        c_DA[(i,j)] = DA[j]*i


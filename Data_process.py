import datetime
import numpy as np
import pandas as pd 
from Settings import Start_date, End_date, Demand_pattern
from Opt_Constants import k_d
#from pyparsing import line
import warnings
import openpyxl


##################### Solar #######################


#Solar production data set
df_solar_prod= pd.read_csv('PV production data 2019-2020.csv',sep=',')
df_solar_prod['time'] = df_solar_prod['time'].astype(str)

#Converting time to datetime
for i in range(0,len(df_solar_prod['time'])):
    df_solar_prod.iloc[i,0] = datetime.datetime.strptime(df_solar_prod.iloc[i,0], '%Y%m%d:%H%M')
    df_solar_prod.iloc[i,0] = df_solar_prod.iloc[i,0].strftime('%Y-%m-%d - %H:%M')
    
df_solar_prod['time'] = df_solar_prod['time'].apply(pd.to_datetime)


#Input for model
#Using year 2020
TimeRange2020PV = (df_solar_prod['time'] >= Start_date) & (df_solar_prod['time']  <= End_date)
df_solar_prod_2020 = df_solar_prod[TimeRange2020PV]

PV_Watt = df_solar_prod_2020['P'].tolist() #Convert from pandas data series to list
PV = [x/1000000 for x in PV_Watt]
P_PV_max = dict(zip(np.arange(1,len(PV)+1),PV))

#print(PV,Start_date,End_date)

#Solar irradiance data set
df_solar_irr= pd.read_csv('Irradiance data 2020-2021.csv',sep=',')
df_solar_irr[['YEAR','MO','DY','HR']] = df_solar_irr[['YEAR','MO','DY','HR']].astype(str)

#Adding 0 to time stamp to get similar length
for i in range(0,len(df_solar_irr['YEAR'])):
    #Month
    if len(df_solar_irr.iloc[i,1]) == 1: 
        df_solar_irr.iloc[i,1] ='0' + df_solar_irr.iloc[i,1] 
    #Day
    if len(df_solar_irr.iloc[i,2]) == 1: 
        df_solar_irr.iloc[i,2] ='0' + df_solar_irr.iloc[i,2] 
    #Hour
    if len(df_solar_irr.iloc[i,3]) == 1: 
        df_solar_irr.iloc[i,3] ='0' + df_solar_irr.iloc[i,3] 


df_solar_irr['Date'] = df_solar_irr['YEAR'] + df_solar_irr['MO'] + df_solar_irr['DY'] + df_solar_irr['HR'] 

#Converting time to datetime
for i in range(0,len(df_solar_irr['Date'])):
    df_solar_irr.iloc[i,9] = datetime.datetime.strptime(df_solar_irr.iloc[i,9], '%Y%m%d%H')
    df_solar_irr.iloc[i,9] = df_solar_irr.iloc[i,9].strftime('%Y-%m-%d - %H')
    

df_solar_irr['Date'] = df_solar_irr['Date'].apply(pd.to_datetime) 


##################### Day ahead #######################
df_DKDA_raw = pd.read_csv('Elspotprices_RAW.csv',sep=';',decimal=',')

#Converting to datetime
df_DKDA_raw[['HourUTC','HourDK']] = df_DKDA_raw[['HourUTC','HourDK']].apply(pd.to_datetime)


#Input for model
#Using year 2020
TimeRange2020DA = (df_DKDA_raw['HourDK'] >= Start_date) & (df_DKDA_raw['HourDK']  <= End_date)
df_DKDA_raw2020 = df_DKDA_raw[TimeRange2020DA]
DA = df_DKDA_raw2020['SpotPriceEUR,,'].tolist()
DA = dict(zip(np.arange(1,len(DA)+1),DA))
#print(DA,Start_date,End_date)

#Getting time range
DateRange = df_DKDA_raw2020['HourDK']




##################### FCR #######################

df_FCRA2020_raw = pd.read_csv('2020 - ANONYM_CAPACITY_BIDS_FCR.csv',sep=',')
df_FCRD2020_raw = pd.read_csv('2020 - DEMAND_CAPACITY_FCR.csv',sep=',')
df_FCRR2020_raw = pd.read_csv('2020 - RESULT_CAPACITY_FCR.csv',sep=',')

df_FCRA2021_raw = pd.read_csv('2021 - ANONYM_CAPACITY_BIDS_FCR.csv',sep=',')
df_FCRD2021_raw = pd.read_csv('2021 - DEMAND_CAPACITY_FCR.csv',sep=',')
df_FCRR2021_raw = pd.read_csv('2021 - RESULT_CAPACITY_FCR.csv',sep=',')

#### Cleaning up FCR data

#Merging the two datasets
df_FCRR_raw = pd.concat([df_FCRR2020_raw, df_FCRR2021_raw], ignore_index=True, sort=False)

#deleteting any row where column 'TENDER_NUMBER' is not '1'
df_FCR = df_FCRR_raw[df_FCRR_raw.TENDER_NUMBER == 1]
#resetting index
df_FCR.reset_index(drop=True)
#creating a dictionary linking the PRODUCTNAME to the amount of hours in block
dic_block = {'NEGPOS_00_24': 24, 'NEGPOS_00_04': 4, 'NEGPOS_04_08': 4, 'NEGPOS_08_12': 4, 'NEGPOS_12_16': 4, 'NEGPOS_16_20': 4, 'NEGPOS_20_24': 4}

#writing new column in FCR data using dictionary
df_FCR['BLOCK_LENGTH'] = df_FCR['PRODUCTNAME'].map(dic_block)
df_FCR = df_FCR.loc[df_FCR.index.repeat(df_FCR.BLOCK_LENGTH)].reset_index(drop=True)

#check if any date is appearing more than once:
#for i in range (0,len(df_FCR['DATE_FROM'])-24):
#    if df_FCR['DATE_FROM'][i] == df_FCR['DATE_FROM'][i+24]:
#        print(i) 

#writing hour to all data
for j in range(0,int(len(df_FCR['DATE_FROM'])/24)):
    i = j*24
    df_FCR['DATE_FROM'][i] = df_FCR['DATE_FROM'][i] + " 00:00"
    df_FCR['DATE_FROM'][i+1] = df_FCR['DATE_FROM'][i+1] + " 01:00" 
    df_FCR['DATE_FROM'][i+2] = df_FCR['DATE_FROM'][i+2] + " 02:00"
    df_FCR['DATE_FROM'][i+3] = df_FCR['DATE_FROM'][i+3] + " 03:00" 
    df_FCR['DATE_FROM'][i+4] = df_FCR['DATE_FROM'][i+4] + " 04:00"
    df_FCR['DATE_FROM'][i+5] = df_FCR['DATE_FROM'][i+5] + " 05:00" 
    df_FCR['DATE_FROM'][i+6] = df_FCR['DATE_FROM'][i+6] + " 06:00"
    df_FCR['DATE_FROM'][i+7] = df_FCR['DATE_FROM'][i+7] + " 07:00" 
    df_FCR['DATE_FROM'][i+8] = df_FCR['DATE_FROM'][i+8] + " 08:00"
    df_FCR['DATE_FROM'][i+9] = df_FCR['DATE_FROM'][i+9] + " 09:00" 
    df_FCR['DATE_FROM'][i+10] = df_FCR['DATE_FROM'][i+10] + " 10:00"
    df_FCR['DATE_FROM'][i+11] = df_FCR['DATE_FROM'][i+11] + " 11:00" 
    df_FCR['DATE_FROM'][i+12] = df_FCR['DATE_FROM'][i+12] + " 12:00"
    df_FCR['DATE_FROM'][i+13] = df_FCR['DATE_FROM'][i+13] + " 13:00" 
    df_FCR['DATE_FROM'][i+14] = df_FCR['DATE_FROM'][i+14] + " 14:00"
    df_FCR['DATE_FROM'][i+15] = df_FCR['DATE_FROM'][i+15] + " 15:00" 
    df_FCR['DATE_FROM'][i+16] = df_FCR['DATE_FROM'][i+16] + " 16:00"
    df_FCR['DATE_FROM'][i+17] = df_FCR['DATE_FROM'][i+17] + " 17:00" 
    df_FCR['DATE_FROM'][i+18] = df_FCR['DATE_FROM'][i+18] + " 18:00"
    df_FCR['DATE_FROM'][i+19] = df_FCR['DATE_FROM'][i+19] + " 19:00" 
    df_FCR['DATE_FROM'][i+20] = df_FCR['DATE_FROM'][i+20] + " 20:00"
    df_FCR['DATE_FROM'][i+21] = df_FCR['DATE_FROM'][i+21] + " 21:00" 
    df_FCR['DATE_FROM'][i+22] = df_FCR['DATE_FROM'][i+22] + " 22:00"
    df_FCR['DATE_FROM'][i+23] = df_FCR['DATE_FROM'][i+23] + " 23:00"  

df_FCR['DATE_FROM'] = df_FCR['DATE_FROM'].astype(str) #probably not necessary, at the field is already a string
#Converting time to datetime
for i in range(0,len(df_FCR['DATE_FROM'])):
    df_FCR.iloc[i,0] = datetime.datetime.strptime(df_FCR.iloc[i,0], '%d/%m/%Y %H:%M')
    df_FCR.iloc[i,0] = df_FCR.iloc[i,0].strftime('%Y-%m-%d - %H:%M')

#Input for model
TimeRange_FCR = (df_FCR['DATE_FROM'] >= Start_date) & (df_FCR['DATE_FROM']  <= End_date)
df_FCR = df_FCR[TimeRange_FCR]

list_FCR = df_FCR['DK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() #Convert from pandas data series to list
c_FCR = dict(zip(np.arange(1,len(list_FCR)+1),list_FCR))


##################### mFRR #######################

df_DKmFRR_raw = pd.read_csv('MfrrReservesDK1.csv',sep=';', decimal=',')

#Converting to datetime
df_DKmFRR_raw[['HourUTC','HourDK']] =  df_DKmFRR_raw[['HourUTC','HourDK']].apply(pd.to_datetime)


##################### aFRR #######################
#Reading data from csv
# Finland
df_FIafrr2020_raw = pd.read_excel('FI_AFFR_2020.xlsx', index_col=0)
df_FIafrr2021_raw = pd.read_excel('FI_AFFR_2021.xlsx', index_col=0)
#'End time UTC' 
#'Start time UTC+02:00' 
#'End time UTC+02:00'
#'Automatic Frequency Restoration Reserve, price, down'
#'Automatic Frequency Restoration Reserve, capacity, up'
#'Automatic Frequency Restoration Reserve, capacity, down'
#'Automatic Frequency Restoration Reserve, price, up'


#Sweden
df_SEafrr2020_raw = pd.read_csv('SE_AFRR_2020.csv',sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
df_SEafrr2021_raw = pd.read_csv('SE_AFRR_2021.csv',sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
#'Period'
# #'Elområde'
#'aFRR Upp Pris (EUR/MW)'
#'aFRR Upp Volym (MW)'
#'aFRR Ned Pris (EUR/MW)'
#'aFRR Ned Volym (MW)'
#'Publiceringstidpunkt' 
#'Unnamed: 7']
#Drop last (25) rows
#Sweden
df_SEafrr2020_raw.drop(df_SEafrr2020_raw.tail(25).index, inplace=True) #Dropping last row as it is a sum
df_SEafrr2021_raw.drop(df_SEafrr2021_raw.tail(25).index, inplace=True) #Dropping last row as it is a sum

#Converting time
#Sweden
df_SEafrr2020_raw['Period'] =  df_SEafrr2020_raw['Period'].apply(pd.to_datetime)
df_SEafrr2020_raw['Publiceringstidpunkt'] =  df_SEafrr2020_raw['Publiceringstidpunkt'].apply(pd.to_datetime)
df_SEafrr2021_raw['Period'] =  df_SEafrr2021_raw['Period'].apply(pd.to_datetime)
df_SEafrr2021_raw['Publiceringstidpunkt'] =  df_SEafrr2021_raw['Publiceringstidpunkt'].apply(pd.to_datetime)

#combine the two
df_aFRR_up = pd.concat([df_SEafrr2020_raw, df_SEafrr2021_raw], ignore_index=True, sort=False)


TimeRange_aFRR_up = (df_aFRR_up['Period'] >= Start_date) & (df_aFRR_up['Period']  <= End_date)
df_aFRR_up = df_aFRR_up[TimeRange_aFRR_up]

list_aFRR_up = df_aFRR_up['aFRR Upp Pris (EUR/MW)'].tolist() #Convert from pandas data series to list
c_aFRR_up = dict(zip(np.arange(1,len(list_aFRR_up)+1),list_aFRR_up))



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
    for i in range(0,int(len(PV)/(24*7))):
        Demand[i*24*7-1] = k_d*24*7

    dw = len(PV)/(24*7) - int(len(PV)/(24*7))
    if dw > 0:
        Demand[len(PV)-1] = dw*7*24*k_d


            
Demand = dict(zip(np.arange(1,len(Demand)+1),Demand))



m_H2 = 1 # kg/s
m_CO2 = 1000 # kg/s
T_in = 300 # Kelvin
CP_H2 = 14.307 #kJ/kgK
CP_CO2 = 0.846 #kJ/kgK
γ_H2 = 1.405 
γ_CO2 = 1.289 
η_th = 0.7 #Thermal efficiency (Asssumed to be 70%) 
PR_H2 = 80/1
PR_CO2 = 80/45

P_CON_H2 = (m_H2*T_in*CP_H2*(PR_H2**((γ_H2)/(γ_H2-1))-1))/η_th

P_CON_CO2 = (m_CO2*T_in*CP_H2*(PR_H2**((γ_H2-1)/(γ_H2))-1))/η_th





#240 @ 1000 W/m^2

#n_PV = (300*10**6)/310
#P_pv = n_PV*(df_solar_irr['ALLSKY_SFC_SW_DWN']/800)ø





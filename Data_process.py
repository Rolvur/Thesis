import datetime
import numpy as np
import pandas as pd 
from Settings import Start_date, End_date, Demand_pattern
from Opt_Constants import k_d
#from pyparsing import line



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




##################### FCR #######################

df_FCRA2020_raw = pd.read_csv('2020 - ANONYM_CAPACITY_BIDS_FCR.csv',sep=',')
df_FCRD2020_raw = pd.read_csv('2020 - DEMAND_CAPACITY_FCR.csv',sep=',')
df_FCRR2020_raw = pd.read_csv('2020 - RESULT_CAPACITY_FCR.csv',sep=',')

df_FCRA2021_raw = pd.read_csv('C:2021 - ANONYM_CAPACITY_BIDS_FCR.csv',sep=',')
df_FCRD2021_raw = pd.read_csv('2021 - DEMAND_CAPACITY_FCR.csv',sep=',')
df_FCRR2021_raw = pd.read_csv('2021 - RESULT_CAPACITY_FCR.csv',sep=',')


##################### mFRR #######################

df_DKmFRR_raw = pd.read_csv('MfrrReservesDK1.csv',sep=';', decimal=',')

#Converting to datetime
df_DKmFRR_raw[['HourUTC','HourDK']] =  df_DKmFRR_raw[['HourUTC','HourDK']].apply(pd.to_datetime)


##################### aFRR #######################
#Reading data from csv
# Finland
df_FIafrr2020_raw = pd.read_excel('FI_AFFR_2020.xlsx', index_col=0)
df_FIafrr2021_raw = pd.read_excel('FI_AFFR_2021.xlsx', index_col=0)

#Sweden
df_SEafrr2020_raw = pd.read_csv('SE_AFRR_2020.csv',sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
df_SEafrr2021_raw = pd.read_csv('SE_AFRR_2021.csv',sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .

#Drop last row
#Sweden
df_SEafrr2020_raw.drop(index=df_SEafrr2020_raw.index[-1],axis=0,inplace=True) #Dropping last row as it is a sum
df_SEafrr2021_raw.drop(index=df_SEafrr2021_raw.index[-1],axis=0,inplace=True) #Dropping last row as it is a sum

#Converting time
#Sweden
df_SEafrr2020_raw['Period'] =  df_SEafrr2020_raw['Period'].apply(pd.to_datetime)
df_SEafrr2020_raw['Publiceringstidpunkt'] =  df_SEafrr2020_raw['Publiceringstidpunkt'].apply(pd.to_datetime)
df_SEafrr2021_raw['Period'] =  df_SEafrr2020_raw['Period'].apply(pd.to_datetime)
df_SEafrr2021_raw['Publiceringstidpunkt'] =  df_SEafrr2020_raw['Publiceringstidpunkt'].apply(pd.to_datetime)

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





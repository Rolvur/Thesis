import datetime
from numpy import zeros
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as md
from pyparsing import line

##################### Methanol Demand #######################

Tot_Demand = 32000 #ton methanol annually
Weekly_demand = (32000/365)*7  #Seven days demand


#Input for model
Demand = list(0 for i in range(0,8760))

for i in range(0,len(Demand),7):
    Demand[i] = Weekly_demand



##################### Solar #######################


#Solar production data set
df_solar_prod= pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/PV production data 2019-2020.csv',sep=',')
df_solar_prod['time'] = df_solar_prod['time'].astype(str)

#Converting time to datetime
for i in range(0,len(df_solar_prod['time'])):
    df_solar_prod.iloc[i,0] = datetime.datetime.strptime(df_solar_prod.iloc[i,0], '%Y%m%d:%H%M')
    df_solar_prod.iloc[i,0] = df_solar_prod.iloc[i,0].strftime('%Y-%m-%d - %H:%M')
    
df_solar_prod['time'] = df_solar_prod['time'].apply(pd.to_datetime)


#Input for model
#Using year 2020
TimeRange2020PV = (df_solar_prod['time'] > '2020-01-01') & (df_solar_prod['time']  <= '2020-12-31')
df_solar_prod_2020 = df_solar_prod[TimeRange2020PV]

PV = df_solar_prod_2020['P'].tolist() #Convert from pandas data series to list



#Solar irradiance data set
df_solar_irr= pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/Irradiance data 2020-2021.csv',sep=',')
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
df_DKDA_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/Elspotprices_RAW.csv',sep=';',decimal=',')

#Converting to datetime
df_DKDA_raw[['HourUTC','HourDK']] = df_DKDA_raw[['HourUTC','HourDK']].apply(pd.to_datetime)


#Input for model
#Using year 2020
TimeRange2020DA = (df_DKDA_raw['HourDK'] > '2020-01-01') & (df_DKDA_raw['HourDK']  <= '2020-12-31')
df_DKDA_raw2020 = df_DKDA_raw[TimeRange2020DA]
P_DA = df_DKDA_raw2020['SpotPriceEUR,,'].tolist()






##################### FCR #######################

df_FCRA2020_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/2020 - ANONYM_CAPACITY_BIDS_FCR.csv',sep=',')
df_FCRD2020_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/2020 - DEMAND_CAPACITY_FCR.csv',sep=',')
df_FCRR2020_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/2020 - RESULT_CAPACITY_FCR.csv',sep=',')

df_FCRA2021_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/2021 - ANONYM_CAPACITY_BIDS_FCR.csv',sep=',')
df_FCRD2021_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/2021 - DEMAND_CAPACITY_FCR.csv',sep=',')
df_FCRR2021_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/2021 - RESULT_CAPACITY_FCR.csv',sep=',')


##################### mFRR #######################

df_DKmFRR_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/MfrrReservesDK1.csv',sep=';', decimal=',')

#Converting to datetime
df_DKmFRR_raw[['HourUTC','HourDK']] =  df_DKmFRR_raw[['HourUTC','HourDK']].apply(pd.to_datetime)


##################### aFRR #######################
#Reading data from csv
# Finland
df_FIafrr2020_raw = pd.read_excel('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/FI_AFFR_2020.xlsx', index_col=0)
df_FIafrr2021_raw = pd.read_excel('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/FI_AFFR_2021.xlsx', index_col=0)

#Sweden
df_SEafrr2020_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/SE_AFRR_2020.csv',sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
df_SEafrr2021_raw = pd.read_csv('C:/Users/Rolvur Reinert/Desktop/Data/Python_data/SE_AFRR_2021.csv',sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .

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














#240 @ 1000 W/m^2

#n_PV = (300*10**6)/310
#P_pv = n_PV*(df_solar_irr['ALLSKY_SFC_SW_DWN']/800)ø





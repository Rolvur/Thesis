
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from Data_process import df_DKDA_raw, PV,df_aFRR_scen,df_mFRR_scen,df_FCR_DE_scen,DA_list_scen
from statsmodels.tsa.seasonal import seasonal_decompose


## Data Process ## 

#DA
df_DKDA_raw.index = df_DKDA_raw['HourDK']
df_DKDA_raw
del df_DKDA_raw['HourUTC'],df_DKDA_raw['HourDK'],df_DKDA_raw['PriceArea'],df_DKDA_raw['SpotPriceDKK'] 
sd_DA = seasonal_decompose(df_DKDA_raw, period=24)


fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)


x = df_DKDA_raw.index

ax.plot(x, sd_DA.seasonal, color='g',linestyle = '-', label ='DA')

ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()   

seasonal_decompose(df_DKDA_raw, model='additive').plot()





#FCR
df_FCR = pd.DataFrame() 
df_FCR['Price'] = df_FCR_DE_scen['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]']
df_FCR.index = df_FCR_DE_scen['DATE_FROM']
sd_FCR = seasonal_decompose(df_FCR, period=24)

fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)


x = df_FCR.index

ax.plot(x, sd_FCR.seasonal, color='g',linestyle = '-', label ='FCR')

ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show() 





#aFRR up 
df_aFRR_up = df_aFRR_scen
df_aFRR_up.index = df_aFRR_scen['Period']

del df_aFRR_up['Unnamed: 0'],df_aFRR_up['Period'],df_aFRR_up['Elområde'],df_aFRR_up['aFRR Upp Volym (MW)'],df_aFRR_up['aFRR Ned Pris (EUR/MW)'],df_aFRR_up['aFRR Ned Volym (MW)'],df_aFRR_up['Publiceringstidpunkt'],df_aFRR_up['Unnamed: 7']

sd_aFRR_up = seasonal_decompose(df_aFRR_up, period=24) ## Periods is the nub,er of cycles that is expected to be in the data


fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)


x = df_aFRR_up.index

ax.plot(x, sd_aFRR_up.seasonal, color='g',linestyle = '-', label ='aFRR_up')

ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show() 




#aFRR down 
df_aFRR_dwn = df_aFRR_scen
df_aFRR_dwn.index = df_aFRR_scen['Period']

del df_aFRR_dwn['Unnamed: 0'],df_aFRR_dwn['Period'],df_aFRR_dwn['Elområde'],df_aFRR_dwn['aFRR Upp Volym (MW)'],df_aFRR_dwn['aFRR Upp Pris (EUR/MW)'],df_aFRR_dwn['aFRR Ned Volym (MW)'],df_aFRR_dwn['Publiceringstidpunkt'],df_aFRR_dwn['Unnamed: 7']

df_aFRR_dwn = seasonal_decompose(df_aFRR_dwn, period=24) ## Periods is the nub,er of cycles that is expected to be in the data

fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)


x = df_aFRR_dwn.index

ax.plot(x, sd_aFRR_up.seasonal, color='g',linestyle = '-', label ='aFRR_up')

ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show() 

#mFRR 
df_mFRR = pd.DataFrame()
df_mFRR.index = df_mFRR_scen['HourDK']
df_mFRR['Price'] = df_mFRR_scen['mFRR_UpPriceEUR']
sd_mFRR = seasonal_decompose(df_mFRR, period=24) ## Periods is the nub,er of cycles that is expected to be in the data




fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)


x = df_mFRR.index

ax.plot(x, sd_mFRR.seasonal, color='g',linestyle = '-', label ='aFRR_up')

ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show() 






#seasonal = sd.collect()['SEASONAL']


DA_season = sd.seasonal



fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)


x = df_resultsM1_2020['HourDK']

ax1.plot(x, df_resultsM1_2020['DA'], color='g',linestyle = '-', label ='DA')
ax2.plot(x, df_resultsM1_2020['P_PV'], color='r',linestyle = '-', label ='PV')
ax2.plot(x, df_resultsM1_2020['P_PEM'], color='b',linestyle = '-', label ='PEM')
ax2.plot(x, df_resultsM1_2020['P_grid'], color='purple',linestyle = '-', label ='Grid')


ax1.set_ylabel('€')
ax2.set_ylabel('MW')
ax1.legend(loc='best')
ax2.legend(loc='best')
#ax1.set_ylim[300000,-600000]


ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()   





combine_seasonal_cols(df, sd) # custom helper function















dta = sm.datasets.sunspots.load_pandas().data

















dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]


del df_DKDA_raw['HourUTC'],df_DKDA_raw['HourDK'], df_DKDA_raw['PriceArea'], df_DKDA_raw['SpotPriceDKK']

df_DKDA_raw

sm.graphics.tsa.plot_acf(df_DKDA_raw.values.squeeze(), lags=40)
plt.show()




import numpy as np 
np.sqrt((5**2)+(7**2))




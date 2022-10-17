from cProfile import label
import sys, os
sys.path.append('C:/Users/Rolvur Reinert/Desktop/Data/Python_data')
from Data_process import df_solar_prod, df_DKDA_raw, df_FIafrr2020_raw, df_SEafrr2020_raw
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from IPython.display import display

####################### Plot ####################### 
#Selecting time range for DataFrame
TimeRange2019 = (df_solar_prod['time'] > '2019-01-01') & (df_solar_prod['time'] <= '2019-12-31')
TimeRange2020 = (df_solar_prod['time'] > '2020-01-01') & (df_solar_prod['time'] <= '2020-12-31')

df2019 = df_solar_prod.loc[TimeRange2019]
df2020 = df_solar_prod.loc[TimeRange2020]

#Grupe by mean so getting average of a periode(e.g. month)
x2019 = df2019.groupby(pd.PeriodIndex(df2019['time'], freq="M"))['P'].mean().reset_index()
x2020 = df2020.groupby(pd.PeriodIndex(df2020['time'], freq="M"))['P'].mean().reset_index()


x2019['time'] = x2019['time'].astype(str)
x2020['time'] = x2020['time'].apply(pd.to_datetime) 

x2019['time'] = x2019['time'].astype(str)
x2020['time'] = x2020['time'].apply(pd.to_datetime) 


## Calculate correlations EXAMPLE 
df_FIafrr2020_raw
df_SEafrr2020_raw
TimeRange2020_FI = (df_FIafrr2020_raw['Start time UTC+02:00'] > '2020-01-01') & (df_FIafrr2020_raw['Start time UTC+02:00']  <= '2020-12-31')

TimeRange2020_SE = (df_SEafrr2020_raw['Period'] > '2020-01-01') & (df_SEafrr2020_raw['Period']  <= '2020-12-31')

df_FI2020 = df_FIafrr2020_raw[TimeRange2020_FI]
df_SE2020 = df_SEafrr2020_raw[TimeRange2020_SE]

df = pd.DataFrame() #Creating empty dataframe


df['Finland Up'] = df_FI2020['Automatic Frequency Restoration Reserve, price, up']
df = df.reset_index(drop=True)
df['Sweden Up'] = df_SE2020['aFRR Upp Pris (EUR/MW)']
matrix = df.corr() 

print(matrix)
sns.heatmap(matrix)

matrix.style.background_gradient(cmap='coolwarm')
plt.show()


#Plot (Gruped bar) 

x_axis = np.arange(12) #Number of  stamps on x axis (Months)
x_label = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August', 'Sept','Okt', 'Nov', 'Dec']
width = 0.3

fig, ax = plt.subplots()

fig1 = ax.bar(x_axis-width,x2019['P'],width = width, label ='Avg Prod 2019') 
fig2 = ax.bar(x_axis,x2020['P'],width = width ,label ='Avg Prod 2020')

ax.set_ylabel('[MW/h]')
ax.set_title('PV Production')
plt.xticks(x_axis,x_label) 
plt.xticks(rotation=25)
plt.legend()
plt.show()




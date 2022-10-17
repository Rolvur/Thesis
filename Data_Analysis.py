import sys, os
sys.path.append('C:/Users/Rolvur Reinert/Desktop/Data/Python_data')
from Data_process import df_solar_prod
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as md

####################### Plot ####################### 
#Selecting time range for DataFrame
TimeRange2020 = (df_solar_prod['time'] > '2020-01-01') & (df_solar_prod['time'] <= '2020-01-31')
TimeRange2021 = (df_solar_prod['time'] > '2021-01-01') & (df_solar_prod['time'] <= '2021-01-31')

df2020 = df_solar_prod.loc[TimeRange2020]
df2021 = df_solar_prod.loc[TimeRange2021]

#Grupe by mean so getting average of a periode(e.g. month)
x2020 = df2020.groupby(pd.PeriodIndex(df2020['time'], freq="M"))['P'].mean().reset_index()
x2021 = df2021.groupby(pd.PeriodIndex(df2021['time'], freq="M"))['P'].mean().reset_index()


x2020['time'] = x2020['time'].astype(str)
x2020['time'] = x2020['time'].apply(pd.to_datetime) 

x2021['time'] = x2021['time'].astype(str)
x2021['time'] = x2021['time'].apply(pd.to_datetime) 


x_axis = x2021['time']


plt.bar(x_axis, x2020['P'], color ='maroon', width = 0.5,
        edgecolor ='grey', label ='Avg Prod 2020')
plt.bar(x_axis, x2021['P'], color ='g', width = 0.5,
        edgecolor ='grey', label ='Avg Prod 2020')






plt.legend()
plt.show()


plt.subplots_adjust(bottom=0.2)

#Plot general
plt.title('PV Production')

#Axis config
plt.xlabel("Time")
plt.ylabel("MW")

plt.tight_layout()
plt.xticks(rotation=25)
ax=plt.gca()
xfmt = md.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(xfmt)

plt.bar(x_axis,y_axis, color = 'maroon', width=5)
plt.show()




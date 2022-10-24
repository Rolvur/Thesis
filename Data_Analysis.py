from cProfile import label
import sys, os
from tkinter import Y
from turtle import width
from Data_process import DA
#sys.path.append('C:/Users/Rolvur Reinert/Desktop/Data/Python_data')
#from Data_process import df_solar_prod, df_DKDA_raw, df_FIafrr2020_raw, df_SEafrr2020_raw
from Opt_Model_V1 import df_results

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
#from matplotlib import pyplot as plt
#import seaborn as sns
#from IPython.display import display



####################### Plot Model Results #######################


#Plot style
plt.style.use() ## 'fivethirtyeight' 'seaborn' etc



## Subplot example
def SubPlot1(Results):


    #Defingin x axes
    x = Results.index


    ## Subplot example
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True) 
    #nrows and ncols is how to show the plots in the frame. Sharex=True(both plots share xakis)

    #ax1 & ax2 two subplots in fig
    ax1.plot(x, Results['DA'], color='b',linestyle = '--', label ='Day Ahead Price')
    ax2.plot(x, Results['P_PEM'],color='red', label ='PEM Production')
    #ax2.plot(x, df_results['P_PV'], label ='PV Production')

    ax1.legend()
    ax1.set_title('MONS')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('EUR/MWh')


    ax2.legend(loc='upper left')
    ax2.set_ylim([0, 70])
    #ax2.set_title('TONS')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('MW')
    ax2.tick_params(labelrotation=45)

    ax3 = ax2.twinx()
    ax3.plot(x, Results['P_PV'],color='g', label ='PV Production')

    ax3.legend()
    ax3.set_ylim([0, 300])
    #ax3.set_title('TONS')
    #ax3.set_xlabel('Time')
    ax3.set_ylabel('MW')


    plt.tight_layout()
    ax1.grid()
    plt.show()

SubPlot1(df_results)

##Two line plot
x = df_results.index

fig, ax = plt.subplots()

ax.plot(x, df_results['P_PEM'], color='b',linestyle = '--', label ='PEM')

ax.plot(x, df_results['P_sRaw'], color='b',linestyle = '--', label ='Raw Storage')







## Multiple bar and line plot together


x_indexes = np.arange(len(df_results.index))
width_x = 1

x = df_results.index

fig, ax = plt.subplots()
ax.bar(x-width_x, df_results['P_PEM'],width = width_x, align = 'center')
ax.bar(x, df_results['P_PV'], width = width_x , align = 'center')
#ax.bar(df.index, df['C'],width = 5, align = 'center')
#ax.bar(df.index, df['D'], width = 5 , align = 'center')
ax.xaxis_date()
ax.get_xaxis().set_major_locator(mdates.MonthLocator())
ax.get_xaxis().set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
fig.autofmt_xdate()
plt.show()



















#Plot style
plt.style.use('fivethirtyeight') ## 'fivethirtyeight' 'seaborn' etc

#Defingin x axes
x = df_results.index

plt.bar(x_indexes-width,df_results['P_PEM'],width=width, color='r', label = 'PEM') #-width  = shift bar to left
plt.bar(x_indexes,df_results['P_PV'],width=width, color='b', label = 'PV')


#plt.xticks(ticks=x_indexes, labels=x)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()


#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.legend()
plt.tight_layout()
plt.show()













plt.plot(x, df_results.DA, label = "line 1", linestyle="-")
plt.plot(x, df_results.DA, label = "line 2", linestyle="--")
plt.plot(x, df_results.P_sRaw, label = "curve 1")
plt.xticks(rotation=20)
plt.legend()
plt.show()

df_results[['P_sRaw','P_sPu']].plot(kind='bar', width = width)
df_results['DA'].plot(secondary_y=True)

ax = plt.gca()
plt.xlim([-width, len(df_results['P_sRaw'])-width])
ax.set_xticklabels(x)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.show()















labels = df_results.columns #Getting column names


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(df_results.index - width/2, df_results['P_sRaw'], width, label='P_sRaw')
#rects1 = ax.bar(x - width/2, df_results['P_sRaw'], width, label='P_sRaw')
rects2 = ax.bar(x + width/2, df_results['P_sPu'], width, label='P_sPu')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MW')
ax.set_title('Time')
#ax.set_xticks(df_results.index)
ax.legend()

#Shows value on top of bars 
ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()







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




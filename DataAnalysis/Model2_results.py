from turtle import width
from Opt_Constants import *
import scipy
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as md
from statistics import mean
from pathlib import Path

file_to_open1 = Path("Result_files/") / "Model2_2020.xlsx"
file_to_open2 = Path("Result_files/") / "Model2_2021.xlsx"
df_resultsM2_2020 = pd.read_excel(file_to_open1)
df_resultsM2_2021 = pd.read_excel(file_to_open2)




######### Stacked BAR Chart ########### Hourly reserved capacities 
def StackBar(Results,P_pem_cap,P_pem_min):
    x = Results['HourDK']
    #d = scipy.zeros(len(x))

    #See FCR 4 hours blocks 2020 11 06 00:00

    FCR_up = Results['FCR "up"']
    aFRR_up = Results['aFRR_up']
    mFRR_up = Results['mFRR_up']
    aFRR_down = Results['aFRR_down']
    FCR_down = Results['FCR "down"']


    sum_cols = ['FCR "up"','aFRR_up','mFRR_up','aFRR_down','FCR "down"']
    sum_act = Results[sum_cols].sum(axis=1)
    PEM_deviate = P_pem_cap - sum_act - P_pem_min


    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios': [1, 1]})

    
    ax1.plot(x, Results['cFCR'], label= 'FCR', color='darkorange')
    ax1.plot(x, Results['caFRRup'], label= 'aFRR up', color='firebrick')
    ax1.plot(x, Results['caFRRdown'], label= 'aFRR down', color='maroon')
    ax1.plot(x, Results['cmFRRup'], label= 'mFRR up' , color='goldenrod')
    ax1.plot(x, Results['DA'], label= 'Day-ahead' , color='navy')

    ax2.bar(x, P_pem_min, color='lightsteelblue',linestyle = 'solid', label ='PEM min',width=0.02)
    ax2.bar(x, FCR_up, bottom = P_pem_min,  color='darkorange',linestyle = 'solid', label ='FCR up',width=0.02)
    ax2.bar(x, aFRR_up, bottom = P_pem_min+FCR_up,  color='firebrick',linestyle = 'solid', label ='aFRR up',width=0.02)
    ax2.bar(x, mFRR_up, bottom = P_pem_min+FCR_up+aFRR_up,  color='goldenrod',linestyle = 'solid', label ='mFRR up',width=0.02)
    ax2.bar(x, PEM_deviate, bottom = P_pem_min+FCR_up+aFRR_up+mFRR_up,  color='steelblue',linestyle = 'solid', label ='Free',width=0.02)
    ax2.bar(x, aFRR_down, bottom = P_pem_min+FCR_up+aFRR_up+mFRR_up+PEM_deviate,  color='maroon',linestyle = 'solid', label ='aFRR down',width=0.02)
    ax2.bar(x, FCR_down, bottom = P_pem_min+FCR_up+aFRR_up+mFRR_up+PEM_deviate+aFRR_down,  color='darkorange',linestyle = 'solid', label ='FCR down',width=0.02)
    ax2.plot(x, Results['P_PEM'], color='green',linestyle = '-', label ='PEM_Setpoint', linewidth=1.5)


    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.bar(x, df_results['FCR "up"'], color='blue',linestyle = 'solid', label ='FCR "up"',width=0.02)

    ax1.tick_params(axis='x', rotation=45)
    #ax1.set_ylim([0, 57])
    ax1.set_ylabel('€/MW')
    ax1.set_title('Reserve Prices')
    
    ax2.set_title('Reserve Capacity')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0, 57])
    ax2.set_ylabel('MW')


    #ax.title('')
    plt.tight_layout()
    plt.show()
StackBar(df_resultsM2_2020,P_pem_cap,P_pem_min)


#output mio EUR
def CostRevProf(df):

    consumer = df.loc[df['z_grid'] == 1]
    producer = df.loc[df['z_grid'] == 0]

    Cost_DA = sum((CT + consumer['DA'])*consumer['P_grid'])/10**6

    Rev_DA = sum((producer['DA']-PT)* (-producer['P_grid']))/10**6
    Rev_FCR_up = sum(df['FCR "up"'] * df['cFCR'])/10**6
    Rev_FCR_down = sum(df['FCR "down"'] * df['cFCR'])/10**6
    
    Rev_aFRR_up = sum(df['aFRR_up'] * df['caFRRup'])/10**6
    Rev_aFRR_down = sum(df['aFRR_down'] * df['caFRRdown'])/10**6

    Rev_mFRR_up = sum(df['mFRR_up'] * df['cmFRRup'])/10**6

    Profit = Rev_DA + Rev_FCR_up + Rev_aFRR_up + Rev_aFRR_down + Rev_mFRR_up- Cost_DA

    return Rev_DA, Rev_FCR_up, Rev_FCR_down, Rev_aFRR_up, Rev_aFRR_down, Rev_mFRR_up, Cost_DA, Profit



## DA Cost plot w. tarrif ## 
def CostRev(df_resultsM2_2020,df_resultsM2_2021):
    Rev_DA2020, Rev_FCR_up2020, Rev_FCR_down2020, Rev_aFRR_up2020, Rev_aFRR_down2020, Rev_mFRR_up2020, Cost_DA2020, Profit2020 = CostRevProf(df_resultsM2_2020)
    Rev_DA2021, Rev_FCR_up2021, Rev_FCR_down2021, Rev_aFRR_up2021, Rev_aFRR_down2021, Rev_mFRR_up2021, Cost_DA2021, Profit2021 = CostRevProf(df_resultsM2_2021)


    DA_Rev = np.array([Rev_DA2020,Rev_DA2021])
    DA_Cost = np.array([-Cost_DA2020,-Cost_DA2021])
    FCR_Rev = np.array([Rev_FCR_up2020,Rev_FCR_up2021])
    aFRR_up = np.array([Rev_aFRR_up2020, Rev_aFRR_up2021])
    aFRR_down = np.array([Rev_aFRR_down2020, Rev_aFRR_down2021])
    mFRR = np.array([Rev_mFRR_up2020,Rev_mFRR_up2021])

    var1 = DA_Rev + FCR_Rev
    var2 = var1 + aFRR_up 
    var3 = aFRR_down + var2

    x = ['2020','2021']
    fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharex=True,gridspec_kw={'width_ratios': [1.3, 1]})

    

    ax1.bar(x, mFRR, bottom=var3, color='goldenrod',linestyle = 'solid', label ='mFRR Up',width=0.75)
    ax1.bar(x, aFRR_down, bottom=var2, color='navy',linestyle = 'solid', label ='aFRR Down',width=0.75)
    ax1.bar(x, aFRR_up, bottom=var1, color='royalblue',linestyle = 'solid', label ='aFRR Up',width=0.75)
    ax1.bar(x, FCR_Rev,bottom= DA_Rev ,color='darkorange',linestyle = 'solid', label ='FCR',width=0.75)
    ax1.bar(x, DA_Rev, color='darkgreen',linestyle = 'solid', label ='DA_Rev',width=0.75)

    ax1.bar(x , DA_Cost, color='maroon',linestyle = 'solid', label ='DA_Cost',width=0.75)
    

    ax1.plot(x, [Profit2020,Profit2021], color='navy', label='Profit',linestyle='dashed', marker='o')

    ax2.bar(x, [df_resultsM2_2020['P_grid'].sum(),df_resultsM2_2021['P_grid'].sum()], color ='steelblue', label = 'Net import' )
    
    ax1.set_ylabel('mio €')
    ax1.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
    ax1.set_ylim([-20,30])

    ax2.set_ylabel('MWh')
    ax2.legend(loc='upper left')
    ax2.set_ylim([0,70000])

    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()  



CostRev(df_resultsM2_2020,df_resultsM2_2021)









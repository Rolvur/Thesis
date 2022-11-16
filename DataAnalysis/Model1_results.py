
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



file_to_open1 = Path("Result_files/") / "Model1_All2020.xlsx"
file_to_open2 = Path("Result_files/") / "Model1_All2021.xlsx"
df_resultsM1_2020 = pd.read_excel(file_to_open1)
df_resultsM1_2021 = pd.read_excel(file_to_open2)


DA_wPT = np.zeros(len(df_resultsM1_2020))
DA_wCT = np.zeros(len(df_resultsM1_2020))
for i in range(0,len(df_resultsM1_2020)):

    if df_resultsM1_2020['DA'][i] < 0:
        DA_wPT[i] = df_resultsM1_2020['DA'][i] - PT
        DA_wCT[i] = df_resultsM1_2020['DA'][i] + CT
    
    if df_resultsM1_2020['DA'][i] >= 0:
        DA_wPT[i] = df_resultsM1_2020['DA'][i] - PT
        DA_wCT[i] = df_resultsM1_2020['DA'][i] + CT    


df_resultsM1_2020['DA incl PT'] = DA_wPT
df_resultsM1_2020['DA incl CT'] = DA_wCT


#Subplot of Storage level(Shaded region), stack and line
def Model1StorageFlow(Results):

    ##Two line(shaded under line) plot also subplots
    x = Results['HourDK']
    d = scipy.zeros(len(x))

    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)

    #1st Subplot 
    ax4 = ax1.twinx()
    ax1.fill_between(x, Results['Pure Storage'], where=Results['Pure Storage']>=d, interpolate=True, color='lightblue',label='Pure Storage')
    ax1.bar(x, Results['Demand'], color='red',linestyle = 'solid', label ='Demand',width=0.05)

    #ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylabel('kg')
    ax1.set_ylim([0, 650000])
    ax1.legend(loc='upper left')
    ax1.set_title('Pure Methanol')

    ax4.plot(x, Results['Pure_In'], color='midnightblue',linestyle = '--', label ='Pure In')
    ax4.set_ylabel('kg/s')
    ax4.set_ylim([0, 5000])
    ax4.legend(loc='upper right')

    #2nd Subplot
    ax3 = ax2.twinx()

    ax2.fill_between(x, Results['Raw Storage'], where=Results['Raw Storage']>=d, interpolate=True, color='lightgrey',label='Raw Storage')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel('kg')
    ax2.set_ylim([0, 110000])
    ax2.legend(loc='upper left')


    ax3.plot(x, Results['Raw_In'], color='forestgreen',linestyle = 'solid', label ='Raw In')
    ax3.plot(x, Results['Raw Out'], color='midnightblue',linestyle = '--', label ='Raw Out')
    ax3.set_title('Raw Methanol')
    ax3.set_ylabel('kg/s')
    ax3.set_ylim([0, 11000])
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
Model1StorageFlow(df_resultsM1_2020)


#output mio EUR
def DACostRevProf(df):

    consumer = df.loc[df['zT'] == 0]
    producer = df.loc[df['zT'] == 1]

    Cost_DA = sum((CT + consumer['DA'])*consumer['P_grid'])/10**6

    Rev_DA = sum((producer['DA']-PT)* (-producer['P_grid']))/10**6

    Profit = Rev_DA - Cost_DA

    return Cost_DA,Rev_DA,Profit


## DA Cost plot w. tarrif ## 
def DAcostRev():
    DA_Cost2020,DA_Rev_2020,DA_Profit_2020 = DACostRevProf(df_resultsM1_2020)
    DA_Cost2021,DA_Rev_2021,DA_Profit_2021 = DACostRevProf(df_resultsM1_2021)


    x = ['2020','2021']
    fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharex=True)

    ax1.bar(x, [DA_Rev_2020,DA_Rev_2021], color='darkgreen',linestyle = 'solid', label ='DA Revenue',width=0.75)

    ax1.bar(x , [-DA_Cost2020,-DA_Cost2021], color='maroon',linestyle = '-', label ='DA Cost',width=0.75)
    ax1.plot(x, [DA_Profit_2020,DA_Profit_2021], color='navy', label='Profit',linestyle='dashed', marker='o')

    ax2.bar(x, [df_resultsM1_2020['P_grid'].sum(),df_resultsM1_2021['P_grid'].sum()], color ='steelblue', label = 'Net Power import' )
    
    ax1.set_ylabel('mio €')
    ax1.legend(loc='upper left')


    ax2.set_ylabel('MWh')
    ax2.legend(loc='upper left')


    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()  


## DA price with tariff vs Pem, grid and PV
def DAtariffvsPower(df_resultsM1_2020):
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)


    x = df_resultsM1_2020['HourDK']

    ax1.plot(x, df_resultsM1_2020['DA'], color='navy',linestyle = '-', label ='DA excl. tariff')
    ax1.plot(x, df_resultsM1_2020['DA incl PT'], color='g',linestyle = '-', label ='DA incl. PT')
    ax1.plot(x, df_resultsM1_2020['DA incl CT'], color='r',linestyle = '-', label ='DA incl. CT')

    ax2.plot(x, df_resultsM1_2020['P_PV'], color='r',linestyle = '-', label ='PV')
    ax2.plot(x, df_resultsM1_2020['P_PEM'], color='b',linestyle = '-', label ='PEM')
    ax2.plot(x, df_resultsM1_2020['P_grid'], color='purple',linestyle = '-', label ='Grid')

    ax1.axhline(y = 0, color = 'gray', linestyle = '--')

    ax1.set_ylabel('€/MWh')
    ax2.set_ylabel('MW')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    #ax1.set_ylim[300000,-600000]


    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()   
DAtariffvsPower(df_resultsM1_2020)

## Power consumption
def PowerCon(df_resultsM1_2020,df_resultsM1_2021):


    df_results = pd.concat([df_resultsM1_2020 , df_resultsM1_2021],ignore_index=True)

    P_sys = np.zeros((len(df_results),2))   #[PV,Grid]

    for i in range(0,len(df_results)):

        if df_results['P_grid'][i] < 0: #Exporting power
            P_sys[i,0] = df_results['P_PV'][i] + df_results['P_grid'][i]


        if df_results['P_grid'][i] >= 0: #Imporitn power 
            P_sys[i,0] = df_results['P_PV'][i]
            P_sys[i,1] = df_results['P_grid'][i]



    df_Psys = pd.DataFrame({'PV': P_sys[:,0], 'Grid': P_sys[:,1], 'P_sysTot': P_sys[:,0]+P_sys[:,1], 'P_PEM': df_results['P_PEM'].tolist()}, index=df_results['HourDK'])

    #demand_avg  = df_resultsM1_2020.groupby(pd.PeriodIndex(df_resultsM1_2020['HourDK'], freq="M"))['Demand'].mean()


    df_Psys_avg = df_Psys.groupby(pd.PeriodIndex(df_Psys.index, freq="M"))['PV','Grid','P_sysTot','P_PEM'].mean()

    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)

    x = df_Psys_avg.index

    x = x.astype(str)

    ax.bar(x, df_Psys_avg['Grid'], label = 'Grid', color= 'teal') 
    ax.bar(x, df_Psys_avg['PV'] ,bottom = df_Psys_avg['Grid'], label='PV', color= 'olivedrab') 

    ax.plot(x, df_Psys_avg['P_PEM'], color='red', label='PEM',linestyle='solid', marker='o')

    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('MW')
    plt.legend()
    loc = plticker.MultipleLocator(base=2.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    #ax.set_ylim([0, 57])
    #ax.title('')
    plt.tight_layout()
    plt.show()

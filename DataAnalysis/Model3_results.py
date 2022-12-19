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
import os, sys
import seaborn as sns
from IPython.display import display

#Function for getting all SolX results files in as a combined DataFrame
def importSolX():

    FileNames = []

    for file in os.listdir("Result_files"):
        if file.startswith("V3_SolX_20"):
            FileNames.append(file)


    DataFiles = []

    for filename in FileNames:

        file_to_open = Path("Result_files/") / filename
        df_file = pd.read_excel(file_to_open)
        DataFiles.append(df_file)


    df_DataFiles = pd.concat(DataFiles)

    #df_DataFiles.to_excel('MONS.xlsx')

    return df_DataFiles

df_SolX = importSolX()

# Import result data from V2  
def importV2():

    FileNames = []

    for file in os.listdir("Result_files"):
        if file.startswith("V2_20"):
            FileNames.append(file)


    DataFiles = []

    for filename in FileNames:

        file_to_open = Path("Result_files/") / filename
        df_file = pd.read_excel(file_to_open)
        DataFiles.append(df_file)


    df_DataFiles = pd.concat(DataFiles)

    #df_DataFiles.to_excel('MONS.xlsx')

    return df_DataFiles

df_V2 = importV2()

CapNames = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
PriceNamesV2 = ['c_FCR','c_aFRR_up','c_aFRR_down','c_mFRRup']
PriceNamesSolX = ['c_FCR','c_aFRR_up','c_aFRRdown','c_mFRR_up']
Markets =['FCR','aFRR Up','aFRR Down','mFRR']



TotRevV2 = []
TotRevSolX = []



for t in range(0,len(df_SolX)):
    RevV2 = RevV2 + df_SolX['P_export'].iloc[t]*df_SolX[PriceNamesV2[i]].iloc[t]
     

TotRevV2.append(RevV2)
TotRevSolX.append(RevSolX(y))






## Reading Rep Weeks 
file_to_open = Path("Result_files/") / 'RepWeeks.xlsx'
df_RepWeeks = pd.read_excel(file_to_open)

# Plot start & end date 
plot_start = '2021-01-01 00:00'
plot_end = '2021-12-31 23:59'

TimeRangeSolX = (df_SolX['HourDK'] >= plot_start) & (df_SolX['HourDK']  <= plot_end)
TimeRangeV2 = (df_V2['HourDK'] >= plot_start) & (df_V2['HourDK']  <= plot_end)

df_SolX = df_SolX[TimeRangeSolX]
df_V2 = df_V2[TimeRangeV2]

## Defingin Functions ## 
def PieChartCap(df_results):


    Names = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']
    TotVol = []
    # Sum the values
    for i in Names: 
        TotVol.append(sum(df_results[i]))


    colors = ['#008fd5','#fc4f30','#e5ae37','#6d904f']
    explode = [0,0.1,0,0]
    plt.pie(TotVol,labels=Markets,colors=colors,explode=explode,shadow=True,startangle=0,wedgeprops={'edgecolor':'black'})

    plt.title('Total accepted bid capacity')

    plt.tight_layout()
    plt.show()

def BarHCap(df_SolX,df_V2):

    Names = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']
    TotVolV2 = []
    TotVolSolX = []

    # Sum the values
    for i in Names: 
        TotVolV2.append(sum(df_V2[i]))
        TotVolSolX.append(sum(df_SolX[i]))

    df = pd.DataFrame({'Potential': TotVolV2,
                        'Realized': TotVolSolX}, index=Markets)

    df.plot.barh(color=['#008fd5','#fc4f30'])
    plt.legend(loc='lower right')
    plt.xlabel('MW')
    plt.tight_layout()
    plt.show()

def BarHRev(df_V2,df_SolX):

    # Calculate Revenue # 

    CapNames = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    PriceNamesV2 = ['c_FCR','c_aFRR_up','c_aFRR_down','c_mFRRup']
    PriceNamesSolX = ['c_FCR','c_aFRR_up','c_aFRRdown','c_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']

    TotRevV2 = []
    TotRevSolX = []


    for i in range(0,len(Markets)):
        RevV2 = 0
        RevSolX = 0
        for t in range(0,len(df_V2)):
            RevV2 = RevV2 + df_V2[CapNames[i]].iloc[t]*df_V2[PriceNamesV2[i]].iloc[t]
            RevSolX = RevSolX + df_SolX[CapNames[i]].iloc[t]*df_SolX[PriceNamesSolX[i]].iloc[t] 
        
        TotRevV2.append(RevV2)
        TotRevSolX.append(RevSolX)



    df = pd.DataFrame({'Potential': [TotRevV2[i]/1000000 for i in range(0,len(TotRevV2))],
                        'Realized': [TotRevSolX[i]/1000000 for i in range(0,len(TotRevSolX))]}, index=Markets)

    df.plot.barh(color=['#008fd5','#fc4f30'])
    plt.legend(loc='lower right')
    plt.xlabel('mEUR')
    plt.tight_layout()
    plt.show()

def AvgClPrice(df_SolX):

    CapNames = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    PriceNamesSolX = ['c_FCR','c_aFRR_up','c_aFRRdown','c_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']


    cP_reserves = []

    for i in range(0,len(Markets)):
        
        cP = 0
        count = 0

        for t in range(0,len(df_SolX)):
            if df_SolX[CapNames[i]].iloc[t] > 0: 

                cP = cP + df_SolX[PriceNamesSolX[i]].iloc[t]
                count +=1

        cP_reserves.append(cP/count)


    #           FCR,  aFRR_up , aFRR_down , mFRR
    color = ['darkorange','royalblue','navy','goldenrod']


    df = pd.DataFrame({'Avg Price': [cP_reserves[i] for i in range(0,len(cP_reserves))]}, index=Markets)

    df.plot.bar(color = color[2])
    plt.legend(loc='upper right')
    plt.xlabel('EUR')
    plt.tight_layout()
    plt.show()
    #If want to specify different colors for each market use this ## 
    #plt.bar(df.index, df['Avg Price'], color=color)
    #plt.show()


## Change manually to switch between years
def RepWeekToYear(df_SolX):

    weeks = 168
    Tot_obs = [] 


    while weeks <= len(df_SolX):
        
        w = df_SolX.iloc[(weeks-168):weeks,:]
        start = w.iloc[0,0]

        for i in range(0,len(df_RepWeeks.iloc[:,0])):
            
            if start == df_RepWeeks.iloc[i,3]:
                obs = w['P_PEM'].value_counts()
                rep_obs = obs * (365/7)*df_RepWeeks.iloc[i,4]
                Tot_obs.append(rep_obs)

        
        weeks += 168


    df_obs = pd.DataFrame()
    for i in range(0,len(Tot_obs)):
        df_obs = pd.concat([df_obs, pd.DataFrame(Tot_obs[i])])




    df_obs = df_obs.sort_index(axis=0,ascending=False)
    
    return df_obs



### Pie Chart ###  
PieChartCap(df_SolX)


### Horizontal Bar Chart ### 
BarHCap(df_SolX,df_V2)


## Horizontal revenue plot ## 
BarHRev(df_V2,df_SolX)


## Average clearing price of accepted bids (SolX)## 
AvgClPrice(df_SolX)


## P_PEM setpoint distribution ## 

#Converting Rep weeks to whole year

##PLOT
df_obs2020 = RepWeekToYear(df_SolX)
df_obs2021 = RepWeekToYear(df_SolX)

df_obsTot = [df_obs2020.index,df_obs2021.index]



df_obs2020_Int = df_obs2020

df_obs2020_Int.index = df_obs2020_Int.index.astype(int)

df_obs2021_Int = df_obs2021

df_obs2021_Int.index = df_obs2021_Int.index.astype(int)


# matplotlib histogram of observations
plt.bar(df_obs2020_Int.index-0.25,df_obs2020['P_PEM'] ,label='PEM Setpoint 2020',width=0.45, color = '#a83232')
plt.bar(df_obs2021_Int.index+0.25,df_obs2021['P_PEM'] ,label='PEM Setpoint 2021',width=0.45, color ='#3255a8')
plt.xlabel('MW')
plt.legend()
plt.ylabel('Hours')
plt.show()




## 








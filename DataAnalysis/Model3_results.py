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


# Plot start & end date 
plot_start = '2020-08-31 00:00'
plot_end = '2021-12-31 23:59'

TimeRangeSolX = (df_SolX['HourDK'] >= plot_start) & (df_SolX['HourDK']  <= plot_end)
TimeRangeV2 = (df_V2['HourDK'] >= plot_start) & (df_V2['HourDK']  <= plot_end)

df_SolX = df_SolX[TimeRangeSolX]
df_V2 = df_V2[TimeRangeV2]

### Pie Chart ###  
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
PieChartCap(df_SolX)


### Horizontal Bar Chart ### 

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

BarHCap(df_SolX,df_V2)



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






















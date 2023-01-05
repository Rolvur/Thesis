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

#Function for getting all SolX single results files in as a combined DataFrame
def importSolXSingle():

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

df_SolX_single = importSolXSingle()

def importSolXCombined():

    FileNames = []

    for file in os.listdir("Result_files"):
        if file.startswith("V3_SolX_combined_20"):
            FileNames.append(file)


    DataFiles = []

    for filename in FileNames:

        file_to_open = Path("Result_files/") / filename
        df_file = pd.read_excel(file_to_open)
        DataFiles.append(df_file)


    df_DataFiles = pd.concat(DataFiles)

    #df_DataFiles.to_excel('MONS.xlsx')

    return df_DataFiles

df_SolX_combined = importSolXCombined()

# Import result data from V2  
def importV2_weeks():

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

df_V2_weeks = importV2_weeks()


## Importin rep week used for scaling  
file_to_open = Path("Result_files/") / 'RepWeeks.xlsx'
df_RepWeeks = pd.read_excel(file_to_open)

## Sotring the Rep week data so it matches the other data 
df_RepWeeks2020 = df_RepWeeks.iloc[:,1:3]
df_RepWeeks2020['Rep Weeks for 2020'] = pd.to_datetime(df_RepWeeks2020['Rep Weeks for 2020'])
df_RepWeeks2020 = df_RepWeeks2020.sort_values(by='Rep Weeks for 2020')


df_RepWeeks2021 = df_RepWeeks.iloc[:,3:5]
df_RepWeeks2021['Rep Weeks for 2021'] = pd.to_datetime(df_RepWeeks2021['Rep Weeks for 2021'])
df_RepWeeks2021 = df_RepWeeks2021.sort_values(by='Rep Weeks for 2021')

#Defining lists with names 
CapNames = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
PriceNamesV2 = ['c_FCR','c_aFRR_up','c_aFRR_down','c_mFRRup']
PriceNamesSolX = ['c_FCR','c_aFRR_up','c_aFRRdown','c_mFRR_up']
Markets =['FCR','aFRR Up','aFRR Down','mFRR']





df_SolX_single.index = df_SolX_single['HourDK']
df_SolX_combined.index = df_SolX_combined['HourDK']





def AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,bidVol,df_RepWeeks,week_start,week_end):

    #Bid_prices_validS = df_SolX_single[bidVol] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceS = df_SolX_single[bidVol]


    #Bid_prices_validC = df_SolX_combined[bidVol] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceC = df_SolX_combined[bidVol]

    bidsS_tot = np.zeros((10,24))
    bidsC_tot = np.zeros((10,24))

    x = np.arange(0,24,1)

    Week_nr = 0
    t = 168*week_start

    while t <= 168*week_end:

        start = df_SolX_single.index[t-168]
        end = df_SolX_single.index[t-1]

        WeekS = (BidPriceS.index >= start) & (BidPriceS.index  <= end)
        WeekC = (BidPriceC.index >= start) & (BidPriceC.index  <= end)


        bidsS = BidPriceS[WeekS].groupby(BidPriceS[WeekS].index.hour).mean()
        bidsC = BidPriceC[WeekC].groupby(BidPriceC[WeekC].index.hour).mean()

        for i in x:

            if i in bidsS.index:
                print('Wunderbar')
            else:
                print('Missing', i)
                bidsS.loc[i] =  0
        
        for j in x: 
            if j in bidsC.index:
                print('Wunderbar')
            else:
                print('Missing', j)
                bidsC.loc[j] =  0
        
        
        bidsS = bidsS.sort_index()
        bidsC = bidsC.sort_index()
        
        bidsS_tot[Week_nr] = bidsS
        bidsC_tot[Week_nr] = bidsC



        Week_nr += 1
        t +=168


    avgS = np.zeros(24)
    avgC = np.zeros(24)

    

    for i in range(0,len(bidsS_tot)):
        avgS = avgS + bidsS_tot[i] * df_RepWeeks.iloc[i,1]
        avgC = avgC + bidsC_tot[i] * df_RepWeeks.iloc[i,1]
        


    return avgS , avgC


## Bid vol ##
avgS_FCRbidVol_2020,avgC_FCRbidVol_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_FCR',df_RepWeeks2020,1,10)
avgS_FCRbidVol_2021,avgC_FCRbidVol_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_FCR',df_RepWeeks2021,11,20)

avgS_aFRR_up_bidVol_2020,avgC_aFRR_up_bidVol_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_aFRR_up',df_RepWeeks2020,1,10)
avgS_aFRR_up_bidVol_2021,avgC_aFRR_up_bidVol_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_aFRR_up',df_RepWeeks2021,11,20)

avgS_aFRR_down_bidVol_2020,avgC_aFRR_down_bidVol_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_aFRR_down',df_RepWeeks2020,1,10)
avgS_aFRR_down_bidVol_2021,avgC_aFRR_down_bidVol_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_aFRR_down',df_RepWeeks2021,11,20)

avgS_mFRR_bidVol_2020,avgC_mFRR_bidVol_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_mFRR_up',df_RepWeeks2020,1,10)
avgS_mFRR_bidVol_2021,avgC_mFRR_bidVol_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_mFRR_up',df_RepWeeks2021,11,20)

## Bid accepted ## 
avgS_FCRbidVolAcc_2020,avgC_FCRbidVolAcc_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_FCR',df_RepWeeks2020,1,10)
avgS_FCRbidVolAcc_2021,avgC_FCRbidVolAcc_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_FCR',df_RepWeeks2021,11,20)

avgS_aFRR_up_bidVolAcc_2020,avgC_aFRR_up_bidVolAcc_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_aFRR_up',df_RepWeeks2020,1,10)
avgS_aFRR_up_bidVolAcc_2021,avgC_aFRR_up_bidVolAcc_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_aFRR_up',df_RepWeeks2021,11,20)

avgS_aFRR_down_bidVolAcc_2020,avgC_aFRR_down_bidVolAcc_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_aFRR_down',df_RepWeeks2020,1,10)
avgS_aFRR_down_bidVolAcc_2021,avgC_aFRR_down_bidVolAcc_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_aFRR_down',df_RepWeeks2021,11,20)

avgS_mFRR_bidVolAcc_2020,avgC_mFRR_bidVolAcc_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_mFRR_up',df_RepWeeks2020,1,10)
avgS_mFRR_bidVolAcc_2021,avgC_mFRR_bidVolAcc_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'r_mFRR_up',df_RepWeeks2021,11,20)

## clearing prices ## 

avgS_FCRClear_2020,avgC_FCRClear_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_FCR',df_RepWeeks2020,1,10)
avgS_FCRClear_2021,avgC_FCRClear_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_FCR',df_RepWeeks2021,11,20)

avgS_aFRR_up_Clear_2020,avgC_aFRR_up_Clear_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_aFRR_up',df_RepWeeks2020,1,10)
avgS_aFRR_up_Clear_2021,avgC_aFRR_up_Clear_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_aFRR_up',df_RepWeeks2021,11,20)

avgS_aFRR_down_Clear_2020,avgC_aFRR_down_Clear_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_aFRRdown',df_RepWeeks2020,1,10)
avgS_aFRR_down_Clear_2021,avgC_aFRR_down_Clear_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_aFRRdown',df_RepWeeks2021,11,20)

avgS_mFRR_Clear_2020,avgC_mFRR_Clear_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_mFRR_up',df_RepWeeks2020,1,10)
avgS_mFRR_Clear_2021,avgC_mFRR_Clear_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'c_mFRR_up',df_RepWeeks2021,11,20)

avgS_DA_Clear_2020,avgC_DA_Clear_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'DA_clearing',df_RepWeeks2020,1,10)
avgS_DA_Clear_2021,avgC_DA_Clear_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'DA_clearing',df_RepWeeks2021,11,20)

## Avg PV Priduction 
#avgS_PV_2020,avgC_PV_2020 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_mFRR_up',df_RepWeeks2020,1,10)
#avgS_PV_2021,avgC_PV_2021 = AvgWeightedBidVolandOtherHourly(df_SolX_single,df_SolX_combined,'b_mFRR_up',df_RepWeeks2021,11,20)






def AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,bidVol,bidPrice,df_RepWeeks,week_start,week_end):

    Bid_prices_validS = df_SolX_single[bidVol] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceS = df_SolX_single[Bid_prices_validS][bidPrice]


    Bid_prices_validC = df_SolX_combined[bidVol] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceC = df_SolX_combined[Bid_prices_validC][bidPrice]

    bidsS_tot = np.zeros((10,24))
    bidsC_tot = np.zeros((10,24))

    x = np.arange(0,24,1)

    Week_nr = 0
    t = 168*week_start

    while t <= 168*week_end:

        start = df_SolX_single.index[t-168]
        end = df_SolX_single.index[t-1]

        WeekS = (BidPriceS.index >= start) & (BidPriceS.index  <= end)
        WeekC = (BidPriceC.index >= start) & (BidPriceC.index  <= end)


        bidsS = BidPriceS[WeekS].groupby(BidPriceS[WeekS].index.hour).mean()
        bidsC = BidPriceC[WeekC].groupby(BidPriceC[WeekC].index.hour).mean()

        for i in x:

            if i in bidsS.index:
                print('Wunderbar')
            else:
                print('Missing', i)
                bidsS.loc[i] =  0
        
        for j in x: 
            if j in bidsC.index:
                print('Wunderbar')
            else:
                print('Missing', j)
                bidsC.loc[j] =  0
        
        
        bidsS = bidsS.sort_index()
        bidsC = bidsC.sort_index()
        
        bidsS_tot[Week_nr] = bidsS
        bidsC_tot[Week_nr] = bidsC



        Week_nr += 1
        t +=168


    avgS = np.zeros(24)
    avgC = np.zeros(24)

    

    for i in range(0,len(bidsS_tot)):
        avgS = avgS + bidsS_tot[i] * df_RepWeeks.iloc[i,1]
        avgC = avgC + bidsC_tot[i] * df_RepWeeks.iloc[i,1]
        


    return avgS , avgC

## Bid Price ##
avgS_FCRbids_2020,avgC_FCRbids_2020 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_FCR','beta_FCR',df_RepWeeks2020,1,10)
avgS_FCRbids_2021,avgC_FCRbids_2021 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_FCR','beta_FCR',df_RepWeeks2021,11,20)

avgS_aFRR_up_bids_2020,avgC_aFRR_up_bids_2020 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_aFRR_up','beta_aFRR_up',df_RepWeeks2020,1,10)
avgS_aFRR_up_bids_2021,avgC_aFRR_up_bids_2021 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_aFRR_up','beta_aFRR_up',df_RepWeeks2021,11,20)

avgS_aFRR_down_bids_2020,avgC_aFRR_down_bids_2020 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_aFRR_down','beta_aFRR_down',df_RepWeeks2020,1,10)
avgS_aFRR_down_bids_2021,avgC_aFRR_down_bids_2021 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_aFRR_down','beta_aFRR_down',df_RepWeeks2021,11,20)

avgS_mFRR_bids_2020,avgC_mFRR_bids_2020 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_mFRR_up','beta_mFRR_up',df_RepWeeks2020,1,10)
avgS_mFRR_bids_2021,avgC_mFRR_bids_2021 = AvgWeightedBidPriceHourly(df_SolX_single,df_SolX_combined,'b_mFRR_up','beta_mFRR_up',df_RepWeeks2021,11,20)







fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

color = ['#a83232','#1f1485','#08a7c7','#148710', '#fc9803','#4e5450']
width = 0.5
x = np.arange((0,24,1))


# DA vs accepted reserve volumen only combined 

""" 
#Da vs Bid vol and acc vol
ax2 = axes.twinx()
axes.plot(x,avgS_FCRbidVolAcc_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'AV FCR')
axes.plot(x,avgS_aFRR_up_bidVolAcc_2020,linestyle='--', marker='o',markersize=5, color = color[0], label = 'AV aFRR Up')
axes.plot(x,avgS_aFRR_down_bidVolAcc_2020,linestyle='--', marker='o',markersize=5, color = color[3], label = 'AV aFRR Down')
axes.plot(x,avgS_mFRR_bidVolAcc_2020,linestyle='--', marker='o',markersize=5, color = color[4], label = 'AV mFRR')
ax2.bar(x,avgC_DA_Clear_2020, color = color[5], label = 'DA Price',width=width,alpha=0.4)

"""
""" 
# FCR Volume #
#axes[0].bar(x-(width/2),avgS_FCRbidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03) #Accepted Volume
#axes[0].bar(x+(width/2),avgS_FCRbidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
#axes[0].plot(x,avgS_FCRbidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020') #Bid Volume
#axes[0].plot(x,avgS_FCRbidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')

axes[1].bar(x-(width/2),avgC_FCRbidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03)
axes[1].bar(x+(width/2),avgC_FCRbidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
axes[1].plot(x,avgC_FCRbidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020')
axes[1].plot(x,avgC_FCRbidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')

"""

""" 
# FCR Prices #

#axes[0].bar(x-(width/2),avgS_FCRbids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
#axes[0].bar(x+(width/2),avgS_FCRbids_2021, color = color[3], label = 'BP 2021',width=width-0.03)
#axes[0].plot(x,avgS_FCRClear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
#axes[0].plot(x,avgS_FCRClear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')

axes[0].bar(x-(width/2),avgC_FCRbids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
axes[0].bar(x+(width/2),avgC_FCRbids_2021, color = color[3], label = 'BP 2021',width=width-0.03)
axes[0].plot(x,avgC_FCRClear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
axes[0].plot(x,avgC_FCRClear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')

"""
 
""" 
#aFRR up Volumen
#axes[0].bar(x-(width/2),avgS_aFRR_up_bidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03) #Accepted Volume
#axes[0].bar(x+(width/2),avgS_aFRR_up_bidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
#axes[0].plot(x,avgS_aFRR_up_bidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020') #Bid Volume
#axes[0].plot(x,avgS_aFRR_up_bidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')

axes[1].bar(x-(width/2),avgC_aFRR_up_bidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03)
axes[1].bar(x+(width/2),avgC_aFRR_up_bidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
axes[1].plot(x,avgC_aFRR_up_bidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020')
axes[1].plot(x,avgC_aFRR_up_bidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')
"""

""" 
#aFRR up  Prices
#axes[0].bar(x-(width/2),avgS_aFRR_up_bids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
#axes[0].bar(x+(width/2),avgS_aFRR_up_bids_2021, color = color[3], label = 'BP 2021',width=width-0.03)
#axes[0].plot(x,avgS_aFRR_up_Clear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
#axes[0].plot(x,avgS_aFRR_up_Clear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')

axes[0].bar(x-(width/2),avgC_aFRR_up_bids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
axes[0].bar(x+(width/2),avgC_aFRR_up_bids_2021, color = color[3], label = 'BP 2021',width=width-0.03)
axes[0].plot(x,avgC_aFRR_up_Clear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
axes[0].plot(x,avgC_aFRR_up_Clear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')
"""

""" 
#aFRR down Volumen
#axes[0].bar(x-(width/2),avgS_aFRR_down_bidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03) #Accepted Volume
#axes[0].bar(x+(width/2),avgS_aFRR_down_bidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
#axes[0].plot(x,avgS_aFRR_down_bidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020') #Bid Volume
#axes[0].plot(x,avgS_aFRR_down_bidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')

axes[1].bar(x-(width/2),avgC_aFRR_down_bidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03)
axes[1].bar(x+(width/2),avgC_aFRR_down_bidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
axes[1].plot(x,avgC_aFRR_down_bidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020')
axes[1].plot(x,avgC_aFRR_down_bidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')

"""

""" 
#aFRR downPrices
#axes[0].bar(x-(width/2),avgS_aFRR_down_bids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
#axes[0].bar(x+(width/2),avgS_aFRR_down_bids_2021, color = color[3], label = 'BP 2021',width=width-0.03)
#axes[0].plot(x,avgS_aFRR_down_Clear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
#axes[0].plot(x,avgS_aFRR_down_Clear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')

axes[0].bar(x-(width/2),avgC_aFRR_down_bids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
axes[0].bar(x+(width/2),avgC_aFRR_down_bids_2021, color = color[3], label = 'BP 2021',width=width-0.03)
axes[0].plot(x,avgC_aFRR_down_Clear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
axes[0].plot(x,avgC_aFRR_down_Clear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')
"""

 
""" 
#mfrr Prices
#axes[0].bar(x-(width/2),avgS_mFRR_bids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
#axes[0].bar(x+(width/2),avgS_mFRR_bids_2021, color = color[3], label = 'BP 2021',width=width-0.03)
#axes[0].plot(x,avgS_mFRR_Clear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
#axes[0].plot(x,avgS_mFRR_Clear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')

axes[0].bar(x-(width/2),avgC_mFRR_bids_2020, color = color[2], label = 'BP 2020',width=width-0.03)
axes[0].bar(x+(width/2),avgC_mFRR_bids_2020, color = color[3], label = 'BP 2021',width=width-0.03)
axes[0].plot(x,avgC_mFRR_Clear_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'CP 2020')
axes[0].plot(x,avgC_mFRR_Clear_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'CP 2021')
 """

"""  
#mFRR Volumen
axes[0].bar(x-(width/2),avgS_mFRR_bidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03) #Accepted Volume
axes[0].bar(x+(width/2),avgS_mFRR_bidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
axes[0].plot(x,avgS_mFRR_bidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020') #Bid Volume
axes[0].plot(x,avgS_mFRR_bidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')

axes[1].bar(x-(width/2),avgC_mFRR_bidVolAcc_2020, color = color[2], label = 'AV 2020',width=width-0.03)
axes[1].bar(x+(width/2),avgC_mFRR_bidVolAcc_2021, color = color[3], label = 'AV 2021',width=width-0.03)
axes[1].plot(x,avgC_mFRR_bidVol_2020,linestyle='--', marker='o',markersize=5, color = color[1], label = 'BV 2020')
axes[1].plot(x,avgC_mFRR_bidVol_2021,linestyle='--', marker='o',markersize=5, color = color[0], label = 'BV 2021')
 """
#axes[0,1].bar(x-(width/2),avgS_aFRR_up_bids_2020, color = color[3], label = 'BP 2020',width=width-0.03)
#axes[0,1].bar(x+(width/2),avgS_aFRR_up_bids_2021, color = color[2], label = 'BP 2021',width=width-0.03)
#axes[0,1].plot(x,avgS_aFRR_up_Clear_2020,linestyle='--', marker='o',markersize=4, color = color[1], label = 'CP 2020')
#axes[0,1].plot(x,avgS_aFRR_up_Clear_2021,linestyle='--', marker='o',markersize=4, color = color[0], label = 'CP 2021')

#axes[1,0].bar(x-(width/2),avgS_aFRR_down_bids_2020, color = color[3], label = 'BP 2020',width=width-0.03)
#axes[1,0].bar(x+(width/2),avgS_aFRR_down_bids_2021, color = color[2], label = 'BP 2021',width=width-0.03)
#axes[1,0].plot(x,avgS_aFRR_down_Clear_2020,linestyle='--', marker='o',markersize=4, color = color[1], label = 'CP 2020')
#axes[1,0].plot(x,avgS_aFRR_down_Clear_2021,linestyle='--', marker='o',markersize=4, color = color[0], label = 'CP 2021')

#axes[1].bar(x-(width/2),avgS_mFRR_bids_2020, color = color[3], label = 'BP 2020',width=width-0.03)
#axes[1].bar(x+(width/2),avgS_mFRR_bids_2021, color = color[2], label = 'BP 2021',width=width-0.03)
#axes[1].plot(x,avgS_mFRR_Clear_2020,linestyle='--', marker='o',markersize=4, color = color[1], label = 'CP 2020')
#axes[1].plot(x,avgS_mFRR_Clear_2021,linestyle='--', marker='o',markersize=4, color = color[0], label = 'CP 2021')

#axes.set_xlim(-1.5,24)
axes[0].set_xlim(-1.5,24)
#axes[0,1].set_xlim(-1.5,24)
#axes[1,0].set_xlim(-1.5,24)
axes[1].set_xlim(-1.5,24)
axes[0].set_title('IS: mFRR',fontsize=12,pad=15)
#axes[0,1].set_title('aFRR Up',fontsize=12,pad=15)
#axes[1,0].set_title('aFRR Down',fontsize=12,pad=15)
axes[1].set_title('DS: mFRR',fontsize=12,pad=15)

axes[0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.18), ncol=4,frameon=False)
#axes.legend(loc='upper center',bbox_to_anchor=(0.5, 1.17), ncol=4,frameon=False)
#axes[0,1].legend(loc='upper center',bbox_to_anchor=(0.5, 1.2), ncol=2,frameon=False)
#axes[1,0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.2), ncol=2,frameon=False)
axes[1].legend(loc='upper center',bbox_to_anchor=(0.5, 1.18), ncol=4,frameon=False)

#axes.set_ylabel('MW',fontsize=10)
#ax2.set_ylabel('€/MWh',fontsize=10)


axes[0].set_ylabel('MW',fontsize=12)
#axes[0,1].set_ylabel('€/MW',fontsize=10)
#axes[1,0].set_ylabel('€/MW',fontsize=10)
#axes[1].set_ylabel('MW',fontsize=10)
axes[1].set_ylabel('MW',fontsize=12)
axes[1].set_xlabel('Hours',fontsize=12)

loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
#axes[0].xaxis.set_major_locator(loc)
axes[1].xaxis.set_major_locator(loc)
#axes[0,1].xaxis.set_major_locator(loc)
#axes[1,0].xaxis.set_major_locator(loc)
#axes[1].xaxis.set_major_locator(loc)

fig.suptitle('mFRR',fontsize=14)
#fig.suptitle.set_position([.5, 1.05]
plt.tight_layout()
plt.show()



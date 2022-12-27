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


#Improting V2 year vOPEX
file_to_open = Path("DataAnalysis/BASE/") / 'BASE_EconResults_All.xlsx'
DataX = pd.ExcelFile(file_to_open)
df_results = pd.read_excel(file_to_open, sheet_name=DataX.sheet_names)
sheet = list(df_results)


#Importing V2 year 
file_to_open = Path("Result_files/") / 'V2_year_2020-01-01_2020-12-31.xlsx'
df_results2020M2 = pd.read_excel(file_to_open)
file_to_open = Path("Result_files/") / 'V2_year_2021-01-01_2021-12-31.xlsx'
df_results2021M2 = pd.read_excel(file_to_open)

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



## Investigating if V2 weeks represent V2 year by comparing vOPEX for both 

def RepWeekCostBreaktoYear(df_V2_weeks,df_RepWeeks2020,Markets,PriceNamesV2,CapNames,from_week,end_week):

    TotRevV2 = np.zeros((len(df_RepWeeks2020),len(Markets)))
    DA_Tot = np.zeros((len(df_RepWeeks2020),4))
    Power = np.zeros((len(df_RepWeeks2020),2))  #Import , Export 
    weeks = 168 *from_week
    prob_week = 0

    while weeks <= 168*end_week: # for one year

        df_w = df_V2_weeks.iloc[(weeks-168):weeks,:]
        print(df_w.iloc[0,0])


        ## DA Cost and Rev 
        DA_rev = 0
        DA_cost = 0
        ConT = 0
        ProdT = 0
        P_import = 0
        P_export = 0

        for t in range(0,len(df_w)):
            DA_rev = DA_rev + df_w['P_export'].iloc[t] * df_w['c_DA'].iloc[t]
            DA_cost = DA_cost + df_w['P_import'].iloc[t] * df_w['c_DA'].iloc[t]
            ConT = ConT + df_w['P_export'].iloc[t] * CT
            ProdT = ProdT + df_w['P_import'].iloc[t] * PT
            P_import = P_import + df_w['P_import'].iloc[t]
            P_export = P_export + df_w['P_export'].iloc[t]


        DA_Tot[prob_week][0] = DA_rev * (365/7) * df_RepWeeks2020.iloc[prob_week,1]
        DA_Tot[prob_week][1] = DA_cost * (365/7) * df_RepWeeks2020.iloc[prob_week,1]
        DA_Tot[prob_week][2] = ConT * (365/7) * df_RepWeeks2020.iloc[prob_week,1]
        DA_Tot[prob_week][3] = ProdT * (365/7) * df_RepWeeks2020.iloc[prob_week,1]
        Power[prob_week][0] = P_import * (365/7) * df_RepWeeks2020.iloc[prob_week,1]
        Power[prob_week][1] = P_export * (365/7) * df_RepWeeks2020.iloc[prob_week,1]
        ## Reserve Revenue 
        for i in range(0,len(Markets)): 
            Res_RevV2 = 0 

            for t in range(0,len(df_w)):
                Res_RevV2 = Res_RevV2 + df_w[CapNames[i]].iloc[t]*df_w[PriceNamesV2[i]].iloc[t]

            
            TotRevV2[prob_week][i] = Res_RevV2 * (365/7)*df_RepWeeks2020.iloc[prob_week,1] 
            


        prob_week += 1
        weeks += 168




    return TotRevV2, DA_Tot, Power

TotRevV22020,DA_Tot2020,Power = RepWeekCostBreaktoYear(df_V2_weeks,df_RepWeeks2020,Markets,PriceNamesV2,CapNames,1,10)

def PlotRepWeekvsYearV2costBreak():


    TotRevV22020,DA_Tot2020,Power = RepWeekCostBreaktoYear(df_V2_weeks,df_RepWeeks2020,Markets,PriceNamesV2,CapNames,1,10)
    TotRevV22021,DA_Tot2021,Power = RepWeekCostBreaktoYear(df_V2_weeks,df_RepWeeks2021,Markets,PriceNamesV2,CapNames,11,20)




    vOpexV2_2020 = df_results[sheet[4]]['vOPEX_DA_revenue'][1] + df_results[sheet[4]]['vOPEX_DA_expenses'][1] + df_results[sheet[4]]['vOPEX_CT'][1] + df_results[sheet[4]]['vOPEX_PT'][1] + df_results[sheet[4]]['vOPEX_FCR'][1] + df_results[sheet[4]]['vOPEX_aFRRup'][1] + df_results[sheet[4]]['vOPEX_aFRRdown'][1]+ df_results[sheet[4]]['vOPEX_mFRRup'][1] #2020 & 2021
    vOpexV2_2021 = df_results[sheet[5]]['vOPEX_DA_revenue'][1] + df_results[sheet[5]]['vOPEX_DA_expenses'][1] + df_results[sheet[5]]['vOPEX_CT'][1] + df_results[sheet[5]]['vOPEX_PT'][1] + df_results[sheet[5]]['vOPEX_FCR'][1] + df_results[sheet[5]]['vOPEX_aFRRup'][1] + df_results[sheet[5]]['vOPEX_aFRRdown'][1]+ df_results[sheet[5]]['vOPEX_mFRRup'][1] #2020 & 2021

    df_reserveV2_weeks2020 = pd.DataFrame(TotRevV22020, columns=Markets)
    df_reserveV2_weeks2020['DA_Rev'] = DA_Tot2020[:,0]
    df_reserveV2_weeks2020['DA_Cost'] = DA_Tot2020[:,1]
    df_reserveV2_weeks2020['ConT'] = DA_Tot2020[:,2]
    df_reserveV2_weeks2020['ProdT'] = DA_Tot2020[:,3]


    df_reserveV2_weeks2021 = pd.DataFrame(TotRevV22021, columns=Markets)
    df_reserveV2_weeks2021['DA_Rev'] = DA_Tot2021[:,0]
    df_reserveV2_weeks2021['DA_Cost'] = DA_Tot2021[:,1]
    df_reserveV2_weeks2021['ConT'] = DA_Tot2021[:,2]
    df_reserveV2_weeks2021['ProdT'] = DA_Tot2021[:,3]

    ## From EUR to mEUR
    df_reserveV2_weeks2020 = df_reserveV2_weeks2020.div(10**6)
    df_reserveV2_weeks2021 = df_reserveV2_weeks2021.div(10**6)

    fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True,gridspec_kw={'width_ratios': [1, 1]})

    x = ['Rep Year','Full Year']


    #2020
    DA_Rev2020 = np.array([df_reserveV2_weeks2020['DA_Rev'].sum(),-df_results[sheet[4]]['vOPEX_DA_revenue'][1]])
    DA_Cost2020 = np.array([-df_reserveV2_weeks2020['DA_Cost'].sum(),-df_results[sheet[4]]['vOPEX_DA_expenses'][1]])
    FCR_Rev2020 = np.array([df_reserveV2_weeks2020['FCR'].sum(),-df_results[sheet[4]]['vOPEX_FCR'][1]])
    aFRR_up2020 = np.array([df_reserveV2_weeks2020['aFRR Up'].sum(), -df_results[sheet[4]]['vOPEX_aFRRup'][1]])
    aFRR_down2020 = np.array([df_reserveV2_weeks2020['aFRR Down'].sum(), -df_results[sheet[4]]['vOPEX_aFRRdown'][1]])
    mFRR2020 = np.array([df_reserveV2_weeks2020['mFRR'].sum(), -df_results[sheet[4]]['vOPEX_mFRRup'][1]])

    var1_2020 = DA_Rev2020 + FCR_Rev2020
    var2_2020 = var1_2020 + aFRR_up2020 
    var3_2020 = aFRR_down2020 + var2_2020
    var4_2020 = np.array([-df_reserveV2_weeks2020['ProdT'].sum(),-df_results[sheet[4]]['vOPEX_PT'][1]]) + DA_Cost2020



    #2021
    DA_Rev2021 = np.array([df_reserveV2_weeks2021['DA_Rev'].sum(),-df_results[sheet[5]]['vOPEX_DA_revenue'][1]])
    DA_Cost2021 = np.array([-df_reserveV2_weeks2021['DA_Cost'].sum(),-df_results[sheet[5]]['vOPEX_DA_expenses'][1]])
    FCR_Rev2021 = np.array([df_reserveV2_weeks2021['FCR'].sum(),-df_results[sheet[5]]['vOPEX_FCR'][1]])
    aFRR_up2021 = np.array([df_reserveV2_weeks2021['aFRR Up'].sum(), -df_results[sheet[5]]['vOPEX_aFRRup'][1]])
    aFRR_down2021 = np.array([df_reserveV2_weeks2021['aFRR Down'].sum(), -df_results[sheet[5]]['vOPEX_aFRRdown'][1]])
    mFRR2021 = np.array([df_reserveV2_weeks2021['mFRR'].sum(), -df_results[sheet[5]]['vOPEX_mFRRup'][1]])

    var1_2021 = DA_Rev2021 + FCR_Rev2021
    var2_2021 = var1_2021 + aFRR_up2021
    var3_2021 = aFRR_down2021 + var2_2021
    var4_2021 = np.array([-df_reserveV2_weeks2021['ProdT'].sum(),-df_results[sheet[5]]['vOPEX_PT'][1]]) + DA_Cost2021





    #Revenue
    ax1.bar(x, mFRR2020, bottom=var3_2020, color='goldenrod',linestyle = 'solid', label ='mFRR Up',width=0.75)
    ax1.bar(x, aFRR_down2020, bottom=var2_2020, color='navy',linestyle = 'solid', label ='aFRR Down',width=0.75)
    ax1.bar(x, aFRR_up2020, bottom=var1_2020, color='royalblue',linestyle = 'solid', label ='aFRR Up',width=0.75)
    ax1.bar(x, FCR_Rev2020,bottom= DA_Rev2020 ,color='darkorange',linestyle = 'solid', label ='FCR',width=0.75)
    ax1.bar(x, DA_Rev2020, color='#0a940a',linestyle = 'solid', label ='DA Revenue',width=0.75)

    #Cost
    ax1.bar(x , DA_Cost2020, color='#d90f0f',linestyle = 'solid', label ='DA Cost',width=0.75)  
    ax1.bar(x , [-df_reserveV2_weeks2020['ProdT'].sum(),-df_results[sheet[4]]['vOPEX_PT'][1]], bottom= DA_Cost2020,  color='#1a0606',linestyle = '-', label ='Producer T',width=0.75)
    ax1.bar(x , [-df_reserveV2_weeks2020['ConT'].sum(),-df_results[sheet[4]]['vOPEX_CT'][1]], color='#8a0a0a',bottom=var4_2020 ,linestyle = '-', label ='Consumer T',width=0.75)
    ax1.set_title('2020')


    #Revenue
    ax2.bar(x, mFRR2021, bottom=var3_2021, color='goldenrod',linestyle = 'solid', label ='mFRR Up',width=0.75)
    ax2.bar(x, aFRR_down2021, bottom=var2_2021, color='navy',linestyle = 'solid', label ='aFRR Down',width=0.75)
    ax2.bar(x, aFRR_up2021, bottom=var1_2021, color='royalblue',linestyle = 'solid', label ='aFRR Up',width=0.75)
    ax2.bar(x, FCR_Rev2021,bottom= DA_Rev2021 ,color='darkorange',linestyle = 'solid', label ='FCR',width=0.75)
    ax2.bar(x, DA_Rev2021, color='#0a940a',linestyle = 'solid', label ='DA Revenue',width=0.75)

    #Cost
    ax2.bar(x , DA_Cost2021, color='#d90f0f',linestyle = 'solid', label ='DA Cost',width=0.75)  
    ax2.bar(x , [-df_reserveV2_weeks2021['ProdT'].sum(),-df_results[sheet[5]]['vOPEX_PT'][1]], bottom= DA_Cost2021,  color='#1a0606',linestyle = '-', label ='Producer T',width=0.75)
    ax2.bar(x , [-df_reserveV2_weeks2021['ConT'].sum(),-df_results[sheet[5]]['vOPEX_CT'][1]], color='#8a0a0a',bottom=var4_2021 ,linestyle = '-', label ='Consumer T',width=0.75)



    #ax2.set_ylabel('mio €')
    ax2.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
    ax2.set_title('2021')
    #ax2.set_ylim([0,15])

    ax2.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.show()  

PlotRepWeekvsYearV2costBreak()

#Net import Full year vs Rep Year !! To see if CT and PT makes sense  

def NetPowerImport():
    TotRevV22020, DA_Tot2020, Power2020 = RepWeekCostBreaktoYear(df_V2_weeks,df_RepWeeks2020,Markets,PriceNamesV2,CapNames,1,10)
    TotRevV22021, DA_Tot2021, Power2021 = RepWeekCostBreaktoYear(df_V2_weeks,df_RepWeeks2021,Markets,PriceNamesV2,CapNames,11,20)

    NetImport2020Rep = sum(Power2020[:,0]) - sum(Power2020[:,1])
    NetImport2021Rep = sum(Power2021[:,0]) - sum(Power2021[:,1])

    x = np.arange(0,2,1)
    width = 0.3 
    fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2)


    ax1.bar(x, [sum(Power2020[:,0]),sum(Power2021[:,0])], color ='#148710', label = 'Rep Year Import',width=width-0.05 )
    ax1.bar(x+width, [df_results2020M2['P_import'].sum(),df_results2021M2['P_import'].sum()], color ='#1f1485', label = 'Full Year import',width=width-0.05)

    ax2.bar(x, [sum(Power2020[:,1]),sum(Power2021[:,1])], label = 'Rep Year Export',width=width-0.05 )
    ax2.bar(x+width, [df_results2020M2['P_export'].sum(),df_results2021M2['P_export'].sum()], label = 'Full Year export',width=width-0.05)


    ax1.set_ylabel('MWh')
    ax2.set_ylabel('MWh')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    #ax1.set_ylim([0,15])

    ax1.tick_params(axis='x', rotation=0)
    ax2.tick_params(axis='x', rotation=0)
    ax1.set_xticks(x + width / 2, ('2020', '2021'))
    ax2.set_xticks(x + width / 2, ('2020', '2021'))
    plt.tight_layout()
    plt.show()

NetPowerImport()














# Plot start & end date 
plot_start = '2020-01-01 00:00'
plot_end = '2020-12-31 23:59'


TimeRangeV2 = (df_V2_weeks['HourDK'] >= plot_start) & (df_V2_weeks['HourDK']  <= plot_end)
TimeRangeSolXSingle = (df_SolX_single['HourDK'] >= plot_start) & (df_SolX_single['HourDK']  <= plot_end)
TimeRangeSolXCombined = (df_SolX_combined['HourDK'] >= plot_start) & (df_SolX_combined['HourDK']  <= plot_end)


df_V2PLot = df_V2_weeks[TimeRangeV2]
df_SolX_singlePLot = df_SolX_single[TimeRangeSolXSingle]
df_SolX_combinedPLot = df_SolX_combined[TimeRangeSolXCombined]


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

def BarHCap(df_SolXSingle,df_SolXCombined,df_V2): #Not weighted yet Should be done!! 

    Names = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']
    TotVolV2 = []
    TotVolSolXSingle = []
    TotVolSolXCombined = []

    # Sum the values
    for i in Names: 
        TotVolV2.append(sum(df_V2[i]))
        TotVolSolXSingle.append(sum(df_SolXSingle[i]))
        TotVolSolXCombined.append(sum(df_SolXCombined[i]))

    df = pd.DataFrame({'Potential': TotVolV2,
                        'Realized IS': TotVolSolXSingle,
                        'Realized DS': TotVolSolXCombined}, index=Markets)

    df.plot.barh(color=['#008fd5','#fc4f30','#43eb34'])
    plt.legend(loc='lower right')
    plt.xlabel('MW')
    plt.tight_layout()
    plt.show()




#Not weighted yet Should be done!!
def BarHRev(df_V2,df_SolXsingle,df_SolXcombined):

    # Calculate Revenue # 

    CapNames = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    PriceNamesV2 = ['c_FCR','c_aFRR_up','c_aFRR_down','c_mFRRup']
    PriceNamesSolX = ['c_FCR','c_aFRR_up','c_aFRRdown','c_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']

    TotRevV2 = []
    TotRevSolXSingle = []
    TotRevSolXCombined = []


    for i in range(0,len(Markets)):
        RevV2 = 0
        RevSolXsingle = 0
        RevSolXcombined = 0
        

        for t in range(0,len(df_V2)):
            RevV2 = RevV2 + df_V2[CapNames[i]].iloc[t]*df_V2[PriceNamesV2[i]].iloc[t]
            RevSolXsingle = RevSolXsingle + df_SolXsingle[CapNames[i]].iloc[t]*df_SolXsingle[PriceNamesSolX[i]].iloc[t] 
            RevSolXcombined = RevSolXcombined + df_SolXcombined[CapNames[i]].iloc[t]*df_SolXcombined[PriceNamesSolX[i]].iloc[t] 
        
        TotRevV2.append(RevV2)
        TotRevSolXSingle.append(RevSolXsingle)
        TotRevSolXCombined.append(RevSolXcombined)



    df = pd.DataFrame({'Potential': [TotRevV2[i]/1000000 for i in range(0,len(TotRevV2))],
                        'Realized IS': [TotRevSolXSingle[i]/1000000 for i in range(0,len(TotRevSolXSingle))],
                        'Realized DS': [TotRevSolXCombined[i]/1000000 for i in range(0,len(TotRevSolXCombined))]}, index=Markets)

    df.plot.barh(color=['#008fd5','#fc4f30','#43eb34'])
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


## Change manually to switch between years P_PEM set point 
def RepWeekToYear(df_SolX):

    weeks = 168  #hours in a week
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
BarHCap(df_SolX_singlePLot,df_SolX_combinedPLot,df_V2PLot)


## Horizontal revenue plot ## 
BarHRev(df_V2PLot,df_SolX_singlePLot,df_SolX_combinedPLot)


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








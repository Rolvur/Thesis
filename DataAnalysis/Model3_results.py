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
            ProdT = ProdT + df_w['P_export'].iloc[t] * PT
            ConT = ConT + df_w['P_import'].iloc[t] * CT
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

    #Calculating the difference for rep years in percenatage  

    #DA
    diff_DArev_2020 = (df_reserveV2_weeks2020['DA_Rev'].sum()-(-df_results[sheet[4]]['vOPEX_DA_revenue'][1]))/-df_results[sheet[4]]['vOPEX_DA_revenue'][1]
    diff_DArev_2021 = (df_reserveV2_weeks2021['DA_Rev'].sum()-(-df_results[sheet[5]]['vOPEX_DA_revenue'][1]))/-df_results[sheet[5]]['vOPEX_DA_revenue'][1]
    diff_DAcost_2020 = (df_reserveV2_weeks2020['DA_Cost'].sum()-(df_results[sheet[4]]['vOPEX_DA_expenses'][1]))/df_results[sheet[4]]['vOPEX_DA_expenses'][1]
    diff_DAcost_2021 = (df_reserveV2_weeks2021['DA_Cost'].sum()-(df_results[sheet[5]]['vOPEX_DA_expenses'][1]))/df_results[sheet[5]]['vOPEX_DA_expenses'][1]

    #CT and PT 
    diff_CT_2020 = (df_reserveV2_weeks2020['ConT'].sum() - df_results[sheet[4]]['vOPEX_CT'][1]) / df_results[sheet[4]]['vOPEX_CT'][1]
    diff_CT_2021 = (df_reserveV2_weeks2021['ConT'].sum() - df_results[sheet[5]]['vOPEX_CT'][1]) / df_results[sheet[5]]['vOPEX_CT'][1]

    diff_PT_2020 = (df_reserveV2_weeks2020['ProdT'].sum() - df_results[sheet[4]]['vOPEX_PT'][1]) / df_results[sheet[4]]['vOPEX_PT'][1]
    diff_PT_2021 = (df_reserveV2_weeks2021['ProdT'].sum() - df_results[sheet[5]]['vOPEX_PT'][1]) / df_results[sheet[5]]['vOPEX_PT'][1]

    print('diff_PT_2020', diff_PT_2020)
    print('diff_PT_2021', diff_PT_2021)

    #FCR 
    diff_FCR_2020 = (df_reserveV2_weeks2020['FCR'].sum() - (-df_results[sheet[4]]['vOPEX_FCR'][1])) / (-df_results[sheet[4]]['vOPEX_FCR'][1])
    diff_FCR_2021 = (df_reserveV2_weeks2021['FCR'].sum() - (-df_results[sheet[5]]['vOPEX_FCR'][1])) / (-df_results[sheet[5]]['vOPEX_FCR'][1])

    #aFRR 
    diff_aFRRup_2020 = (df_reserveV2_weeks2020['aFRR Up'].sum() - (-df_results[sheet[4]]['vOPEX_aFRRup'][1])) / (-df_results[sheet[4]]['vOPEX_aFRRup'][1])
    diff_aFRRup_2021 = (df_reserveV2_weeks2021['aFRR Up'].sum() - (-df_results[sheet[5]]['vOPEX_aFRRup'][1])) / (-df_results[sheet[5]]['vOPEX_aFRRup'][1])

    diff_aFRRdown_2020 = (df_reserveV2_weeks2020['aFRR Down'].sum() - (-df_results[sheet[4]]['vOPEX_aFRRdown'][1])) / (-df_results[sheet[4]]['vOPEX_aFRRdown'][1])
    diff_aFRRdown_2021 = (df_reserveV2_weeks2021['aFRR Down'].sum() - (-df_results[sheet[5]]['vOPEX_aFRRdown'][1])) / (-df_results[sheet[5]]['vOPEX_aFRRdown'][1])

    #mFRR 
    diff_mFRR_2020 = (df_reserveV2_weeks2020['mFRR'].sum() - (-df_results[sheet[4]]['vOPEX_mFRRup'][1])) / (-df_results[sheet[4]]['vOPEX_mFRRup'][1])
    diff_mFRR_2021 = (df_reserveV2_weeks2021['mFRR'].sum() - (-df_results[sheet[5]]['vOPEX_mFRRup'][1])) / (-df_results[sheet[5]]['vOPEX_mFRRup'][1])




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


    ax1.bar(x, [sum(Power2020[:,0])/1000,sum(Power2021[:,0])/1000], color ='#148710', label = 'Rep Year',width=width-0.05 )
    ax1.bar(x+width, [df_results2020M2['P_import'].sum()/1000,df_results2021M2['P_import'].sum()/1000], color ='#1f1485', label = 'Full Year',width=width-0.05)

    ax2.bar(x, [sum(Power2020[:,1])/1000,sum(Power2021[:,1])/1000], label = 'Rep Year',width=width-0.05 )
    ax2.bar(x+width, [df_results2020M2['P_export'].sum()/1000,df_results2021M2['P_export'].sum()/1000], label = 'Full Year',width=width-0.05)


    ax1.set_ylabel('GWh')
    ax2.set_ylabel('GWh')
    
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    
    ax1.set_title('Import')
    ax2.set_title('Export')


    ax1.set_ylim([0,230])
    ax2.set_ylim([0,170])

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




## Use this for plot one year at a time 
def BarHCap(df_V2,df_SolXSingle,df_SolXCombined,df_RepWeeks):  

    Names = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']
    
    TotVolV2 = np.zeros((len(df_RepWeeks),len(Markets)))
    TotVolSolXSingle = np.zeros((len(df_RepWeeks),len(Markets)))
    TotVolSolXCombined = np.zeros((len(df_RepWeeks),len(Markets)))
    

    weeks = 168     # 168 for 2020 and 168*11 for 2021 (in hours)
    end_week = 1680  # 1680 for 2020 and 3360 for 2021 (in hours)
    prob_week = 0

    while weeks <= end_week:

        
        df_weekV2 = df_V2.iloc[(weeks-168):weeks,:]
        df_weekSingle = df_SolXSingle.iloc[(weeks-168):weeks,:]
        df_weekCombined = df_SolXCombined.iloc[(weeks-168):weeks,:]


        # Sum the values
        for i in range(0,len(Names)): 

            CapacityV2 = 0
            CapacitySingle = 0
            CapacityComb = 0
            for t in range(0,len(df_weekV2)):

                CapacityV2 = CapacityV2 + df_weekV2[Names[i]].iloc[t]
                CapacitySingle = CapacitySingle + df_weekSingle[Names[i]].iloc[t]
                CapacityComb = CapacityComb + df_weekCombined[Names[i]].iloc[t]



            TotVolV2[prob_week][i] = CapacityV2 * (365/7) * df_RepWeeks.iloc[prob_week,1]
            TotVolSolXSingle[prob_week][i] = CapacitySingle * (365/7) * df_RepWeeks.iloc[prob_week,1]
            TotVolSolXCombined[prob_week][i] = CapacityComb * (365/7) * df_RepWeeks.iloc[prob_week,1]


        weeks += 168 
        prob_week += 1

    TotVolV2_sum = np.zeros(len(Markets))
    TotVolSolXSingle_sum = np.zeros(len(Markets))
    TotVolSolXCombined_sum = np.zeros(len(Markets))

    for i in range(0,len(Markets)):
        TotVolV2_sum[i] = TotVolV2[:,i].sum()/1000
        TotVolSolXSingle_sum[i] = TotVolSolXSingle[:,i].sum()/1000
        TotVolSolXCombined_sum[i] = TotVolSolXCombined[:,i].sum()/1000



    
    df = pd.DataFrame({'Potential': TotVolV2_sum,
                        'Realized IS': TotVolSolXSingle_sum,
                        'Realized DS': TotVolSolXCombined_sum}, index=Markets)

    df.plot.barh(color=['#008fd5','#fc4f30','#43eb34'])
    plt.legend(loc='lower right')
    plt.xlabel('GW$^*$')
    plt.tight_layout()
    plt.show()

    return df


### Horizontal Bar Chart Not used ### 

""" BarHCap2020 = BarHCap(df_V2PLot,df_SolX_singlePLot,df_SolX_combinedPLot,df_RepWeeks2020)

BarHCap2021 = BarHCap(df_V2PLot,df_SolX_singlePLot,df_SolX_combinedPLot,df_RepWeeks2021)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

axes[0].grid()
axes[1].grid()

BarHCap2020.plot.barh(ax=axes[0])
BarHCap2021.plot.barh(ax=axes[1],legend=False)

axes[0].set_title('2020',fontsize=12)
axes[1].set_title('2021',fontsize=12)
axes[1].set_xlabel('m€',fontsize=12)


plt.tight_layout()
plt.show()

 """


""" #Not weighted yet Should be done!!
def BarHRev(df_V2,df_SolXsingle,df_SolXcombined,df_RepWeeks):

    # Calculate Revenue # 

    CapNames = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    PriceNamesV2 = ['c_FCR','c_aFRR_up','c_aFRR_down','c_mFRRup']
    PriceNamesSolX = ['c_FCR','c_aFRR_up','c_aFRRdown','c_mFRR_up']
    Markets =['FCR','aFRR Up','aFRR Down','mFRR']

    TotVolV2 = np.zeros((len(df_RepWeeks),len(Markets)))
    TotVolSolXSingle = np.zeros((len(df_RepWeeks),len(Markets)))
    TotVolSolXCombined = np.zeros((len(df_RepWeeks),len(Markets)))
    

    weeks = 168     # 168 for 2020 and 168*11 for 2021 (in hours)
    end_week = 1680  # 1680 for 2020 and 3360 for 2021 (in hours)
    prob_week = 0

    while weeks <= end_week:

        
        df_weekV2 = df_V2.iloc[(weeks-168):weeks,:]
        df_weekSingle = df_SolXSingle.iloc[(weeks-168):weeks,:]
        df_weekCombined = df_SolXCombined.iloc[(weeks-168):weeks,:]


        # Sum the values
        for i in range(0,len(Names)): 

            CapacityV2 = 0
            CapacitySingle = 0
            CapacityComb = 0
            for t in range(0,len(df_weekV2)):

                CapacityV2 = CapacityV2 + df_weekV2[Names[i]].iloc[t]
                CapacitySingle = CapacitySingle + df_weekSingle[Names[i]].iloc[t]
                CapacityComb = CapacityComb + df_weekCombined[Names[i]].iloc[t]



            TotVolV2[prob_week][i] = CapacityV2 * (365/7) * df_RepWeeks.iloc[prob_week,1]
            TotVolSolXSingle[prob_week][i] = CapacitySingle * (365/7) * df_RepWeeks.iloc[prob_week,1]
            TotVolSolXCombined[prob_week][i] = CapacityComb * (365/7) * df_RepWeeks.iloc[prob_week,1]


        weeks += 168 
        prob_week += 1

    TotVolV2_sum = np.zeros(len(Markets))
    TotVolSolXSingle_sum = np.zeros(len(Markets))
    TotVolSolXCombined_sum = np.zeros(len(Markets))

    for i in range(0,len(Markets)):
        TotVolV2_sum[i] = TotVolV2[:,i].sum()/1000
        TotVolSolXSingle_sum[i] = TotVolSolXSingle[:,i].sum()/1000
        TotVolSolXCombined_sum[i] = TotVolSolXCombined[:,i].sum()/1000
 """


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




## Weighted weeks convertion 

CapNames = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
PriceNames = ['c_FCR','c_aFRR_up','c_aFRR_down','c_mFRRup'] #For V2
DA_namesV2 = ['P_import','P_export','c_DA'] # For V2 

PriceNamesSolX = ['c_FCR','c_aFRR_up','c_aFRRdown','c_mFRR_up'] # For V3
DA_namesV3 = ['P_import','P_export','DA_clearing'] # For V3 
Markets =['FCR','aFRR Up','aFRR Down','mFRR']

def WeeklyConversion(df_weeks,df_RepWeeks,CapNames,DA_names,PriceNames,Markets,start,end,CT,PT):

    weeks = 168*start     # 168 for 2020 and 168*11 for 2021 (in hours)
    end_week = 168*end  # 1680 for 2020 and 3360 for 2021 (in hours)
    prob_week = 0

    
    DA_Tot = np.zeros((len(df_RepWeeks),4))
    Power = np.zeros((len(df_RepWeeks),2))
    ResVol = np.zeros((len(df_RepWeeks),len(Markets)))
    ResRev = np.zeros((len(df_RepWeeks),len(Markets)))

    while weeks <= end_week:

        
        df_w = df_weeks.iloc[(weeks-168):weeks,:]
    
        DA_rev = 0
        DA_cost = 0
        ConT = 0
        ProdT = 0
        P_import = 0
        P_export = 0

        for t in range(0,len(df_w)):
            DA_rev = DA_rev + df_w[DA_names[1]].iloc[t] * df_w[DA_names[2]].iloc[t]
            DA_cost = DA_cost + df_w[DA_names[0]].iloc[t] * df_w[DA_names[2]].iloc[t]
            ConT = ConT + df_w[DA_names[0]].iloc[t] * CT
            ProdT = ProdT + df_w[DA_names[1]].iloc[t] * PT
            P_import = P_import + df_w[DA_names[0]].iloc[t]
            P_export = P_export + df_w[DA_names[1]].iloc[t]


        DA_Tot[prob_week][0] = DA_rev 
        DA_Tot[prob_week][1] = DA_cost 
        DA_Tot[prob_week][2] = ConT 
        DA_Tot[prob_week][3] = ProdT 
        Power[prob_week][0] = P_import 
        Power[prob_week][1] = P_export 
        # Reserve capacity and revenue
        for i in range(0,len(CapNames)): 

            Capacity = 0
            Revenue = 0
            
            for t in range(0,len(df_w)):

                Capacity = Capacity + df_w[CapNames[i]].iloc[t]
                Revenue = Revenue + df_w[CapNames[i]].iloc[t] * df_w[PriceNames[i]].iloc[t]     


            ResVol[prob_week][i] = Capacity 
            ResRev[prob_week][i] = Revenue 
        

        weeks += 168 
        prob_week += 1


    df = pd.DataFrame({'DA Rev [EUR]': DA_Tot[:,0], 
                        'DA Cost [EUR]':DA_Tot[:,1],
                        'ConT [EUR]':DA_Tot[:,2],
                        'ProdT [EUR]':DA_Tot[:,3],
                        'P import [MWh]':Power[:,0],
                        'P export [MWh]':Power[:,1],
                        'FCR Vol [MW]':ResVol[:,0],
                        'aFRR up Vol [MW]':ResVol[:,1],
                        'aFRR down Vol [MW]':ResVol[:,2],
                        'mFRR Vol [MW]': ResVol[:,3],
                        'FCR Rev [EUR]': ResRev[:,0],
                        'aFRR up Rev [EUR]':ResRev[:,1],
                        'aFRR down Rev [EUR]':ResRev[:,2],
                        'mFRR Rev [EUR]': ResRev[:,3]})

    return df


##Plotting Acctepted Capacity 
def AcceptedCapacityPlot(): 

    #Start = 1 & end = 10 for 2020 and start = 11 & end = 20 for 2021
    dfweek_2020V2 = WeeklyConversion(df_V2_weeks,df_RepWeeks2020,CapNames,DA_namesV2,PriceNames,Markets,1,10,CT,PT)
    dfweek_2021V2 = WeeklyConversion(df_V2_weeks,df_RepWeeks2021,CapNames,DA_namesV2,PriceNames,Markets,11,20,CT,PT) 
    dfweek_2020V3_Single = WeeklyConversion(df_SolX_single,df_RepWeeks2020,CapNames,DA_namesV3,PriceNamesSolX,Markets,1,10,CT,PT)
    dfweek_2021V3_Single = WeeklyConversion(df_SolX_single,df_RepWeeks2021,CapNames,DA_namesV3,PriceNamesSolX,Markets,11,20,CT,PT) 
    dfweek_2020V3_Combined = WeeklyConversion(df_SolX_combined,df_RepWeeks2020,CapNames,DA_namesV3,PriceNamesSolX,Markets,1,10,CT,PT)
    dfweek_2021V3_Combined = WeeklyConversion(df_SolX_combined,df_RepWeeks2021,CapNames,DA_namesV3,PriceNamesSolX,Markets,11,20,CT,PT) 

    ## Reserves 2020
    reserves_Cap2020 = np.zeros((4,2)) 
    for i in range(0,len(df_RepWeeks2020)):
        reserves_Cap2020[0,0] += (dfweek_2020V3_Single['FCR Vol [MW]'].iloc[i] / dfweek_2020V2['FCR Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_Cap2020[0,1] += (dfweek_2020V3_Combined['FCR Vol [MW]'].iloc[i] / dfweek_2020V2['FCR Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]

        reserves_Cap2020[1,0] += (dfweek_2020V3_Single['aFRR up Vol [MW]'].iloc[i] / dfweek_2020V2['aFRR up Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_Cap2020[1,1] += (dfweek_2020V3_Combined['aFRR up Vol [MW]'].iloc[i] / dfweek_2020V2['aFRR up Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        
        reserves_Cap2020[2,0] += (dfweek_2020V3_Single['aFRR down Vol [MW]'].iloc[i] / dfweek_2020V2['aFRR down Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_Cap2020[2,1] += (dfweek_2020V3_Combined['aFRR down Vol [MW]'].iloc[i] / dfweek_2020V2['aFRR down Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        
        reserves_Cap2020[3,0] += (dfweek_2020V3_Single['mFRR Vol [MW]'].iloc[i] / dfweek_2020V2['mFRR Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_Cap2020[3,1] += (dfweek_2020V3_Combined['mFRR Vol [MW]'].iloc[i] / dfweek_2020V2['mFRR Vol [MW]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        

    ## Reserves 2021
    reserves_Cap2021 = np.zeros((4,2)) 
    for i in range(0,len(df_RepWeeks2021)):
        reserves_Cap2021[0,0] += (dfweek_2021V3_Single['FCR Vol [MW]'].iloc[i] / dfweek_2021V2['FCR Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_Cap2021[0,1] += (dfweek_2021V3_Combined['FCR Vol [MW]'].iloc[i] / dfweek_2021V2['FCR Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]

        reserves_Cap2021[1,0] += (dfweek_2021V3_Single['aFRR up Vol [MW]'].iloc[i] / dfweek_2021V2['aFRR up Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_Cap2021[1,1] += (dfweek_2021V3_Combined['aFRR up Vol [MW]'].iloc[i] / dfweek_2021V2['aFRR up Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        
        reserves_Cap2021[2,0] += (dfweek_2021V3_Single['aFRR down Vol [MW]'].iloc[i] / dfweek_2021V2['aFRR down Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_Cap2021[2,1] += (dfweek_2021V3_Combined['aFRR down Vol [MW]'].iloc[i] / dfweek_2021V2['aFRR down Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        
        reserves_Cap2021[3,0] += (dfweek_2021V3_Single['mFRR Vol [MW]'].iloc[i] / dfweek_2021V2['mFRR Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_Cap2021[3,1] += (dfweek_2021V3_Combined['mFRR Vol [MW]'].iloc[i] / dfweek_2021V2['mFRR Vol [MW]'].iloc[i])*df_RepWeeks2021.iloc[i,1]


    df_V2_resVol2020 = [df_results2020M2['r_FCR'].sum(),df_results2020M2['r_aFRR_up'].sum(),df_results2020M2['r_aFRR_down'].sum(),df_results2020M2['r_mFRR_up'].sum()]
    df_V2_resVol2021 = [df_results2021M2['r_FCR'].sum(),df_results2021M2['r_aFRR_up'].sum(),df_results2021M2['r_aFRR_down'].sum(),df_results2021M2['r_mFRR_up'].sum()]

    df_V2_resVol2020[:] = [x / 1000 for x in df_V2_resVol2020]
    df_V2_resVol2021[:] = [x / 1000 for x in df_V2_resVol2021]


    df2020 = pd.DataFrame({'Potential': df_V2_resVol2020,
                        'Realized IS': df_V2_resVol2020*reserves_Cap2020[:,0],
                        'Realized DS': df_V2_resVol2020*reserves_Cap2020[:,1]}, index=Markets)

    df2021 = pd.DataFrame({'Potential': df_V2_resVol2021,
                        'Realized IS': df_V2_resVol2021*reserves_Cap2021[:,0],
                        'Realized DS': df_V2_resVol2021*reserves_Cap2021[:,1]}, index=Markets)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    axes[0].grid()
    axes[1].grid()

    df2020.plot.barh(ax=axes[0], color = ['#148710','#1f1485','#de0719'])
    df2021.plot.barh(ax=axes[1],legend=False, color = ['#148710','#1f1485','#de0719'])

    axes[0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.4), ncol=3)
    axes[0].set_title('2020',fontsize=12)
    axes[1].set_title('2021',fontsize=12)
    axes[1].set_xlabel('GW$^*$',fontsize=12)


    plt.tight_layout()
    plt.show()

AcceptedCapacityPlot ()


#Plotting Reserve Revenue
def ReserveRevevenuePlot():

    #Start = 1 & end = 10 for 2020 and start = 11 & end = 20 for 2021
    dfweek_2020V2 = WeeklyConversion(df_V2_weeks,df_RepWeeks2020,CapNames,DA_namesV2,PriceNames,Markets,1,10,CT,PT)
    dfweek_2021V2 = WeeklyConversion(df_V2_weeks,df_RepWeeks2021,CapNames,DA_namesV2,PriceNames,Markets,11,20,CT,PT) 
    dfweek_2020V3_Single = WeeklyConversion(df_SolX_single,df_RepWeeks2020,CapNames,DA_namesV3,PriceNamesSolX,Markets,1,10,CT,PT)
    dfweek_2021V3_Single = WeeklyConversion(df_SolX_single,df_RepWeeks2021,CapNames,DA_namesV3,PriceNamesSolX,Markets,11,20,CT,PT) 
    dfweek_2020V3_Combined = WeeklyConversion(df_SolX_combined,df_RepWeeks2020,CapNames,DA_namesV3,PriceNamesSolX,Markets,1,10,CT,PT)
    dfweek_2021V3_Combined = WeeklyConversion(df_SolX_combined,df_RepWeeks2021,CapNames,DA_namesV3,PriceNamesSolX,Markets,11,20,CT,PT) 


    ## Reserves 2020
    reserves_rev2020 = np.zeros((4,2)) 
    for i in range(0,len(df_RepWeeks2020)):
        reserves_rev2020[0,0] += (dfweek_2020V3_Single['FCR Rev [EUR]'].iloc[i] / dfweek_2020V2['FCR Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_rev2020[0,1] += (dfweek_2020V3_Combined['FCR Rev [EUR]'].iloc[i] / dfweek_2020V2['FCR Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]

        reserves_rev2020[1,0] += (dfweek_2020V3_Single['aFRR up Rev [EUR]'].iloc[i] / dfweek_2020V2['aFRR up Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_rev2020[1,1] += (dfweek_2020V3_Combined['aFRR up Rev [EUR]'].iloc[i] / dfweek_2020V2['aFRR up Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        
        reserves_rev2020[2,0] += (dfweek_2020V3_Single['aFRR down Rev [EUR]'].iloc[i] / dfweek_2020V2['aFRR down Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_rev2020[2,1] += (dfweek_2020V3_Combined['aFRR down Rev [EUR]'].iloc[i] / dfweek_2020V2['aFRR down Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        
        reserves_rev2020[3,0] += (dfweek_2020V3_Single['mFRR Rev [EUR]'].iloc[i] / dfweek_2020V2['mFRR Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        reserves_rev2020[3,1] += (dfweek_2020V3_Combined['mFRR Rev [EUR]'].iloc[i] / dfweek_2020V2['mFRR Rev [EUR]'].iloc[i])*df_RepWeeks2020.iloc[i,1]
        

    ## Reserves 2020
    reserves_rev2021 = np.zeros((4,2)) 
    for i in range(0,len(df_RepWeeks2021)):
        reserves_rev2021[0,0] += (dfweek_2021V3_Single['FCR Rev [EUR]'].iloc[i] / dfweek_2021V2['FCR Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_rev2021[0,1] += (dfweek_2021V3_Combined['FCR Rev [EUR]'].iloc[i] / dfweek_2021V2['FCR Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]

        reserves_rev2021[1,0] += (dfweek_2021V3_Single['aFRR up Rev [EUR]'].iloc[i] / dfweek_2021V2['aFRR up Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_rev2021[1,1] += (dfweek_2021V3_Combined['aFRR up Rev [EUR]'].iloc[i] / dfweek_2021V2['aFRR up Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        
        reserves_rev2021[2,0] += (dfweek_2021V3_Single['aFRR down Rev [EUR]'].iloc[i] / dfweek_2021V2['aFRR down Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_rev2021[2,1] += (dfweek_2021V3_Combined['aFRR down Rev [EUR]'].iloc[i] / dfweek_2021V2['aFRR down Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        
        reserves_rev2021[3,0] += (dfweek_2021V3_Single['mFRR Rev [EUR]'].iloc[i] / dfweek_2021V2['mFRR Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
        reserves_rev2021[3,1] += (dfweek_2021V3_Combined['mFRR Rev [EUR]'].iloc[i] / dfweek_2021V2['mFRR Rev [EUR]'].iloc[i])*df_RepWeeks2021.iloc[i,1]
            

    V2_res_rev2020 = [-df_results[sheet[4]]['vOPEX_FCR'][1],-df_results[sheet[4]]['vOPEX_aFRRup'][1],-df_results[sheet[4]]['vOPEX_aFRRdown'][1],-df_results[sheet[4]]['vOPEX_mFRRup'][1]]
    V2_res_rev2021 = [-df_results[sheet[5]]['vOPEX_FCR'][1],-df_results[sheet[5]]['vOPEX_aFRRup'][1],-df_results[sheet[5]]['vOPEX_aFRRdown'][1],-df_results[sheet[5]]['vOPEX_mFRRup'][1]]


    df2020 = pd.DataFrame({'Potential': V2_res_rev2020,
                        'Realized IS': V2_res_rev2020*reserves_rev2020[:,0],
                        'Realized DS': V2_res_rev2020*reserves_rev2020[:,1]}, index=Markets)

    df2021 = pd.DataFrame({'Potential': V2_res_rev2021,
                        'Realized IS': V2_res_rev2021*reserves_rev2021[:,0],
                        'Realized DS': V2_res_rev2021*reserves_rev2021[:,1]}, index=Markets)



    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    axes[0].grid()
    axes[1].grid()

    df2020.plot.barh(ax=axes[0], color = ['#148710','#1f1485','#de0719'])
    df2021.plot.barh(ax=axes[1],legend=False, color = ['#148710','#1f1485','#de0719'])

    axes[0].set_title('2020',fontsize=12)
    axes[1].set_title('2021',fontsize=12)
    axes[1].set_xlabel('m€',fontsize=12)


    plt.tight_layout()
    plt.show()

ReserveRevevenuePlot()



## Average hourly bid price/vol and acc bid vol and clearing market price

#Function that get the bid price for bids where bid vol is > 0 (That the model wants to bid)

#Setting dates as index 

df_SolX_single = importSolXSingle()
df_SolX_combined = importSolXCombined()


def getBidPriceHourlyAVG(df_SolX_single,df_SolX_combined,df_RepWeeks):

    df_SolX_combined.index = df_SolX_combined['HourDK']
    df_SolX_single.index = df_SolX_single['HourDK']

    hours = 168*10 
    max_hours = 168*20

    names = ['beta_FCR','beta_aFRR_up','beta_aFRR_down','beta_mFRR_up']

    #Split week and weight


    for i in range(0,len(names)):  
        df_SolX_combined.iloc[168*10:168*11][names[i]] * df_RepWeeks.iloc[0,1]
        df_SolX_combined.iloc[168*11:168*12][names[i]]  * df_RepWeeks.iloc[1,1]
        df_SolX_combined.iloc[168*12:168*13][names[i]] * df_RepWeeks.iloc[2,1]
        df_SolX_combined.iloc[168*13:168*14][names[i]] * df_RepWeeks.iloc[3,1]
        df_SolX_combined.iloc[168*14:168*15][names[i]] * df_RepWeeks.iloc[4,1]
        df_SolX_combined.iloc[168*15:168*16][names[i]] * df_RepWeeks.iloc[5,1]
        df_SolX_combined.iloc[168*16:168*17][names[i]] * df_RepWeeks.iloc[6,1]
        df_SolX_combined.iloc[168*17:168*18][names[i]] * df_RepWeeks.iloc[7,1]
        df_SolX_combined.iloc[168*18:168*19][names[i]] * df_RepWeeks.iloc[8,1]
        df_SolX_combined.iloc[168*19:168*20][names[i]] * df_RepWeeks.iloc[9,1]

        df_SolX_single.iloc[168*10:168*11][names[i]] * df_RepWeeks.iloc[0,1]
        df_SolX_single.iloc[168*11:168*12][names[i]]  * df_RepWeeks.iloc[1,1]
        df_SolX_single.iloc[168*12:168*13][names[i]] * df_RepWeeks.iloc[2,1]
        df_SolX_single.iloc[168*13:168*14][names[i]] * df_RepWeeks.iloc[3,1]
        df_SolX_single.iloc[168*14:168*15][names[i]] * df_RepWeeks.iloc[4,1]
        df_SolX_single.iloc[168*15:168*16][names[i]] * df_RepWeeks.iloc[5,1]
        df_SolX_single.iloc[168*16:168*17][names[i]] * df_RepWeeks.iloc[6,1]
        df_SolX_single.iloc[168*17:168*18][names[i]] * df_RepWeeks.iloc[7,1]
        df_SolX_single.iloc[168*18:168*19][names[i]] * df_RepWeeks.iloc[8,1]
        df_SolX_single.iloc[168*19:168*20][names[i]] * df_RepWeeks.iloc[9,1]


    



    #FCR 
    #Combined
    Bid_prices_FCR_validC = df_SolX_combined['b_FCR'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceFCRC = df_SolX_combined[Bid_prices_FCR_validC]['beta_FCR'] 
    #Single
    Bid_prices_FCR_validS = df_SolX_combined['b_FCR'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceFCRS = df_SolX_combined[Bid_prices_FCR_validS]['beta_FCR'] 


    #aFRR up
    # Combined 
    Bid_prices_aFRRup_validC = df_SolX_combined['b_aFRR_up'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceaFRRupC = df_SolX_combined[Bid_prices_aFRRup_validC]['beta_aFRR_up'] 
    #Single
    Bid_prices_aFRRup_validS = df_SolX_single['b_aFRR_up'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceaFRRupS = df_SolX_single[Bid_prices_aFRRup_validS]['beta_aFRR_up'] 

    #aFRR down
    #Combined
    Bid_prices_aFRRdown_validC = df_SolX_combined['b_aFRR_down'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceaFRRdownC = df_SolX_combined[Bid_prices_aFRRdown_validC]['beta_aFRR_down'] 
    #Single
    Bid_prices_aFRRdown_validS = df_SolX_single['b_aFRR_down'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPriceaFRRdownS = df_SolX_single[Bid_prices_aFRRdown_validS]['beta_aFRR_down'] 

    #mFRR 
    #Combined
    Bid_prices_mFRR_validC = df_SolX_combined['b_mFRR_up'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPricemFRRC = df_SolX_combined[Bid_prices_mFRR_validC]['beta_mFRR_up'] 
    #Single
    Bid_prices_mFRR_validS = df_SolX_single['b_mFRR_up'] > 0  #In order to only have bid price for bids where bid vol is > 0 (That the model wants to bid)
    BidPricemFRRS = df_SolX_single[Bid_prices_mFRR_validS]['beta_mFRR_up'] 


    #Getting average hourly values
    BP_FCR_Cavg = BidPriceFCRC.groupby(BidPriceFCRC.index.hour).mean()
    BP_FCR_Savg = BidPriceFCRS.groupby(BidPriceFCRS.index.hour).mean()

    BP_aFRRup_Cavg = BidPriceaFRRupC.groupby(BidPriceaFRRupC.index.hour).mean()
    BP_aFRRup_Savg = BidPriceaFRRupS.groupby(BidPriceaFRRupS.index.hour).mean()

    BP_aFRRdown_Cavg = BidPriceaFRRdownC.groupby(BidPriceaFRRdownC.index.hour).mean()
    BP_aFRRdown_Savg = BidPriceaFRRdownS.groupby(BidPriceaFRRdownS.index.hour).mean()

    BP_mFRR_Cavg = BidPricemFRRC.groupby(BidPricemFRRC.index.hour).mean()
    BP_mFRR_Savg = BidPricemFRRS.groupby(BidPricemFRRS.index.hour).mean()


    return BP_FCR_Cavg, BP_FCR_Savg, BP_aFRRup_Cavg, BP_aFRRup_Savg, BP_aFRRdown_Cavg,BP_aFRRdown_Savg, BP_mFRR_Cavg ,BP_mFRR_Savg




BP_FCR_Cavg, BP_FCR_Savg, BP_aFRRup_Cavg, BP_aFRRup_Savg, BP_aFRRdown_Cavg,BP_aFRRdown_Savg, BP_mFRR_Cavg ,BP_mFRR_Savg = getBidPriceHourlyAVG(df_SolX_single,df_SolX_combined,df_RepWeeks2020)


BP_FCR_Cavg2021, BP_FCR_Savg2021, BP_aFRRup_Cavg2021, BP_aFRRup_Savg2021, BP_aFRRdown_Cavg2021,BP_aFRRdown_Savg2021, BP_mFRR_Cavg2021 ,BP_mFRR_Savg2021 = getBidPriceHourlyAVG(df_SolX_single,df_SolX_combined,df_RepWeeks2021)






df_SolX_combined.index = df_SolX_combined['HourDK']
df_SolX_single.index = df_SolX_single['HourDK']

hours = 0 
max_hours = 168*10

names = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']

#Split week and weight
while hours < max_hours: 

    for i in range(0,len(names)):  
        df_SolX_combined.iloc[168*0:168*1][names[i]] * df_RepWeeks2020.iloc[0,1]
        df_SolX_combined.iloc[168*1:168*2][names[i]]  * df_RepWeeks2020.iloc[1,1]
        df_SolX_combined.iloc[168*2:168*3][names[i]] * df_RepWeeks2020.iloc[2,1]
        df_SolX_combined.iloc[168*3:168*4][names[i]] * df_RepWeeks2020.iloc[3,1]
        df_SolX_combined.iloc[168*4:168*5][names[i]] * df_RepWeeks2020.iloc[4,1]
        df_SolX_combined.iloc[168*5:168*6][names[i]] * df_RepWeeks2020.iloc[5,1]
        df_SolX_combined.iloc[168*6:168*7][names[i]] * df_RepWeeks2020.iloc[6,1]
        df_SolX_combined.iloc[168*7:168*8][names[i]] * df_RepWeeks2020.iloc[7,1]
        df_SolX_combined.iloc[168*8:168*9][names[i]] * df_RepWeeks2020.iloc[8,1]
        df_SolX_combined.iloc[168*9:168*10][names[i]] * df_RepWeeks2020.iloc[9,1]

        df_SolX_single.iloc[168*0:168*1][names[i]] * df_RepWeeks2020.iloc[0,1]
        df_SolX_single.iloc[168*1:168*2][names[i]]  * df_RepWeeks2020.iloc[1,1]
        df_SolX_single.iloc[168*2:168*3][names[i]] * df_RepWeeks2020.iloc[2,1]
        df_SolX_single.iloc[168*3:168*4][names[i]] * df_RepWeeks2020.iloc[3,1]
        df_SolX_single.iloc[168*4:168*5][names[i]] * df_RepWeeks2020.iloc[4,1]
        df_SolX_single.iloc[168*5:168*6][names[i]] * df_RepWeeks2020.iloc[5,1]
        df_SolX_single.iloc[168*6:168*7][names[i]] * df_RepWeeks2020.iloc[6,1]
        df_SolX_single.iloc[168*7:168*8][names[i]] * df_RepWeeks2020.iloc[7,1]
        df_SolX_single.iloc[168*8:168*9][names[i]] * df_RepWeeks2020.iloc[8,1]
        df_SolX_single.iloc[168*9:168*10][names[i]] * df_RepWeeks2020.iloc[9,1]


    hours += 168

#Getting average hourly values
r_FCR_Cavg = df_SolX_combined['r_FCR'].groupby(df_SolX_combined.index.hour).mean()
r_FCR_Savg = df_SolX_single['r_FCR'].groupby(df_SolX_single.index.hour).mean()

r_aFRRup_Cavg = df_SolX_combined['r_FCR'].groupby(df_SolX_combined.index.hour).mean()
r_aFRRup_Savg = df_SolX_single['r_FCR'].groupby(df_SolX_single.index.hour).mean()

r_aFRRdown_Cavg = df_SolX_combined['r_FCR'].groupby(df_SolX_combined.index.hour).mean()
r_aFRRdown_Savg = df_SolX_single['r_FCR'].groupby(df_SolX_single.index.hour).mean()

r_mFRR_Cavg = df_SolX_combined['r_FCR'].groupby(df_SolX_combined.index.hour).mean()
r_mFRR_Savg = df_SolX_single['r_FCR'].groupby(df_SolX_single.index.hour).mean()




fig, axes = plt.subplots(nrows=2, ncols=2)

color = ['#a83232','#3255a8']

axes[0,0].bar(BP_FCR_Cavg.index,BP_FCR_Cavg, color = color[0])

axes[0,1].bar(BP_aFRRup_Cavg.index,BP_aFRRup_Cavg, color = color[0])

axes[1,0].bar(BP_aFRRdown_Cavg.index,BP_aFRRdown_Cavg, color = color[0])

axes[1,1].bar(BP_mFRR_Cavg.index,BP_mFRR_Cavg, color = color[0])

axes[0,0].set_xlim(-1.5,24)
axes[0,1].set_xlim(-1.5,24)
axes[1,0].set_xlim(-1.5,24)
axes[1,1].set_xlim(-1.5,24)
axes[0,0].set_title('FCR',fontsize=12)
axes[0,1].set_title('aFRR Up',fontsize=12)
axes[1,0].set_title('aFRR Down',fontsize=12)
axes[1,1].set_title('mFRR',fontsize=12)


axes[0,0].set_ylabel('€/MW',fontsize=10)
axes[0,1].set_ylabel('€/MW',fontsize=10)
axes[1,0].set_ylabel('€/MW',fontsize=10)
axes[1,1].set_ylabel('€/MW',fontsize=10)

fig.suptitle('2020 Average Bid Price',fontsize=14)
plt.tight_layout()
plt.show()

















#Gettin average clearing price
df_SolX_combined.index = df_SolX_combined['HourDK']
df_SolX_single.index = df_SolX_single['HourDK']

c_FCR = df_SolX_combined['c_FCR']

#BidPriceFCRC.groupby(BidPriceFCRC.index.hour).mean()


#FCR hourly average bidprice 

## Horizontal revenue plot ## 
#BarHRev(df_V2PLot,df_SolX_singlePLot,df_SolX_combinedPLot)


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








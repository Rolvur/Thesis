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


#Calculating LCOMe
file_to_open = Path("DataAnalysis/BASE/") / 'BASE_EconResults_All.xlsx'
DataX = pd.ExcelFile(file_to_open)
df_results = pd.read_excel(file_to_open, sheet_name=DataX.sheet_names)
sheet = list(df_results)

MethProd = 32000000 #kg methanol 
lifetime = 25  #years 


#Model V1k

fOpexV1k = df_results[sheet[0]]['fOPEX_sum'].sum() #2020 & 2021
vOpexV1k2020 = df_results[sheet[0]]['vOPEX_DA_revenue'].sum() + df_results[sheet[0]]['vOPEX_DA_expenses'].sum() + df_results[sheet[0]]['vOPEX_CT'].sum() + df_results[sheet[0]]['vOPEX_PT'].sum() #2020 
vOpexV1k2021 = df_results[sheet[1]]['vOPEX_DA_revenue'].sum() + df_results[sheet[1]]['vOPEX_DA_expenses'].sum() + df_results[sheet[1]]['vOPEX_CT'].sum() + df_results[sheet[1]]['vOPEX_PT'].sum() #2021 
CAPEXV1k = df_results[sheet[0]]['CAPEX_sum'][0]

LCOMeV1k2020 = 10**6*(fOpexV1k + vOpexV1k2020 + CAPEXV1k)/(MethProd*lifetime)
LCOMeV1k2021 = 10**6*(fOpexV1k + vOpexV1k2021 + CAPEXV1k)/(MethProd*lifetime)



#Model V1pw

fOpexV1pw = df_results[sheet[2]]['fOPEX_sum'].sum() #2020 & 2021
vOpexV12020pw = df_results[sheet[2]]['vOPEX_DA_revenue'].sum() + df_results[sheet[2]]['vOPEX_DA_expenses'].sum() + df_results[sheet[2]]['vOPEX_CT'].sum() + df_results[sheet[2]]['vOPEX_PT'].sum() #2020
vOpexV12021pw = df_results[sheet[3]]['vOPEX_DA_revenue'].sum() + df_results[sheet[3]]['vOPEX_DA_expenses'].sum() + df_results[sheet[3]]['vOPEX_CT'].sum() + df_results[sheet[3]]['vOPEX_PT'].sum() #2021

CAPEXV1pw = df_results[sheet[2]]['CAPEX_sum'][0]

LCOMeV1pw2020 = 10**6*(fOpexV1pw + vOpexV12020pw + CAPEXV1pw)/(MethProd*lifetime)
LCOMeV1pw2021 = 10**6*(fOpexV1pw + vOpexV12021pw + CAPEXV1pw)/(MethProd*lifetime)


#Model V2 

fOpexV2pw = df_results[sheet[4]]['fOPEX_sum'].sum() #2020 & 2021
vOpexV2pw2020 = df_results[sheet[4]]['vOPEX_DA_revenue'].sum() + df_results[sheet[4]]['vOPEX_DA_expenses'].sum() + df_results[sheet[4]]['vOPEX_CT'].sum() + df_results[sheet[4]]['vOPEX_PT'].sum() + df_results[sheet[4]]['vOPEX_FCR'].sum() + df_results[sheet[4]]['vOPEX_aFRRup'].sum() + df_results[sheet[4]]['vOPEX_aFRRdown'].sum()+ df_results[sheet[4]]['vOPEX_mFRRup'].sum() #2020 & 2021
vOpexV2pw2021 = df_results[sheet[5]]['vOPEX_DA_revenue'].sum() + df_results[sheet[5]]['vOPEX_DA_expenses'].sum() + df_results[sheet[5]]['vOPEX_CT'].sum() + df_results[sheet[5]]['vOPEX_PT'].sum() + df_results[sheet[5]]['vOPEX_FCR'].sum() + df_results[sheet[5]]['vOPEX_aFRRup'].sum() + df_results[sheet[5]]['vOPEX_aFRRdown'].sum()+ df_results[sheet[5]]['vOPEX_mFRRup'].sum() #2020 & 2021


CAPEXV1pw = df_results[sheet[4]]['CAPEX_sum'][0]

LCOMeV2_2020 = 10**6*(fOpexV2pw + vOpexV2pw2020 + CAPEXV1pw)/(MethProd*lifetime)
LCOMeV2_2021 = 10**6*(fOpexV2pw + vOpexV2pw2021 + CAPEXV1pw)/(MethProd*lifetime)








## M1 and M2 cost breakdown Plot ## 



#M1 k and M1 pw cost breakdown 

def M1kandM1pwCostB(df_results):


    DA_revM1_2020k = -(df_results[sheet[0]]['vOPEX_DA_revenue'][1])
    DA_revM1_2021k = -(df_results[sheet[1]]['vOPEX_DA_revenue'][1])

    DA_CostM1_2020k = -(df_results[sheet[0]]['vOPEX_DA_expenses'][1])
    DA_CostM1_2021k = -(df_results[sheet[1]]['vOPEX_DA_expenses'][1])

    CT_2020k = -df_results[sheet[0]]['vOPEX_CT'][1] 
    CT_2021k = -df_results[sheet[1]]['vOPEX_CT'][1] 
    
    PT_2020k = -df_results[sheet[0]]['vOPEX_PT'][1]
    PT_2021k = -df_results[sheet[1]]['vOPEX_PT'][1]


    ProfitM1_2020k = DA_revM1_2020k - (-DA_CostM1_2020k - CT_2020k - PT_2020k) 
    ProfitM1_2021k = DA_revM1_2021k - (-DA_CostM1_2021k - CT_2021k - PT_2021k)


    DA_revM1_2020pw = -(df_results[sheet[2]]['vOPEX_DA_revenue'][1])
    DA_revM1_2021pw = -(df_results[sheet[3]]['vOPEX_DA_revenue'][1])

    DA_CostM1_2020pw = -(df_results[sheet[2]]['vOPEX_DA_expenses'][1])
    DA_CostM1_2021pw = -(df_results[sheet[3]]['vOPEX_DA_expenses'][1])

    CT_2020pw = -df_results[sheet[2]]['vOPEX_CT'][1] 
    CT_2021pw = -df_results[sheet[3]]['vOPEX_CT'][1] 
    
    PT_2020pw = -df_results[sheet[2]]['vOPEX_PT'][1]
    PT_2021pw = -df_results[sheet[3]]['vOPEX_PT'][1]

    ProfitM1_2020pw = DA_revM1_2020pw - (-DA_CostM1_2020pw - CT_2020pw - PT_2020pw ) 
    ProfitM1_2021pw = DA_revM1_2021pw - (-DA_CostM1_2021pw - CT_2021pw - PT_2021pw)


    x = ['2020','2021']
    fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True,gridspec_kw={'width_ratios': [1, 1]})

    var1k = np.array([PT_2020k,PT_2021k]) + np.array([DA_CostM1_2020k,DA_CostM1_2021k])
    var1pw = np.array([PT_2020pw,PT_2021pw]) + np.array([DA_CostM1_2020pw,DA_CostM1_2021pw])
    ## Model 1 constant efficiency 
    ax1.bar(x, [DA_revM1_2020k,DA_revM1_2021k], color='#0a940a',linestyle = 'solid', label ='DA Revenue',width=0.75)

    ax1.bar(x , [CT_2020k,CT_2021k], color='#8a0a0a',bottom=var1k ,linestyle = '-', label ='Consumer T',width=0.75)
    ax1.bar(x , [PT_2020k,PT_2021k], bottom= [DA_CostM1_2020k,DA_CostM1_2021k],  color='#1a0606',linestyle = '-', label ='Producer T',width=0.75)
    ax1.bar(x , [DA_CostM1_2020k,DA_CostM1_2021k], color='#d90f0f',linestyle = '-', label ='DA Cost',width=0.75)

    ax1.plot(x, [ProfitM1_2020k,ProfitM1_2021k], color='navy', label='Profit k',linestyle='dashed',linewidth=3.0, marker='o')



    ax2.bar(x, [DA_revM1_2020pw,DA_revM1_2021pw], color='#0a940a',linestyle = 'solid', label ='DA Revenue',width=0.75)

    ax2.bar(x , [DA_CostM1_2020pw,DA_CostM1_2021pw], color='#d90f0f',linestyle = '-', label ='DA Cost',width=0.75) 
    ax2.bar(x , [PT_2020pw,PT_2021pw], bottom= [DA_CostM1_2020pw,DA_CostM1_2021pw],  color='#1a0606',linestyle = '-', label ='Producer T',width=0.75)
    ax2.bar(x , [CT_2020pw,CT_2021pw], color='#8a0a0a',bottom=var1pw ,linestyle = '-', label ='Consumer T',width=0.75)

    ax2.plot(x, [ProfitM1_2020k,ProfitM1_2021k], color='navy', label='Profit k',linewidth=3.0,linestyle='dashed', marker='o')
    ax2.plot(x, [ProfitM1_2020pw,ProfitM1_2021pw], color='#7d7d80', label='Profit pw',linewidth=3.0,linestyle='dashed', marker='o')

    ax1.set_ylabel('mio €')
    ax2.legend(loc='upper left',bbox_to_anchor=(1.05, 1))

    #ax1.set_ylim([0,15])

    ax1.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()


    return ProfitM1_2020k, ProfitM1_2021k,ProfitM1_2020pw, ProfitM1_2021pw 

a = M1kandM1pwCostB(df_results)


def M1andM2CostBreak(df_results,sheet):
    #Model 1 

    DA_revM1_2020 = -(df_results[sheet[0]]['vOPEX_DA_revenue'][1])
    DA_revM1_2021 = -(df_results[sheet[1]]['vOPEX_DA_revenue'][1])

    DA_CostM1_2020 = -(df_results[sheet[0]]['vOPEX_DA_expenses'][1])
    DA_CostM1_2021 = -(df_results[sheet[1]]['vOPEX_DA_expenses'][1])

    CT_2020M1 = -df_results[sheet[0]]['vOPEX_CT'][1] 
    CT_2021M1 = -df_results[sheet[1]]['vOPEX_CT'][1] 
    
    PT_2020M1 = -df_results[sheet[0]]['vOPEX_PT'][1]
    PT_2021M1 = -df_results[sheet[1]]['vOPEX_PT'][1]




    ProfitM1_2020 = DA_revM1_2020 - (-DA_CostM1_2020-CT_2020M1-PT_2020M1) 
    ProfitM1_2021 = DA_revM1_2021 - (-DA_CostM1_2021-CT_2021M1-PT_2021M1)


    #Model 2 
    #DA
    DA_revM2_2020 = -df_results[sheet[4]]['vOPEX_DA_revenue'][1] 
    DA_revM2_2021 = -df_results[sheet[5]]['vOPEX_DA_revenue'][1]
    DA_CostM2_2020 = -df_results[sheet[4]]['vOPEX_DA_expenses'][1]
    DA_CostM2_2021 = -df_results[sheet[5]]['vOPEX_DA_expenses'][1] 

    CT_2020 = -df_results[sheet[4]]['vOPEX_CT'][1] 
    CT_2021 = -df_results[sheet[5]]['vOPEX_CT'][1] 
    
    PT_2020 = -df_results[sheet[4]]['vOPEX_PT'][1]
    PT_2021 = -df_results[sheet[5]]['vOPEX_PT'][1]
    



    #aFRR 
    aFRRupM2_2020 = -df_results[sheet[4]]['vOPEX_aFRRup'][1]
    aFRRupM2_2021 = -df_results[sheet[5]]['vOPEX_aFRRup'][1]
    aFRRdownM2_2020 = -df_results[sheet[4]]['vOPEX_aFRRdown'][1]
    aFRRdownM2_2021 = -df_results[sheet[5]]['vOPEX_aFRRdown'][1]

    #FCR 
    FCRM2_2020 = -df_results[sheet[4]]['vOPEX_FCR'][1]
    FCRM2_2021 = -df_results[sheet[5]]['vOPEX_FCR'][1]


    #mFRR 
    mFRRupM2_2020 = -df_results[sheet[4]]['vOPEX_mFRRup'][1]
    mFRRupM2_2021 = -df_results[sheet[5]]['vOPEX_mFRRup'][1]



    ProfitM2_2020 = DA_revM2_2020 + FCRM2_2020 + aFRRupM2_2020 + aFRRdownM2_2020 + mFRRupM2_2020 - (-DA_CostM2_2020 - CT_2020 - PT_2020)
    ProfitM2_2021 = DA_revM2_2021 + FCRM2_2021 + aFRRupM2_2021 + aFRRdownM2_2021 + mFRRupM2_2021 - (-DA_CostM2_2021 - CT_2021 - PT_2021)


    x = ['2020','2021']
    fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True,gridspec_kw={'width_ratios': [1, 1]})


    ## Model 1 

    var5 = np.array([PT_2020M1,PT_2021M1]) + np.array([DA_CostM1_2020,DA_CostM1_2021])

    ax1.bar(x, [DA_revM1_2020,DA_revM1_2021], color='#0a940a',linestyle = 'solid', label ='DA Revenue',width=0.75)
    
    ax1.bar(x , [DA_CostM1_2020,DA_CostM1_2021], color='#d90f0f',linestyle = '-', label ='DA Cost',width=0.75)
    ax1.bar(x , [PT_2020M1,PT_2021M1], bottom= [DA_CostM1_2020,DA_CostM1_2021],  color='#1a0606',linestyle = '-', label ='Producer T',width=0.75)
    ax1.bar(x , [CT_2020M1,CT_2021M1], color='#8a0a0a',bottom=var5 ,linestyle = '-', label ='Consumer T',width=0.75)

    ax1.plot(x, [ProfitM1_2020,ProfitM1_2021], color='navy', label='Profit',linewidth=3.0,linestyle='dashed', marker='o')

    ax1.set_ylabel('mio €')
    #ax1.legend(loc='upper left',bbox_to_anchor=(1.05, 1))

    #ax1.set_ylim([0,15])

    ax1.tick_params(axis='x', rotation=0)


    ## Model 2
    DA_Rev = np.array([DA_revM2_2020,DA_revM2_2021])
    DA_Cost = np.array([DA_CostM2_2020,DA_CostM2_2021])
    FCR_Rev = np.array([FCRM2_2020,FCRM2_2021])
    aFRR_up = np.array([aFRRupM2_2020, aFRRupM2_2021])
    aFRR_down = np.array([aFRRdownM2_2020, aFRRdownM2_2021])
    mFRR = np.array([mFRRupM2_2020,mFRRupM2_2021])

    var1 = DA_Rev + FCR_Rev
    var2 = var1 + aFRR_up 
    var3 = aFRR_down + var2
    var4 = np.array([PT_2020,PT_2021]) + DA_Cost


    ax2.bar(x, mFRR, bottom=var3, color='goldenrod',linestyle = 'solid', label ='mFRR Up',width=0.75)
    ax2.bar(x, aFRR_down, bottom=var2, color='navy',linestyle = 'solid', label ='aFRR Down',width=0.75)
    ax2.bar(x, aFRR_up, bottom=var1, color='royalblue',linestyle = 'solid', label ='aFRR Up',width=0.75)
    ax2.bar(x, FCR_Rev,bottom= DA_Rev ,color='darkorange',linestyle = 'solid', label ='FCR',width=0.75)
    ax2.bar(x, DA_Rev, color='#0a940a',linestyle = 'solid', label ='DA Revenue',width=0.75)
   

    ax2.bar(x , DA_Cost, color='#d90f0f',linestyle = 'solid', label ='DA Cost',width=0.75)  
    ax2.bar(x , [PT_2020,PT_2021], bottom= DA_Cost,  color='#1a0606',linestyle = '-', label ='Producer T',width=0.75)
    ax2.bar(x , [CT_2020,CT_2021], color='#8a0a0a',bottom=var4 ,linestyle = '-', label ='Consumer T',width=0.75)

    ax2.plot(x, [ProfitM2_2020,ProfitM2_2021], color='navy', label='Profit',linewidth=3.0,linestyle='dashed', marker='o')



    #ax2.set_ylabel('mio €')
    ax2.legend(loc='upper left',bbox_to_anchor=(1.05, 1))

    #ax2.set_ylim([0,15])

    ax2.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.show()  

    return DA_Rev, DA_Cost, FCR_Rev, aFRR_up, aFRR_down, mFRR, ProfitM1_2020,ProfitM1_2021,ProfitM2_2020,ProfitM2_2021

DA_Rev, DA_Cost, FCR_Rev, aFRR_up, aFRR_down, mFRR, ProfitM1_2020, ProfitM1_2021, ProfitM2_2020, ProfitM2_2021 = M1andM2CostBreak(df_results,sheet)



## Power consumption breakdown ## 

file_to_open = Path("Result_files/") / 'V1_year_2020-01-01_2020-12-31_k.xlsx'
df_resultsM1_2020 = pd.read_excel(file_to_open)
file_to_open = Path("Result_files/") / 'V1_year_2021-01-01_2021-12-31_k.xlsx'
df_resultsM1_2021 = pd.read_excel(file_to_open)

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

PowerCon(df_resultsM1_2020,df_resultsM1_2021)



#Net import M1 vs M2 PLOT !! 
def NetPowerImport():

    file_to_open = Path("Result_files/") / 'V1_year_2020-01-01_2020-12-31_k.xlsx'
    df_resultsM1_2020 = pd.read_excel(file_to_open)
    file_to_open = Path("Result_files/") / 'V1_year_2021-01-01_2021-12-31_k.xlsx'
    df_resultsM1_2021 = pd.read_excel(file_to_open)
    file_to_open = Path("Result_files/") / 'V2_year_2020-01-01_2020-12-31.xlsx'
    df_resultsM2_2020 = pd.read_excel(file_to_open)
    file_to_open = Path("Result_files/") / 'V2_year_2021-01-01_2021-12-31.xlsx'
    df_resultsM2_2021 = pd.read_excel(file_to_open)


    x = np.arange(0,2,1)
    width = 0.3 
    fig , ax = plt.subplots(nrows=1,ncols=1)


    ax.bar(x, [df_resultsM1_2020['P_grid'].sum(),df_resultsM1_2021['P_grid'].sum()], color ='#148710', label = 'Model 1',width=width-0.05 )
    ax.bar(x+width, [df_resultsM2_2020['P_grid'].sum(),df_resultsM2_2021['P_grid'].sum()], color ='#1f1485', label = 'Model 2',width=width-0.05)


    ax.set_ylabel('MWh')
    ax.legend(loc='upper left',bbox_to_anchor=(1.05, 1))

    #ax1.set_ylim([0,15])

    ax.tick_params(axis='x', rotation=0)
    plt.xticks(x + width / 2, ('2020', '2021'))
    plt.tight_layout()
    plt.show()

NetPowerImport()

## Pie BTM, Export and Import  
def PlotPieCon():
    ### System power balance Pie chart ##
    file_to_open = Path("Result_files/") / 'V1_year_2020-01-01_2020-12-31_k.xlsx'
    df_results2020M1 = pd.read_excel(file_to_open)
    file_to_open = Path("Result_files/") / 'V1_year_2021-01-01_2021-12-31_k.xlsx'
    df_results2021M1 = pd.read_excel(file_to_open)
    file_to_open = Path("Result_files/") / 'V2_year_2020-01-01_2020-12-31.xlsx'
    df_results2020M2 = pd.read_excel(file_to_open)
    file_to_open = Path("Result_files/") / 'V2_year_2021-01-01_2021-12-31.xlsx'
    df_results2021M2 = pd.read_excel(file_to_open)


    #Calculating export [MW]
    def CalcPowerFlow(df_results):
        export = 0
        PV_PEM = 0
        Import = 0
        for i in range(0,len(df_results)):
            if df_results['P_grid'][i] < 0:
                export = export + df_results['P_grid'].iloc[i]

            if df_results['P_grid'].iloc[i] <= 0: 
                PV_PEM = PV_PEM + df_results['P_PV'].iloc[i] + df_results['P_grid'].iloc[i]

            if df_results['P_grid'].iloc[i] > 0 and df_results['P_PV'].iloc[i] > 0: 
                PV_PEM = PV_PEM + df_results['P_PV'].iloc[i]

            if df_results['P_grid'].iloc[i] > 0: 
                Import = Import + df_results['P_grid'].iloc[i] 

            Tot_Power = [PV_PEM,-export,Import]
        return Tot_Power

    Power2020V1 = np.array(CalcPowerFlow(df_results2020M1))
    Power2021V1 = np.array(CalcPowerFlow(df_results2021M1))
    Power2020V2 = np.array(CalcPowerFlow(df_results2020M2))
    Power2021V2 = np.array(CalcPowerFlow(df_results2021M2))


    #values = [round(num)/1000 for num in Power2020V1]

    def absolute_value2020(val):
        a  = np.round((val/100.*Power2020V1.sum())/1000, 1)
        print(a)
        return a

    def absolute_value2021(val):
            a  = np.round((val/100.*Power2021V1.sum())/1000, 1)
            print(a)
            return a


    colors = ['#0fdb20','#0b3de0','#e0190b']
    explode = [0,0,0]


    fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    ax1.pie(Power2020V1,labels=['BTM','Export','Import'],colors=colors,explode=explode,shadow=True,startangle=0,wedgeprops={'edgecolor':'black'},autopct=absolute_value2020)
    ax2.pie(Power2021V1,labels=['BTM','Export','Import'],colors=colors,explode=explode,shadow=True,startangle=0,wedgeprops={'edgecolor':'black'},autopct=absolute_value2021)

    ax1.set_title('2020 [GWh]')
    ax2.set_title('2021 [GWh]')

    plt.tight_layout()
    plt.show()
PlotPieCon()


file_to_open = Path("Result_files/") / 'V1_year_2020-01-01_2020-12-31_k.xlsx'
df_resultsM1_2020 = pd.read_excel(file_to_open)
file_to_open = Path("Result_files/") / 'V1_year_2021-01-01_2021-12-31_k.xlsx'
df_resultsM1_2021 = pd.read_excel(file_to_open)
file_to_open = Path("Result_files/") / 'V2_year_2020-01-01_2020-12-31.xlsx'
df_resultsM2_2020 = pd.read_excel(file_to_open)
file_to_open = Path("Result_files/") / 'V2_year_2021-01-01_2021-12-31.xlsx'
df_resultsM2_2021 = pd.read_excel(file_to_open)

df_resultsM1 = pd.concat([df_resultsM1_2020 , df_resultsM1_2021],ignore_index=True)
df_resultsM2 = pd.concat([df_resultsM2_2020 , df_resultsM2_2021],ignore_index=True)

P_sys = np.zeros((len(df_resultsM2),2))   #[PV,Grid]

for i in range(0,len(df_resultsM2)):

    if df_resultsM2['P_grid'][i] < 0: #Exporting power
        P_sys[i,0] = df_resultsM2['P_PV'][i] + df_resultsM2['P_grid'][i]


    if df_resultsM2['P_grid'][i] >= 0: #Imporitn power 
        P_sys[i,0] = df_resultsM2['P_PV'][i]
        P_sys[i,1] = df_resultsM2['P_grid'][i]



df_Psys = pd.DataFrame({'PV': P_sys[:,0], 'Grid': P_sys[:,1], 'P_sysTot': P_sys[:,0]+P_sys[:,1], 'P_PEM': df_resultsM2['P_PEM'].tolist()}, index=df_resultsM2['HourDK'])

#demand_avg  = df_resultsM1_2020.groupby(pd.PeriodIndex(df_resultsM1_2020['HourDK'], freq="M"))['Demand'].mean()


df_Psys_avg = df_Psys.groupby(pd.PeriodIndex(df_Psys.index, freq="M"))['PV','Grid','P_sysTot','P_PEM'].mean()


df_Psys_avgM1 = df_Psys_avg
df_Psys_avgM2 = df_Psys_avg


fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)

x = df_Psys_avg.index

x = x.astype(str)

ax1.bar(x, df_Psys_avgM1['Grid'], label = 'Grid', color= 'teal') 
ax1.bar(x, df_Psys_avgM1['PV'] ,bottom = df_Psys_avgM1['Grid'], label='PV', color= 'olivedrab') 
ax1.plot(x, df_Psys_avgM1['P_PEM'], color='red', label='PEM',linestyle='solid', marker='o')

ax2.bar(x, df_Psys_avgM2['Grid'], label = 'Grid', color= 'teal') 
ax2.bar(x, df_Psys_avgM2['PV'] ,bottom = df_Psys_avgM2['Grid'], label='PV', color= 'olivedrab') 
ax2.plot(x, df_Psys_avgM2['P_PEM'], color='red', label='PEM',linestyle='solid', marker='o')




ax2.tick_params(axis='x', rotation=45)
ax2.set_ylabel('MW')
#plt.legend()
ax1.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
ax1.set_ylabel('MW')
loc = plticker.MultipleLocator(base=2.0) # this locator puts ticks at regular intervals
ax2.xaxis.set_major_locator(loc)
#ax.set_ylim([0, 57])
ax1.set_title('Model 1')
ax2.set_title('Model 2')
plt.tight_layout()
plt.show()
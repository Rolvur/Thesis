import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as md
from statistics import mean
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from Opt_Constants import PT,CT
#--______________________DATA_IMPORT------------------------------------------------
#Calculating LCOMe
file_to_open = Path("DataAnalysis/BASE/") / 'BASE_EconResults_All.xlsx'
DataX = pd.ExcelFile(file_to_open)
df_results = pd.read_excel(file_to_open, sheet_name=DataX.sheet_names)
sheet = list(df_results)

#Import CAPEX and fixed OPEX values 
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")

fOPEX_V2_yearly = df_results[sheet[4]]['fOPEX_sum'][1]
fOpexV2 = df_results[sheet[4]]['fOPEX_sum'].sum() #2020 & 2021
vOPEX_BASE_2020 = df_results[sheet[4]]['vOPEX_DA_revenue'][1] + df_results[sheet[4]]['vOPEX_DA_expenses'][1] + df_results[sheet[4]]['vOPEX_CT'][1] + df_results[sheet[4]]['vOPEX_PT'][1] + df_results[sheet[4]]['vOPEX_FCR'][1] + df_results[sheet[4]]['vOPEX_aFRRup'][1] + df_results[sheet[4]]['vOPEX_aFRRdown'][1]+ df_results[sheet[4]]['vOPEX_mFRRup'][1] #2020 & 2021

vOPEX_BASE_2021 = df_results[sheet[5]]['vOPEX_DA_revenue'][1] + df_results[sheet[5]]['vOPEX_DA_expenses'][1] + df_results[sheet[5]]['vOPEX_CT'][1] + df_results[sheet[5]]['vOPEX_PT'][1] + df_results[sheet[5]]['vOPEX_FCR'][1] + df_results[sheet[5]]['vOPEX_aFRRup'][1] + df_results[sheet[5]]['vOPEX_aFRRdown'][1]+ df_results[sheet[5]]['vOPEX_mFRRup'][1] #2020 & 2021


CAPEX_PV = dfEconParam['PV_CAPEX'].iloc[0]
CAPEX_PEM = dfEconParam['PEM_CAPEX'].iloc[0]
CAPEX_METHANOL = dfEconParam['METHANOL_CAPEX'].iloc[0]
CAPEX_CO2 = dfEconParam['CO2_CAPEX'].iloc[0]
CAPEX_GRID = dfEconParam['Grid_Connection'].iloc[0]
CAPEX_BASE = CAPEX_PV+CAPEX_PEM+CAPEX_METHANOL+CAPEX_CO2+CAPEX_GRID

fOPEX_PV = dfEconParam['PV_OPEX'].iloc[0]
fOPEX_PEM = dfEconParam['PEM_OPEX'].iloc[0]
fOPEX_METHANOL = dfEconParam['METHANOL_OPEX'].iloc[0]
fOPEX_CO2 = dfEconParam['CO2_OPEX'].iloc[0]
fOPEX_BASE = fOPEX_PV+fOPEX_PEM+fOPEX_METHANOL+fOPEX_CO2 



def create_CAPEX_factors():
    factors = [ [0.80,1,1,1,1],
                [0.85,1,1,1,1],
                [0.9,1,1,1,1],
                [0.95,1,1,1,1],
                [1,1,1,1,1],
                [1.05,1,1,1,1],
                [1.1,1,1,1,1],
                [1.15,1,1,1,1],
                [1.20,1,1,1,1],
                [1,0.80,1,1,1],
                [1,0.85,1,1,1],
                [1,0.9,1,1,1],
                [1,0.95,1,1,1],
                [1,1,1.,1,1],
                [1,1.05,1,1,1],
                [1,1.1,1,1,1],
                [1,1.15,1,1,1],
                [1,1.20,1,1,1],
                [1,1,0.8,1,1],
                [1,1,0.85,1,1],
                [1,1,0.9,1,1],
                [1,1,0.95,1,1],
                [1,1,1,1,1],
                [1,1,1.05,1,1],
                [1,1,1.1,1,1],
                [1,1,1.15,1,1],
                [1,1,1.20,1,1],
                [1,1,1,0.80,1],
                [1,1,1,0.85,1],
                [1,1,1,0.9,1],
                [1,1,1,0.95,1],
                [1,1,1,1,1],
                [1,1,1,1.05,1],
                [1,1,1,1.1,1],
                [1,1,1,1.15,1],
                [1,1,1,1.20,1],
                [1,1,1,1,0.80],
                [1,1,1,1,0.85],
                [1,1,1,1,0.9],
                [1,1,1,1,0.95],
                [1,1,1,1,1],
                [1,1,1,1,1.05],
                [1,1,1,1,1.1],
                [1,1,1,1,1.15],
                [1,1,1,1,1.20]]
    return factors

def create_fOPEX_factors():
    factors = [ [0.80,1,1,1],
                [0.85,1,1,1],
                [0.9,1,1,1],
                [0.95,1,1,1],
                [1,1,1,1],
                [1.05,1,1,1],
                [1.1,1,1,1],
                [1.15,1,1,1],
                [1.20,1,1,1],
                [1,0.80,1,1],
                [1,0.85,1,1],
                [1,0.9,1,1],
                [1,0.95,1,1],
                [1,1,1.,1],
                [1,1.05,1,1],
                [1,1.1,1,1],
                [1,1.15,1,1],
                [1,1.20,1,1],
                [1,1,0.8,1],
                [1,1,0.85,1],
                [1,1,0.9,1],
                [1,1,0.95,1],
                [1,1,1,1],
                [1,1,1.05,1],
                [1,1,1.1,1],
                [1,1,1.15,1],
                [1,1,1.20,1],
                [1,1,1,0.80],
                [1,1,1,0.85],
                [1,1,1,0.9],
                [1,1,1,0.95],
                [1,1,1,1],
                [1,1,1,1.05],
                [1,1,1,1.1],
                [1,1,1,1.15],
                [1,1,1,1.20]]
    return factors

def EX_calc(dict,factors):
    factored_capex = {}
    for i in range(0,len(factors)):
        factored_capex[list(dict.keys())[i]] = dict[list(dict.keys())[i]]*factors[i]
    return factored_capex


def calc_CAPEX_variation(factors,dict_CAPEX):
    dict_CAPEXs = {}
    for i in range(0,len(dict_CAPEX)): #5 CAPEX values: PV PEM, Methanol, CO2, grid
        dict_CAPEXs[list(dict_CAPEX.keys())[i]]={}
        for j in range(0,int(len(factors)/5)): #0.9, 0.95, 1,0, 1.05, 1,1
            print(j)
            dict_CAPEXs[list(dict_CAPEX.keys())[i]][str(round(0.8+0.05*j,2))] = EX_calc(dict_CAPEX,factors[i*9+j])
    return dict_CAPEXs

def calc_fOPEX_variation(factors,dict_fOPEX):
    dict_fOPEXs = {}
    for i in range(0,len(dict_fOPEX)): #5 CAPEX values: PV PEM, Methanol, CO2, grid
        dict_fOPEXs[list(dict_fOPEX.keys())[i]]={}
        for j in range(0,int(len(factors)/4)): #0.9, 0.95, 1,0, 1.05, 1,1
            print(j)
            dict_fOPEXs[list(dict_fOPEX.keys())[i]][str(round(0.8+0.05*j,2))] = EX_calc(dict_fOPEX,factors[i*9+j])
    return dict_fOPEXs


def calc_LC_from_CAPEX(dict,fOPEX_BASE,vOPEX_BASE,WACC,lifetime):
    LCOMe = {}
    for i in range(0,len(dict)):
        component = list(dict.keys())[i]
        for j in range(0,len( dict[component])):
            percentage =list(dict[component].keys())[j]
            CAPEX = sum(CAPEXs[component][percentage].values())
            LCOMe[component+'_'+percentage] = (CAPEX + sum((fOPEX_BASE+vOPEX_BASE)/(1+WACC)**t for t in range(1,lifetime+1)))/sum(32/(1+WACC)**t for t in range(1,lifetime+1))
    return LCOMe

def calc_LC_from_fOPEX(dict,CAPEX_BASE,vOPEX_BASE,WACC,lifetime):
    LCOMe = {}
    for i in range(0,len(dict)):
        component = list(dict.keys())[i]
        for j in range(0,len( dict[component])):
            percentage =list(dict[component].keys())[j]
            fOPEX = sum(dict[component][percentage].values())
            print(fOPEX)
            LCOMe[component+'_'+percentage] = (CAPEX_BASE + sum((fOPEX+vOPEX_BASE)/(1+WACC)**t for t in range(1,lifetime+1)))/sum(32/(1+WACC)**t for t in range(1,lifetime+1))
    return LCOMe


dict_CAPEX = {  'PV plant':         CAPEX_PV,
                'Electrolyzer':     CAPEX_PEM,
                'Methanol plant':   CAPEX_METHANOL,
                'CO2 storage':      CAPEX_CO2,
                'Grid Connection':  CAPEX_GRID
                }
sum(dict_CAPEX.values())


lifetime = dfEconParam['lifetime'][0]
WACC = dfEconParam['discount rate'][0]
D = 32000000 #kg methanol 
factors = create_CAPEX_factors()
CAPEXs = calc_CAPEX_variation(factors,dict_CAPEX)
LCOMe_2020c=calc_LC_from_CAPEX(CAPEXs,fOPEX_BASE,vOPEX_BASE_2020,WACC,lifetime)
LCOMe_2021c=calc_LC_from_CAPEX(CAPEXs,fOPEX_BASE,vOPEX_BASE_2021,WACC,lifetime)




# PLOT CAPEX Sensitivity - Levelized Cost as function of CAPEX Change -
cPV_2020=list(LCOMe_2020c.values())[0:9]
cPEM_2020=list(LCOMe_2020c.values())[9:18]
cMETH_2020=list(LCOMe_2020c.values())[18:27]
cCO2_2020=list(LCOMe_2020c.values())[27:36]
cGRID_2020=list(LCOMe_2020c.values())[36:45]
cPV_2021=list(LCOMe_2021c.values())[0:9]
cPEM_2021=list(LCOMe_2021c.values())[9:18]
cMETH_2021=list(LCOMe_2021c.values())[18:27]
cCO2_2021=list(LCOMe_2021c.values())[27:36]
cGRID_2021=list(LCOMe_2021c.values())[36:45]


xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.grid()
ax1.plot(xticks, cPV_2020, label= 'PV', color='darkorange')
ax1.plot(xticks, cPEM_2020, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, cMETH_2020, label= 'methanol plant', color='maroon')
ax1.plot(xticks, cCO2_2020, label= 'CO2 storage' , color='goldenrod')
ax1.plot(xticks, cGRID, label= 'grid connection' , color='navy')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/kg')
plt.tight_layout()
plt.show()

# Change in €/tonne
cdPV_2020=(list(LCOMe_2020c.values())[0:9]-list(LCOMe_2020c.values())[4])*1000
cdPEM_2020=(list(LCOMe_2020c.values())[9:18]-list(LCOMe_2020c.values())[13])*1000
cdMETH_2020=(list(LCOMe_2020c.values())[18:27]-list(LCOMe_2020c.values())[22])*1000
cdCO2_2020=(list(LCOMe_2020c.values())[27:36]-list(LCOMe_2020c.values())[31])*1000
cdGRID_2020=(list(LCOMe_2020c.values())[36:45]-list(LCOMe_2020c.values())[40])*1000

cdPV_2021=(list(LCOMe_2021c.values())[0:9]-list(LCOMe_2021c.values())[4])*1000
cdPEM_2021=(list(LCOMe_2021c.values())[9:18]-list(LCOMe_2021c.values())[13])*1000
cdMETH_2021=(list(LCOMe_2021c.values())[18:27]-list(LCOMe_2021c.values())[22])*1000
cdCO2_2021=(list(LCOMe_2021c.values())[27:36]-list(LCOMe_2021c.values())[31])*1000
cdGRID_2021=(list(LCOMe_2021c.values())[36:45]-list(LCOMe_2021c.values())[40])*1000

xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, cdPV_2020, label= 'PV', color='darkorange')
ax1.plot(xticks, cdPEM_2020, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, cdMETH_2020, label= 'methanol plant', color='maroon')
ax1.plot(xticks, cdCO2_2020, label= 'CO2 storage' , color='goldenrod')
ax1.plot(xticks, cdGRID_2020, label= 'grid connection' , color='navy')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/t')
plt.tight_layout()
plt.grid()
plt.show()

#Percent change in LCOE
LCOE_BASE_2020 = cPV_2020[4]
cdPV_perc_2020 = cdPV_2020/(LCOE_BASE_2020*1000/100)
cdPEM_perc_2020 = cdPEM_2020/(LCOE_BASE_2020*1000/100)
cdMETH_perc_2020 = cdMETH_2020/(LCOE_BASE_2020*1000/100)
cdCO2_perc_2020 = cdCO2_2020/(LCOE_BASE_2020*1000/100)
cdGRID_perc_2020 = cdGRID_2020/(LCOE_BASE_2020*1000/100)
LCOE_BASE_2021 = cPV_2021[4]
cdPV_perc_2021 = cdPV_2021/(LCOE_BASE_2021*1000/100)
cdPEM_perc_2021 = cdPEM_2021/(LCOE_BASE_2021*1000/100)
cdMETH_perc_2021 = cdMETH_2021/(LCOE_BASE_2021*1000/100)
cdCO2_perc_2021 = cdCO2_2021/(LCOE_BASE_2021*1000/100)
cdGRID_perc_2021 = cdGRID_2021/(LCOE_BASE_2021*1000/100)


xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, cdPV_perc_2020, label= 'PV', color='darkorange')
ax1.plot(xticks, cdPEM_perc_2020, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, cdMETH_perc_2020, label= 'methanol plant', color='maroon')
ax1.plot(xticks, cdCO2_perc_2020, label= 'CO2 storage' , color='goldenrod')
ax1.plot(xticks, cdGRID_perc_2020, label= 'grid connection' , color='navy')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('%')
plt.tight_layout()
plt.grid()
plt.show()


## ------------------------------- Varying fixed OPEX -------------------------------------
dict_fOPEX = {  'PV plant':         fOPEX_PV,
                'Electrolyzer':     fOPEX_PEM,
                'Methanol plant':   fOPEX_METHANOL,
                'CO2 storage':      fOPEX_CO2
                }
sum(dict_fOPEX.values())

lifetime = dfEconParam['lifetime'][0]
WACC = dfEconParam['discount rate'][0]
D = 32000000 #kg methanol 
factors = create_fOPEX_factors()
fOPEXs = calc_fOPEX_variation(factors,dict_fOPEX)
LCOMe_2020o=calc_LC_from_fOPEX(fOPEXs,CAPEX_BASE,vOPEX_BASE_2020,WACC,lifetime)
LCOMe_2021o=calc_LC_from_fOPEX(fOPEXs,CAPEX_BASE,vOPEX_BASE_2021,WACC,lifetime)


# PLOT CAPEX Sensitivity - Levelized Cost as function of CAPEX Change -
oPV_2020=list(LCOMe_2020o.values())[0:9]
oPEM_2020=list(LCOMe_2020o.values())[9:18]
oMETH_2020=list(LCOMe_2020o.values())[18:27]
oCO2_2020=list(LCOMe_2020o.values())[27:36]
oPV_2021=list(LCOMe_2021o.values())[0:9]
oPEM_2021=list(LCOMe_2021o.values())[9:18]
oMETH_2021=list(LCOMe_2021o.values())[18:27]
oCO2_2021=list(LCOMe_2021o.values())[27:36]
xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, oPV_2020, label= 'PV', color='darkorange')
ax1.plot(xticks, oPEM_2020, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, oMETH_2020, label= 'methanol plant', color='maroon')
ax1.plot(xticks, oCO2_2020, label= 'CO2 storage' , color='goldenrod')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/kg')
plt.tight_layout()
plt.grid()
plt.show()

# Change in €/tonne
odPV_2020=(list(LCOMe_2020o.values())[0:9]-list(LCOMe_2020o.values())[4])
odPEM_2020=(list(LCOMe_2020o.values())[9:18]-list(LCOMe_2020o.values())[13])
odMETH_2020=(list(LCOMe_2020o.values())[18:27]-list(LCOMe_2020o.values())[22])
odCO2_2020=(list(LCOMe_2020o.values())[27:36]-list(LCOMe_2020o.values())[31])
odPV_2021=(list(LCOMe_2021o.values())[0:9]-list(LCOMe_2021o.values())[4])
odPEM_2021=(list(LCOMe_2021o.values())[9:18]-list(LCOMe_2021o.values())[13])
odMETH_20201=(list(LCOMe_2021o.values())[18:27]-list(LCOMe_2021o.values())[22])
odCO2_2021=(list(LCOMe_2021o.values())[27:36]-list(LCOMe_2021o.values())[31])


xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, odPV_2020, label= 'PV', color='darkorange')
ax1.plot(xticks, odPEM_2020, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, odMETH_2020, label= 'methanol plant', color='maroon')
ax1.plot(xticks, odCO2_2020, label= 'CO2 storage' , color='goldenrod')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/t')
plt.tight_layout()
plt.grid()
plt.show()

#Percent change in LCOE
LCOE_BASE_2020 = cPV_2020[4]
odPV_perc_2020 = odPV_2020/(LCOE_BASE_2020*1000/100)
odPEM_perc_2020 = odPEM_2020/(LCOE_BASE_2020*1000/100)
odMETH_perc_2020 = odMETH_2020/(LCOE_BASE_2020*1000/100)
odCO2_perc_2020 = odCO2_2020/(LCOE_BASE_2020*1000/100)
LCOE_BASE_2021 = cPV_2021[4]
odPV_perc_2021 = odPV_2021/(LCOE_BASE_2021*1000/100)
odPEM_perc_2021 = odPEM_2021/(LCOE_BASE_2021*1000/100)
odMETH_perc_2021 = odMETH_2021/(LCOE_BASE_2021*1000/100)
odCO2_perc_2021 = odCO2_2021/(LCOE_BASE_2021*1000/100)

#Collect CAPEX and fOPEX

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(xticks, cPV_2020, label= 'PV', color='darkorange')
ax1.plot(xticks, cPEM_2020, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, cMETH_2020, label= 'methanol plant', color='maroon')
ax1.plot(xticks, cCO2_2020, label= 'CO2 storage' , color='goldenrod')
ax1.plot(xticks, cGRID_2020, label= 'grid connection' , color='navy')
#ax1.legend(loc='bottom left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/kg')
ax2.plot(xticks, oPV_2020, label= 'PV', color='darkorange')
ax2.plot(xticks, oPEM_2020, label= 'electrolyzer', color='firebrick')
ax2.plot(xticks, oMETH_2020, label= 'methanol plant', color='maroon')
ax2.plot(xticks, oCO2_2020, label= 'CO2 storage' , color='goldenrod')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_ylabel('€/kg')
plt.tight_layout()
plt.show()

def plot_CAPEX_fOPEX(cPV,cPEM,cMETH,cCO2,cGRID,oPV,oPEM,oMETH,oCO2):
    figaro, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, sharey = True)
    xticks = ['-20%','-15%','-10%','-5%','0','+5%','+10%','+15%','+20%']
    ax1.grid()
    ax2.grid()
    cPV_plotresult = ax1.plot(np.arange(0,len(cPV),1), cPV,marker='o', label = 'PV plant',color = '#ff6361')
    cPEM_plotresult = ax1.plot(np.arange(0,len(cPEM),1), cPEM,marker='h', label = 'Electrolyzer',color = '#bc5090')
    cMETH_plotresult = ax1.plot(np.arange(0,len(cMETH),1), cMETH,marker='v', label = 'Methanol Plant',color = '#58508d')
    cCO2_plotresult = ax1.plot(np.arange(0,len(cCO2),1), cCO2,marker='D', fillstyle='none', label = 'CO2',color = '#ffa600')
    cGRID_plotresult = ax1.plot(np.arange(0,len(cGRID),1), cGRID,marker='P', label = 'Grid connection',color = '#003f5c')
    #ax1.plot(np.arange(0,len(dict_sto_2020),1), dict_sto_2020.values(),marker='h', label = 'Raw methanol storage capacity',color = '#58508d')
    ax2.plot(np.arange(0,len(oPV),1), oPV,marker='o', label = 'PV plant',color = '#ff6361')
    ax2.plot(np.arange(0,len(oPEM),1), oPEM,marker='h', label = 'Electrolyzer',color = '#bc5090')
    ax2.plot(np.arange(0,len(oMETH),1), oMETH,marker='v',fillstyle='none', label = 'Methanol plant',color = '#58508d')
    ax2.plot(np.arange(0,len(oCO2),1), oCO2,marker='D',fillstyle='none', label = 'CO2',color = '#ffa600')
    #ax2.plot(np.arange(0,len(dict_eff_2021),1), dict_eff_2021.values(),marker='h', label = 'Electrolyzer efficiency',color = '#ff6361')
    #ax2.plot(np.arange(0,len(dict_sto_2021),1), dict_sto_2021.values(),marker='h', label = 'Raw methanol storage capacity',color = '#58508d')
    x = np.arange(0,9,2)
    x_ticks_labels = ['-20','-10','0','+10','+20']
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks_labels)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_ticks_labels)
    #pos = ax1.get_position()
    #ax1.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    #ax1.legend(
    #    loc='upper center', 
    #    bbox_to_anchor=(1.2, 1.32),
    #    ncol=2, 
    #)
    #pos = ax2.get_position()
    #ax2.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    #ax2.legend(
    #    loc='center', 
    #    bbox_to_anchor=(1.5, 1.0),
    #    ncol=1, 
    #)
#    ax1.title.set_text('CAPEX variations')
#    ax2.title.set_text('fOPEX variations')
    #plt.figlegend([cPV_plotresult,cPEM_plotresult,cMETH_plotresult,cCO2_plotresult,cGRID_plotresult], ['1','2','3','4','5'], loc = 'lower center', ncol=1, labelspacing=0.)
    #list_of_lines = [cPV_plotresult,cPEM_plotresult,cMETH_plotresult,cCO2_plotresult,cGRID_plotresult]
    list_of_labels = ['PV plant','Electrolyzer','Methanol reactor','CO2','Grid']
    figaro.legend([cPV_plotresult,cPEM_plotresult,cMETH_plotresult,cCO2_plotresult,cGRID_plotresult],labels=list_of_labels, loc="right", ncol=1, bbox_to_anchor=(0.3, 0))
    #plt.tight_layout()
    #figaro.subplots_adjust(bottom=0.2)
    #figaro.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, ncol=5)
    plt.show()
cPV_res = plot_CAPEX_fOPEX(cPV_2020,cPEM_2020,cMETH_2020,cCO2_2020,cGRID_2020,oPV_2020,oPEM_2020,oMETH_2020,oCO2_2020)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------



file_2020 = Path("DataAnalysis/") / 'Sensitivity_2020.xlsx'
DataX = pd.ExcelFile(file_2020)
df_all_2020 = pd.read_excel(file_2020, sheet_name=DataX.sheet_names)
sheets_2020 = list(df_all_2020)
file_2021 = Path("DataAnalysis/") / 'Sensitivity_2021.xlsx'
DataX = pd.ExcelFile(file_2021)
df_all_2021 = pd.read_excel(file_2021, sheet_name=DataX.sheet_names)
sheets_2021 = list(df_all_2021)

#dem: Change yearly demand (and thereby flow of pure methanol + Pure storage)
#eff: Change the conversion rate from P_PEM to m_H2
#sto: Change raw methanol storage capacity, thereby changing the flexibility of the plant e.g. in terms of producing DA prices are lowest
index_dem = [0,1,2,3,4,5,6,7,8]
index_eff = [9,10,11,12,13,14,15,16, 17]
index_sto = [18,19,20,21,22,23,24,25,26]

dict_dem_2020 = {}
dict_eff_2020 = {}
dict_sto_2020 = {}
for i in range(0,len(index_dem)):
        key = sheets_2020[index_dem[i]]
        df = df_all_2020[key]
        dict_dem_2020[key] = sum(df['P_export']*df['c_DA']-df['P_export']*PT-df['P_import']*df['c_DA']-df['P_import']*CT+df['r_FCR']*df['c_FCR']+df['r_aFRR_up']*df['c_aFRR_up']+df['r_aFRR_down']*df['c_aFRR_down']+df['r_mFRR_up']*df['c_mFRRup'])/10**6
for i in range(0,len(index_eff)):
        key = sheets_2020[index_eff[i]]
        df = df_all_2020[key]
        dict_eff_2020[key] = sum(df['P_export']*df['c_DA']-df['P_export']*PT-df['P_import']*df['c_DA']-df['P_import']*CT+df['r_FCR']*df['c_FCR']+df['r_aFRR_up']*df['c_aFRR_up']+df['r_aFRR_down']*df['c_aFRR_down']+df['r_mFRR_up']*df['c_mFRRup'])/10**6
for i in range(0,len(index_sto)):
        key = sheets_2020[index_sto[i]]
        df = df_all_2020[key]
        dict_sto_2020[key] = sum(df['P_export']*df['c_DA']-df['P_export']*PT-df['P_import']*df['c_DA']-df['P_import']*CT+df['r_FCR']*df['c_FCR']+df['r_aFRR_up']*df['c_aFRR_up']+df['r_aFRR_down']*df['c_aFRR_down']+df['r_mFRR_up']*df['c_mFRRup'])/10**6

dict_dem_2021 = {}
dict_eff_2021 = {}
dict_sto_2021 = {}
for i in range(0,len(index_dem)):
        key = sheets_2021[index_dem[i]]
        df = df_all_2021[key]
        dict_dem_2021[key] = sum(df['P_export']*df['c_DA']-df['P_export']*PT-df['P_import']*df['c_DA']-df['P_import']*CT+df['r_FCR']*df['c_FCR']+df['r_aFRR_up']*df['c_aFRR_up']+df['r_aFRR_down']*df['c_aFRR_down']+df['r_mFRR_up']*df['c_mFRRup'])/10**6
for i in range(0,len(index_eff)):
        key = sheets_2021[index_eff[i]]
        df = df_all_2021[key]
        dict_eff_2021[key] = sum(df['P_export']*df['c_DA']-df['P_export']*PT-df['P_import']*df['c_DA']-df['P_import']*CT+df['r_FCR']*df['c_FCR']+df['r_aFRR_up']*df['c_aFRR_up']+df['r_aFRR_down']*df['c_aFRR_down']+df['r_mFRR_up']*df['c_mFRRup'])/10**6
for i in range(0,len(index_sto)):
        key = sheets_2021[index_sto[i]]
        df = df_all_2021[key]
        dict_sto_2021[key] = sum(df['P_export']*df['c_DA']-df['P_export']*PT-df['P_import']*df['c_DA']-df['P_import']*CT+df['r_FCR']*df['c_FCR']+df['r_aFRR_up']*df['c_aFRR_up']+df['r_aFRR_down']*df['c_aFRR_down']+df['r_mFRR_up']*df['c_mFRRup'])/10**6

# Calculate Levelized cost of methanol based on vOPEX dict generated above
def calc_LC_from_vOPEX(dict,CAPEX_BASE,fOPEX_BASE,WACC,lifetime):
    LCOMe = {}
    component = list(dict.keys())
    if 'dem' in component:
       D_scale = [0.8,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20]
    else:
        D_scale = [1,1,1,1,1,1,1,1,1]
    for j in range(0,len(component)):
        vOPEX = -dict[component[j]]
        LCOMe[component[j]] = (CAPEX_BASE + sum((fOPEX_BASE+vOPEX)/(1+WACC)**t for t in range(1,lifetime+1)))/sum((32*D_scale[j])/(1+WACC)**t for t in range(1,lifetime+1))
    return LCOMe

lifetime = dfEconParam['lifetime'][0]
WACC = dfEconParam['discount rate'][0]
D = 32000000 #kg methanol 

dLCOE_dem_2020 = calc_LC_from_vOPEX(dict_dem_2020,CAPEX_BASE,fOPEX_BASE,WACC,lifetime)
dLCOE_eff_2020 = calc_LC_from_vOPEX(dict_eff_2020,CAPEX_BASE,fOPEX_BASE,WACC,lifetime)
dLCOE_sto_2020 = calc_LC_from_vOPEX(dict_sto_2020,CAPEX_BASE,fOPEX_BASE,WACC,lifetime)
dLCOE_dem_2021 = calc_LC_from_vOPEX(dict_dem_2021,CAPEX_BASE,fOPEX_BASE,WACC,lifetime)
dLCOE_eff_2021 = calc_LC_from_vOPEX(dict_eff_2021,CAPEX_BASE,fOPEX_BASE,WACC,lifetime)
dLCOE_sto_2021 = calc_LC_from_vOPEX(dict_sto_2021,CAPEX_BASE,fOPEX_BASE,WACC,lifetime)


def plot_sensitivity(dict_dem_2020,dict_eff_2020,dict_sto_2020,dict_dem_2021,dict_eff_2021,dict_sto_2021,val_name):
    figaro, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, sharey = True)
    ax1.grid()
    ax2.grid()
    ax1.plot(np.arange(0,len(dict_dem_2020),1), dict_dem_2020.values(),marker='h', label = 'Yearly Demand',color = '#bc5090')
    ax1.plot(np.arange(0,len(dict_eff_2020),1), dict_eff_2020.values(),marker='h', label = 'Electrolyzer efficiency',color = '#ff6361')
    ax1.plot(np.arange(0,len(dict_sto_2020),1), dict_sto_2020.values(),marker='h', label = 'Raw methanol storage capacity',color = '#58508d')
    ax2.plot(np.arange(0,len(dict_dem_2021),1), dict_dem_2021.values(),marker='h', label = 'Yearly Demand',color = '#bc5090')
    ax2.plot(np.arange(0,len(dict_eff_2021),1), dict_eff_2021.values(),marker='h', label = 'Electrolyzer efficiency',color = '#ff6361')
    ax2.plot(np.arange(0,len(dict_sto_2021),1), dict_sto_2021.values(),marker='h', label = 'Raw methanol storage capacity',color = '#58508d')
    #ax2.plot(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1), P_PV_avg24_2020_summer.values(), label = 'V1: PV, jul-aug 2020',color = '#ebc034')
    #ax2.plot(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1), c_DA_avg24_2020_summer.values(), label = 'SPOT price: jul-aug 2020',color = '#04911c')
    x = np.arange(0,9,2)
    x_ticks_labels = ['-20','-10','0','+10','+20']
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks_labels)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_ticks_labels)
    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax1.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.32),
        ncol=1, 
    )
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax2.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.32),
        ncol=1, 
    )
    plt.show()

plot_sensitivity(dLCOE_dem_2020,dLCOE_eff_2020,dLCOE_sto_2020,dLCOE_dem_2021,dLCOE_eff_2021,dLCOE_sto_2021,'Levelized Cost of Methanol')

""" LCOMe[0] = (sum(CAPEXs['0']['2'].values()) + sum((fOPEX_BASE+vOPEX_BASE_2020)/(1+WACC)**t for t in range(1,lifetime+1)))/sum(32/(1+WACC)**t for t in range(1,lifetime+1))
LCOMe[0] = (sum(CAPEXs['0']['2'].values()) + sum((fOPEX_BASE+vOPEX_BASE_2021)/(1+WACC)**t for t in range(1,lifetime+1)))/sum(32/(1+WACC)**t for t in range(1,lifetime+1))
 """




""" df_eff_080_2020 = pd.read_excel("Result_files/Sens_eff_080_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_085_2020 = pd.read_excel("Result_files/Sens_eff_085_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_090_2020 = pd.read_excel("Result_files/Sens_eff_090_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_095_2020 = pd.read_excel("Result_files/Sens_eff_095_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_100_2020 = pd.read_excel("Result_files/Sens_eff_100_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_105_2020 = pd.read_excel("Result_files/Sens_eff_105_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_110_2020 = pd.read_excel("Result_files/Sens_eff_110_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_115_2020 = pd.read_excel("Result_files/Sens_eff_115_V2_year_2020-01-01_2020-12-31.xlsx")
df_eff_120_2020 = pd.read_excel("Result_files/Sens_eff_120_V2_year_2020-01-01_2020-12-31.xlsx")

df_eff_080_2021 = pd.read_excel("Result_files/Sens_eff_080_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_085_2021 = pd.read_excel("Result_files/Sens_eff_085_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_090_2021 = pd.read_excel("Result_files/Sens_eff_090_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_095_2021 = pd.read_excel("Result_files/Sens_eff_095_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_100_2021 = pd.read_excel("Result_files/Sens_eff_100_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_105_2021 = pd.read_excel("Result_files/Sens_eff_105_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_110_2021 = pd.read_excel("Result_files/Sens_eff_110_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_115_2021 = pd.read_excel("Result_files/Sens_eff_115_V2_year_2021-01-01_2021-12-31.xlsx")
df_eff_120_2021 = pd.read_excel("Result_files/Sens_eff_120_V2_year_2021-01-01_2021-12-31.xlsx")
 """

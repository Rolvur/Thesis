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

#--______________________DATA_IMPORT------------------------------------------------
#Calculating LCOMe
file_to_open = Path("DataAnalysis/BASE/") / 'BASE_EconResults_All.xlsx'
DataX = pd.ExcelFile(file_to_open)
df_results = pd.read_excel(file_to_open, sheet_name=DataX.sheet_names)
sheet = list(df_results)

#Import CAPEX and fixed OPEX values 
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")

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
cPV=list(LCOMe_2020c.values())[0:9]
cPEM=list(LCOMe_2020c.values())[9:18]
cMETH=list(LCOMe_2020c.values())[18:27]
cCO2=list(LCOMe_2020c.values())[27:36]
cGRID=list(LCOMe_2020c.values())[36:45]
xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, cPV, label= 'PV', color='darkorange')
ax1.plot(xticks, cPEM, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, cMETH, label= 'methanol plant', color='maroon')
ax1.plot(xticks, cCO2, label= 'CO2 storage' , color='goldenrod')
ax1.plot(xticks, cGRID, label= 'grid connection' , color='navy')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/kg')
plt.tight_layout()
plt.show()

# Change in €/tonne
cdPV=(list(LCOMe_2020c.values())[0:9]-list(LCOMe_2020c.values())[4])*1000
cdPEM=(list(LCOMe_2020c.values())[9:18]-list(LCOMe_2020c.values())[13])*1000
cdMETH=(list(LCOMe_2020c.values())[18:27]-list(LCOMe_2020c.values())[22])*1000
cdCO2=(list(LCOMe_2020c.values())[27:36]-list(LCOMe_2020c.values())[31])*1000
cdGRID=(list(LCOMe_2020c.values())[36:45]-list(LCOMe_2020c.values())[40])*1000
xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, cdPV, label= 'PV', color='darkorange')
ax1.plot(xticks, cdPEM, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, cdMETH, label= 'methanol plant', color='maroon')
ax1.plot(xticks, cdCO2, label= 'CO2 storage' , color='goldenrod')
ax1.plot(xticks, cdGRID, label= 'grid connection' , color='navy')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/t')
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
oPV=list(LCOMe_2020o.values())[0:9]
oPEM=list(LCOMe_2020o.values())[9:18]
oMETH=list(LCOMe_2020o.values())[18:27]
oCO2=list(LCOMe_2020o.values())[27:36]
xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, oPV, label= 'PV', color='darkorange')
ax1.plot(xticks, oPEM, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, oMETH, label= 'methanol plant', color='maroon')
ax1.plot(xticks, oCO2, label= 'CO2 storage' , color='goldenrod')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/kg')
plt.tight_layout()
plt.grid()
plt.show()

# Change in €/tonne
odPV=(list(LCOMe_2020o.values())[0:9]-list(LCOMe_2020o.values())[4])
odPEM=(list(LCOMe_2020o.values())[9:18]-list(LCOMe_2020o.values())[13])
odMETH=(list(LCOMe_2020o.values())[18:27]-list(LCOMe_2020o.values())[22])
odCO2=(list(LCOMe_2020o.values())[27:36]-list(LCOMe_2020o.values())[31])
xticks = ['-20%','-15%','-10%','-5%','baseline','+5%','+10%','+15%','+20%']
fig, (ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(xticks, odPV, label= 'PV', color='darkorange')
ax1.plot(xticks, odPEM, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, odMETH, label= 'methanol plant', color='maroon')
ax1.plot(xticks, odCO2, label= 'CO2 storage' , color='goldenrod')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/t')
plt.tight_layout()
plt.grid()
plt.show()

#Collect CAPEX and fOPEX

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(xticks, cPV, label= 'PV', color='darkorange')
ax1.plot(xticks, cPEM, label= 'electrolyzer', color='firebrick')
ax1.plot(xticks, cMETH, label= 'methanol plant', color='maroon')
ax1.plot(xticks, cCO2, label= 'CO2 storage' , color='goldenrod')
ax1.plot(xticks, cGRID, label= 'grid connection' , color='navy')
#ax1.legend(loc='bottom left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel('€/kg')
ax2.plot(xticks, oPV, label= 'PV', color='darkorange')
ax2.plot(xticks, oPEM, label= 'electrolyzer', color='firebrick')
ax2.plot(xticks, oMETH, label= 'methanol plant', color='maroon')
ax2.plot(xticks, oCO2, label= 'CO2 storage' , color='goldenrod')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_ylabel('€/kg')
plt.tight_layout()
plt.show()




















""" LCOMe[0] = (sum(CAPEXs['0']['2'].values()) + sum((fOPEX_BASE+vOPEX_BASE_2020)/(1+WACC)**t for t in range(1,lifetime+1)))/sum(32/(1+WACC)**t for t in range(1,lifetime+1))
LCOMe[0] = (sum(CAPEXs['0']['2'].values()) + sum((fOPEX_BASE+vOPEX_BASE_2021)/(1+WACC)**t for t in range(1,lifetime+1)))/sum(32/(1+WACC)**t for t in range(1,lifetime+1))
 """

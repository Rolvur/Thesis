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

#--SETTINGS----------------------------------------------------------

#Path to result files to import
path = "Result_files/"  # the "/" is important!

# Import files representing a week in year 'find_year'
find_year = "2021"
find_model = 'V3_SolX' #V1, V2, V3_SolX are options
find_unique = 'V' #can be used to look for specific model runs, e.g. 'pw' or for sensitivity analysis (high/low scen.)

#---FUNCTION DEFINITIONS--------------------------------------------------------

#Import and combine SolX files
def import_model_results(path,find_year,find_model, find_unique):
    files = os.listdir(path)
    files_xls = [f for f in files if f[-4:] == 'xlsx']
    df_import = pd.DataFrame()
    for f in files_xls:
        if (find_year in f) and (find_model in f) and (find_unique in f):
            if not f.startswith("~"):
                print(f)
                data = pd.read_excel(path+f)
                df_import = df_import.append(data)
    return df_import

# Construct Dataframe with CAPEX, vOPEX and fOPEX for all project years
def Econ_Data_Constructor(dfEconParam):
    N = dfEconParam['lifetime'].iloc[0]
    n = [2023+i for i in range(0,N+1)]

    CAPEX_PV = [0 for i in range(0,N+1)]  
    CAPEX_PV[0] = dfEconParam['PV_CAPEX'].iloc[0]
    CAPEX_PEM = [0 for i in range(0,N+1)]  
    CAPEX_PEM[0] = dfEconParam['PEM_CAPEX'].iloc[0]
    CAPEX_METHANOL = [0 for i in range(0,N+1)]  
    CAPEX_METHANOL[0] = dfEconParam['METHANOL_CAPEX'].iloc[0]
    CAPEX_CO2 = [0 for i in range(0,N+1)]  
    CAPEX_CO2[0] = dfEconParam['CO2_CAPEX'].iloc[0]
    CAPEX_GRID = [0 for i in range(0,N+1)]
    CAPEX_GRID[0] = dfEconParam['Grid_Connection'].iloc[0]
    fOPEX_PV = [dfEconParam['PV_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_PV[0] = 0
    fOPEX_PV_disc = [fOPEX_PV[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    fOPEX_PEM = [dfEconParam['PEM_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_PEM[0] = 0
    fOPEX_PEM_disc = [fOPEX_PEM[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    fOPEX_METHANOL = [dfEconParam['METHANOL_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_METHANOL[0] = 0
    fOPEX_METHANOL_disc = [fOPEX_METHANOL[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    fOPEX_CO2 = [dfEconParam['CO2_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_CO2[0] = 0
    fOPEX_CO2_disc = [fOPEX_CO2[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    CAPEX_sum = [CAPEX_PV[i] + CAPEX_PEM[i] + CAPEX_METHANOL[i] + CAPEX_CO2[i] for i in range(0,N+1)]
    fOPEX_sum = [fOPEX_PV[i] + fOPEX_PEM[i] + fOPEX_METHANOL[i] + fOPEX_CO2[i] for i in range(0,N+1)]
    fOPEX_disc_sum = [fOPEX_PV_disc[i] + fOPEX_PEM_disc[i] + fOPEX_METHANOL_disc[i] + fOPEX_CO2_disc[i] for i in range(0,N+1)]
    df = pd.DataFrame({#Col name : Value(list)
                            'PV_CAPEX' : CAPEX_PV,
                            'PEM_CAPEX': CAPEX_PEM,
                            'METH_CAPEX' : CAPEX_METHANOL,
                            'CO2_CAPEX' : CAPEX_CO2,
                            'GRID_CAPEX' : CAPEX_GRID,
                            'PV_fOPEX' : fOPEX_PV,
                            'PEM_fOPEX': fOPEX_PEM,
                            'METH_fOPEX' : fOPEX_METHANOL,
                            'CO2_fOPEX' : fOPEX_CO2,
                            'PV_fOPEX_disc' : fOPEX_PV_disc,
                            'PEM_fOPEX_disc': fOPEX_PEM_disc,
                            'METH_fOPEX_disc' : fOPEX_METHANOL_disc,
                            'CO2_fOPEX_disc' : fOPEX_CO2_disc,
                            'CAPEX_sum' : CAPEX_sum,
                            'fOPEX_sum' : fOPEX_sum,
                            'fOPEX_disc_sum' : fOPEX_disc_sum
                            }, index=n,
                            )
    return df

def import_model_param(path,find_year,find_model, find_unique):
    files = os.listdir(path)
    files_csv = [f for f in files if f[-3:] == 'csv']
    dict_import = {}
    count = 0
    for f in files_csv:
        if (find_year in f) and (find_model in f) and (find_unique in f) :
            if not f.startswith("~"): #avoid temp files
                count += 1
                key = find_year+'_'+find_model+'_'+find_unique
                pd.read_csv(path+f, header=None).T.to_csv(path+f+'_T', header=False, index=False)
                data = pd.read_csv(path+f+'_T')
                os.remove(path+f+'_T')
            ##
            #TO IMPLEMENTS: THROW ERROR MESSAGE IF DIFFERENCES OCCUR WITHIN LOOP
            ## For now, only the last 'data' is returned
            #dict_import[key+'_n'+str(count)] = data
    #return dict_import
    return data

# --------------- test loop, i.e. multiple settings -------------------
#   define to different types of files to be analyzed
#find_year = ['2020','2021']
find_year = ['2021']
find_model = ['V1','V2','V3_SolX']
find_unique = 'V' 
avoid = 'pw'
#   Import ecnonomic data and store in dataframe
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")
dfEcon = Econ_Data_Constructor(dfEconParam)

#   loop over all relevant files and create a dictionary with a dataframe for each "bundle" e.g. model 3 for 2020
All_Data = {}
All_Param = {}

for m in range(0,len(find_model)):
    for y in range(0,len(find_year)):
        dict_key = 'df_'+find_model[m]+'_'+find_year[y]
        print(dict_key)
        All_Data[dict_key] = import_model_results(path,find_year[y],find_model[m],find_unique)
        All_Param[dict_key] = import_model_param(path,find_year[y],find_model[m], find_unique)
        #All_Param[dict_key] = import_model_param(path,find_year,find_model, find_unique)
#What to calculate *hourly:
#DA import cost OBS! different approach for different models (p_grid vs p_import)
        if find_model[m] == 'V1':
            #sum(All_Data['df_V1_2021']['DA_revenue'])
            #sum(All_Data['df_V1_2021']['DA_expenses'])
            #sum(All_Data['df_V1_2021']['CT_expenses'])
            All_Data[dict_key]['DA_revenue'] = All_Data[dict_key]['P_grid']*All_Data[dict_key]['DA']*(-All_Data[dict_key]['zT'])
            All_Data[dict_key]['DA_expenses'] = All_Data[dict_key]['P_grid']*All_Data[dict_key]['DA']*(1-All_Data[dict_key]['zT'])
            All_Data[dict_key]['CT_expenses'] = All_Data[dict_key]['P_grid']*All_Param[dict_key]['CT'][0]*(1-All_Data[dict_key]['zT'])
            All_Data[dict_key]['PT_expenses'] = All_Data[dict_key]['P_grid']*All_Param[dict_key]['PT'][0]*(-All_Data[dict_key]['zT']) 
        else: 
            All_Data[dict_key]['PT_expenses'] = All_Data[dict_key]['P_export']*All_Param[dict_key]['PT'][0]
            All_Data[dict_key]['CT_expenses'] = All_Data[dict_key]['P_import']*All_Param[dict_key]['CT'][0]
            All_Data[dict_key]['FCR_revenue'] = All_Data[dict_key]['r_FCR']*All_Data[dict_key]['c_FCR']
            All_Data[dict_key]['aFRRup_revenue'] = All_Data[dict_key]['r_aFRR_up']*All_Data[dict_key]['c_aFRR_up']
            #All_Data[dict_key]['aFRRdown_revenue'] = All_Data[dict_key]['r_aFRR_down']*All_Data[dict_key]['c_aFRR_down']
            #All_Data[dict_key]['mFRRdown_revenue'] = All_Data[dict_key]['r_mFRR_up']*All_Data[dict_key]['c_mFRR_up']
            #type in DataFrame column naming in model V2 - all results could be rerun but not done at this point
            if find_model[m] == 'V2':            
                All_Data[dict_key]['DA_revenue'] = All_Data[dict_key]['P_import']*All_Data[dict_key]['c_DA']
                All_Data[dict_key]['DA_expenses'] = All_Data[dict_key]['P_export']*All_Data[dict_key]['c_DA']
                All_Data[dict_key]['aFRRdown_revenue'] = All_Data[dict_key]['r_aFRR_down']*All_Data[dict_key]['c_aFRR_down']
                All_Data[dict_key]['mFRRup_revenue'] = All_Data[dict_key]['r_mFRR_up']*All_Data[dict_key]['c_mFRRup']

            elif find_model[m] == 'V3_SolX':            
                All_Data[dict_key]['DA_revenue'] = All_Data[dict_key]['P_import']*All_Data[dict_key]['DA_clearing']
                All_Data[dict_key]['DA_expenses'] = All_Data[dict_key]['P_export']*All_Data[dict_key]['DA_clearing']
                All_Data[dict_key]['aFRRdown_revenue'] = All_Data[dict_key]['r_aFRR_down']*All_Data[dict_key]['c_aFRRdown']
                All_Data[dict_key]['mFRRup_revenue'] = All_Data[dict_key]['r_mFRR_up']*All_Data[dict_key]['c_mFRR_up']

# Calculate yearly values based on the above
vOPEX_year = {}
for m in range(0,len(find_model)):
    for y in range(0,len(find_year)):
        dict_key = 'df_'+find_model[m]+'_'+find_year[y]
        print('HEREHEREHERE'+dict_key)
        vOPEX_year[dict_key] = {}
        vOPEX_year[dict_key]['DA_revenue'] = sum(All_Data[dict_key]['DA_revenue'])*((365*24)/len(All_Data[dict_key]))
        vOPEX_year[dict_key]['DA_expenses'] = sum(All_Data[dict_key]['DA_expenses'])*((365*24)/len(All_Data[dict_key]))
        vOPEX_year[dict_key]['CT_expenses'] = sum(All_Data[dict_key]['CT_expenses'])*((365*24)/len(All_Data[dict_key]))
        vOPEX_year[dict_key]['PT_expenses'] = sum(All_Data[dict_key]['PT_expenses'])*((365*24)/len(All_Data[dict_key]))
        if find_model[m] != 'V1':
            print('Problem?'+dict_key)
            vOPEX_year[dict_key]['aFRRup_revenue'] = sum(All_Data[dict_key]['aFRRup_revenue'])*((365*24)/len(All_Data[dict_key]))
            vOPEX_year[dict_key]['aFRRdown_revenue'] = sum(All_Data[dict_key]['aFRRdown_revenue'])*((365*24)/len(All_Data[dict_key]))
            vOPEX_year[dict_key]['mFRRup_revenue'] = sum(All_Data[dict_key]['mFRRup_revenue'])*((365*24)/len(All_Data[dict_key]))
            vOPEX_year[dict_key]['FCR_revenue'] = sum(All_Data[dict_key]['FCR_revenue'])*((365*24)/len(All_Data[dict_key]))
        
#What to calc yearly total
#DA import cost
#DA export revenue
#r_FCR revenue
#r_aFRR_up revenue
#r_aFRR_down revenue
#r_mFRR_up revenue


#Bar plot example
dfEcon_OPEX_disc = dfEcon[["PV_fOPEX_disc","PEM_fOPEX_disc","METH_fOPEX_disc","CO2_fOPEX_disc"]]
dfEcon_OPEX_disc.plot(kind="bar")
plt.title("CFA")
plt.xlabel("year")
plt.ylabel("million € (2021)}")
plt.show()


def Econ_Data_Constructor(dfEconParam,dict_vOPEX):
    N = dfEconParam['lifetime'].iloc[0]
    n = [2023+i for i in range(0,N+1)]
    #CAPEX
    CAPEX_PV = [0 for i in range(0,N+1)]  
    CAPEX_PV[0] = dfEconParam['PV_CAPEX'].iloc[0]
    CAPEX_PEM = [0 for i in range(0,N+1)]  
    CAPEX_PEM[0] = dfEconParam['PEM_CAPEX'].iloc[0]
    CAPEX_METHANOL = [0 for i in range(0,N+1)]  
    CAPEX_METHANOL[0] = dfEconParam['METHANOL_CAPEX'].iloc[0]
    CAPEX_CO2 = [0 for i in range(0,N+1)]  
    CAPEX_CO2[0] = dfEconParam['CO2_CAPEX'].iloc[0]
    CAPEX_GRID = [0 for i in range(0,N+1)]
    CAPEX_GRID[0] = dfEconParam['Grid_Connection'].iloc[0]
    # Fixed OPEX
    fOPEX_PV = [dfEconParam['PV_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_PV[0] = 0
    fOPEX_PV_disc = [fOPEX_PV[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    fOPEX_PEM = [dfEconParam['PEM_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_PEM[0] = 0
    fOPEX_PEM_disc = [fOPEX_PEM[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    fOPEX_METHANOL = [dfEconParam['METHANOL_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_METHANOL[0] = 0
    fOPEX_METHANOL_disc = [fOPEX_METHANOL[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    fOPEX_CO2 = [dfEconParam['CO2_OPEX'].iloc[0] for i in range(0,N+1)]  
    fOPEX_CO2[0] = 0
    fOPEX_CO2_disc = [fOPEX_CO2[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    #vOPEX (from model results)
    vOPEX_DA_revenue = [-dict_vOPEX['DA_revenue'] for i in range(0,N+1)]
    vOPEX_DA_revenue[0] = 0
    vOPEX_DA_revenue_disc = [vOPEX_DA_revenue[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    vOPEX_DA_expenses = [dict_vOPEX['DA_expenses'] for i in range(0,N+1)]
    vOPEX_DA_expenses[0] = 0
    vOPEX_DA_expenses_disc = [vOPEX_DA_expenses[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_CT = [dict_vOPEX['CT_expenses'] for i in range(0,N+1)]
    vOPEX_CT[0] = 0
    vOPEX_CT_disc = [vOPEX_CT[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_PT = [dict_vOPEX['PT_expenses'] for i in range(0,N+1)]
    vOPEX_PT[0] = 0
    vOPEX_PT_disc = [vOPEX_PT[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_FCR = [-dict_vOPEX['FCR_revenue'] for i in range(0,N+1)]
    vOPEX_FCR[0] = 0
    vOPEX_FCR_disc = [vOPEX_FC[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_aFRRup = [-dict_vOPEX['aFRR_up_revenue'] for i in range(0,N+1)]
    vOPEX_aFRRup[0] = 0
    vOPEX_aFRRup_disc = [vOPEX_aFRRup[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_aFRRdown = [-dict_vOPEX['aFRR_down_revenue'] for i in range(0,N+1)]
    vOPEX_aFRRdown[0] = 0
    vOPEX_aFRRdown_disc = [vOPEX_aFRRdown[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_mFRRup = [-dict_vOPEX['mFRR_up_revenue'] for i in range(0,N+1)]
    vOPEX_mFRRup[0] = 0
    vOPEX_mFRRup_disc = [vOPEX_mFRRup[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]

    # Summation and discounting
    # CAPEX_sum = [CAPEX_PV[i] + CAPEX_PEM[i] + CAPEX_METHANOL[i] + CAPEX_CO2[i] for i in range(0,N+1)]
    fOPEX_sum = [fOPEX_PV[i] + fOPEX_PEM[i] + fOPEX_METHANOL[i] + fOPEX_CO2[i] for i in range(0,N+1)]
    fOPEX_disc_sum = [fOPEX_PV_disc[i] + fOPEX_PEM_disc[i] + fOPEX_METHANOL_disc[i] + fOPEX_CO2_disc[i] for i in range(0,N+1)]
    vOPEX_sum = [vOPEX_DA_revenue[i] + vOPEX_DA_expenses[i] + vOPEX_CT[i] + vOPEX_PT[i] + vOPEX_FCR[i] + vOPEX_aFRRup[i] + vOPEX_aFRRdown[i] + vOPEX_mFRRup[i] for i in range(0,N+1)]
    vOPEX_sum_disc = [vOPEX_DA_revenue_disc[i] + vOPEX_DA_expenses_disc[i] + vOPEX_CT_disc[i] + vOPEX_PT_disc[i] + vOPEX_FCR_disc[i] + vOPEX_aFRRup_disc[i] + vOPEX_aFRRdown_disc[i] + vOPEX_mFRRup_disc[i] for i in range(0,N+1)]
    
    df = pd.DataFrame({#Col name : Value(list)
                            'PV_CAPEX' : CAPEX_PV,
                            'PEM_CAPEX': CAPEX_PEM,
                            'METH_CAPEX' : CAPEX_METHANOL,
                            'CO2_CAPEX' : CAPEX_CO2,
                            'GRID_CAPEX' : CAPEX_GRID,
                            'PV_fOPEX' : fOPEX_PV,
                            'PEM_fOPEX': fOPEX_PEM,
                            'METH_fOPEX' : fOPEX_METHANOL,
                            'CO2_fOPEX' : fOPEX_CO2,
                            'vOPEX_DA_revenue' : vOPEX_DA_revenue,
                            'vOPEX_DA_expenses' : vOPEX_DA_expenses, 
                            'vOPEX_CT' : vOPEX_CT,
                            'vOPEX_PT' : vOPEX_PT,
                            'vOPEX_FCR' : vOPEX_FCR,
                            'vOPEX_aFRRup' : vOPEX_aFRRup,
                            'vOPEX_aFRRdown' : vOPEX_aFRRdown,
                            'vOPEX_mFRRup' : vOPEX_mFRRup,
                            'PV_fOPEX_disc' : fOPEX_PV_disc,
                            'PEM_fOPEX_disc': fOPEX_PEM_disc,
                            'METH_fOPEX_disc' : fOPEX_METHANOL_disc,
                            'CO2_fOPEX_disc' : fOPEX_CO2_disc,
                            'CAPEX_sum' : CAPEX_sum,
                            'fOPEX_sum' : fOPEX_sum,
                            'fOPEX_disc_sum' : fOPEX_disc_sum
                            }, index=n,
                            )
    return df
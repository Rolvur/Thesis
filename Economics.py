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

#---FUNCTION DEFINITIONS--------------------------------------------------------

def findRepWeek(f,year):
    print('f: ')
    print(f)
    if '12-31' in f and '01-01' in f: #OBS!
        π = 1
        i = 0
        return π,i
    else:
        file_to_open = Path("Result_files/") / 'RepWeeks.xlsx'
        df_RepWeeks = pd.read_excel(file_to_open)
        if year == '2020':
            for i in range(0,len(df_RepWeeks)):
                if str(df_RepWeeks[df_RepWeeks.columns[1]][i])[:10] in f:
                    π = df_RepWeeks[df_RepWeeks.columns[2]][i]
                    return i,π
        elif year == '2021':
            for i in range(0,len(df_RepWeeks)):
                if str(df_RepWeeks[df_RepWeeks.columns[3]][i])[:10] in f:
                    π = df_RepWeeks[df_RepWeeks.columns[4]][i]
                    print('f: ')
                    print(f)
                    print('pi: ')
                    print(π)
                    return i,π

def import_w_model_results(path,find_year,find_model, find_unique, avoid):
    files = os.listdir(path)
    files_xls = [f for f in files if f[-4:] == 'xlsx']
    df_import = pd.DataFrame()
    list_pi = []
    list_i = []
    list_f = []
    for f in files_xls:
        if (find_year in f) and (find_model in f) and (find_unique in f):
            if not f.startswith("~"):
                #if not f.endswith('12-31.xlsx'):# only necessary for weeks:  #OBS!
                print('file: '+f)
                i,π = findRepWeek(f,find_year)
                list_f.append(f)
                #list_pi.append(π)
                #list_i.append(i)
                data = pd.read_excel(path+f)
                df_import = df_import.append(data)
    #df_import.index = (np.arange(0,1680,1)) # only needed if weeks: 
    #print(list_pi)
    #print(list_i)
    #print(list_i)
    #print(list_f)
    #print(list_pi)
    #print(df_import['c_DA'])
    #print(list_i)
    #df_import.to_excel("import_data"+find_year+".xlsx")
    return df_import, list_pi
#Import LAST file in "path" containing "find_year, "find_model" and "find_unique" in file name
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

# Generate dictionary with vOPEX components in a dataframe for each "bundle" e.g. model 3 for 2020
def generate_vOPEX(path,find_model,find_year,find_unique,avoid):
    All_Data = {}
    All_Pi = {}
    All_Param = {}

    for m in range(0,len(find_model)):
        for y in range(0,len(find_year)):
            dict_key = 'df_'+find_model[m]+'_'+find_year[y]
            All_Data[dict_key],All_Pi[dict_key] = import_w_model_results(path,find_year[y],find_model[m],find_unique, avoid)
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
                    print('V2 here: ')
                    print(All_Data[dict_key]['c_DA'])
#                    print('V2 here: ')
                                
                    All_Data[dict_key]['DA_revenue'] = All_Data[dict_key]['P_export']*All_Data[dict_key]['c_DA']
                    print(sum(All_Data[dict_key]['DA_revenue']))
                    All_Data[dict_key]['DA_expenses'] = All_Data[dict_key]['P_import']*All_Data[dict_key]['c_DA']
                    print(sum(All_Data[dict_key]['DA_expenses']))
                    All_Data[dict_key]['aFRRdown_revenue'] = All_Data[dict_key]['r_aFRR_down']*All_Data[dict_key]['c_aFRR_down']
                    All_Data[dict_key]['mFRRup_revenue'] = All_Data[dict_key]['r_mFRR_up']*All_Data[dict_key]['c_mFRRup']

                elif find_model[m] == 'V3_SolX':            
                    All_Data[dict_key]['DA_revenue'] = All_Data[dict_key]['P_export']*All_Data[dict_key]['DA_clearing']
                    All_Data[dict_key]['DA_expenses'] = All_Data[dict_key]['P_import']*All_Data[dict_key]['DA_clearing']
                    All_Data[dict_key]['aFRRdown_revenue'] = All_Data[dict_key]['r_aFRR_down']*All_Data[dict_key]['c_aFRRdown']
                    All_Data[dict_key]['mFRRup_revenue'] = All_Data[dict_key]['r_mFRR_up']*All_Data[dict_key]['c_mFRR_up']
    return All_Data, All_Pi

# Calculate yearly values based on the dictionary 
def calc_vOPEX_year(find_model, find_year,All_Data, All_Pi,data_length):
    vOPEX_year = {}
    for m in range(0,len(find_model)):
        for y in range(0,len(find_year)):
            dict_key = 'df_'+find_model[m]+'_'+find_year[y]
            print(dict_key)
            vOPEX_year[dict_key] = {}
            if data_length == 'year':
                print('data_length = year')
                print('DA_revenue: ')
                vOPEX_year[dict_key]['DA_revenue'] = sum(All_Data[dict_key]['DA_revenue'])#*((365*24)/len(All_Data[dict_key]))
                print(vOPEX_year[dict_key]['DA_revenue'])
                vOPEX_year[dict_key]['DA_expenses'] = sum(All_Data[dict_key]['DA_expenses'])#*((365*24)/len(All_Data[dict_key]))
                vOPEX_year[dict_key]['CT_expenses'] = sum(All_Data[dict_key]['CT_expenses'])#*((365*24)/len(All_Data[dict_key]))
                vOPEX_year[dict_key]['PT_expenses'] = sum(All_Data[dict_key]['PT_expenses'])#*((365*24)/len(All_Data[dict_key]))
                if find_model[m] != 'V1':
                    vOPEX_year[dict_key]['aFRRup_revenue'] = sum(All_Data[dict_key]['aFRRup_revenue'])#*((365*24)/len(All_Data[dict_key]))
                    vOPEX_year[dict_key]['aFRRdown_revenue'] = sum(All_Data[dict_key]['aFRRdown_revenue'])#*((365*24)/len(All_Data[dict_key]))
                    vOPEX_year[dict_key]['mFRRup_revenue'] = sum(All_Data[dict_key]['mFRRup_revenue'])#*((365*24)/len(All_Data[dict_key]))
                    vOPEX_year[dict_key]['FCR_revenue'] = sum(All_Data[dict_key]['FCR_revenue'])#*((365*24)/len(All_Data[dict_key]))
            if data_length == 'week':
                vOPEX_year[dict_key]['DA_revenue'] = sum(sum(All_Data[dict_key]['DA_revenue'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)
                vOPEX_year[dict_key]['DA_expenses'] = sum(sum(All_Data[dict_key]['DA_expenses'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)            
                vOPEX_year[dict_key]['CT_expenses'] = sum(sum(All_Data[dict_key]['CT_expenses'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)
                vOPEX_year[dict_key]['PT_expenses'] = sum(sum(All_Data[dict_key]['PT_expenses'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)
                if find_model[m] != 'V1':
                    vOPEX_year[dict_key]['aFRRup_revenue'] = sum(sum(All_Data[dict_key]['aFRRup_revenue'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)
                    vOPEX_year[dict_key]['aFRRdown_revenue'] = sum(sum(All_Data[dict_key]['aFRRdown_revenue'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)
                    vOPEX_year[dict_key]['mFRRup_revenue'] = sum(sum(All_Data[dict_key]['mFRRup_revenue'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)
                    vOPEX_year[dict_key]['FCR_revenue'] = sum(sum(All_Data[dict_key]['FCR_revenue'][π*168:(π+1)*168])*All_Pi[dict_key][π] for π in range(0,len(All_Pi[dict_key]))) * (365/7)    
    return vOPEX_year

# Combine vOPEX with fOPEX and CAPEX - AND - discount/summations etc.
def Econ_Data_Constructor(dfEconParam,dict_vOPEX):
    print(dict_vOPEX)
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
    vOPEX_DA_revenue = [-dict_vOPEX['DA_revenue']/10**6 for i in range(0,N+1)]
    vOPEX_DA_revenue[0] = 0
    vOPEX_DA_revenue_disc = [vOPEX_DA_revenue[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    #vOPEX_DA_revenue_old = [-dict_vOPEX['DA_revenue_old'] for i in range(0,N+1)]
    #vOPEX_DA_revenue_old[0] = 0
    #vOPEX_DA_revenue_old_disc = [vOPEX_DA_revenue_old[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)] 
    vOPEX_DA_expenses = [dict_vOPEX['DA_expenses']/10**6 for i in range(0,N+1)]
    vOPEX_DA_expenses[0] = 0
    vOPEX_DA_expenses_disc = [vOPEX_DA_expenses[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_CT = [dict_vOPEX['CT_expenses']/10**6 for i in range(0,N+1)]
    vOPEX_CT[0] = 0
    vOPEX_CT_disc = [vOPEX_CT[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    vOPEX_PT = [dict_vOPEX['PT_expenses']/10**6 for i in range(0,N+1)]
    vOPEX_PT[0] = 0
    vOPEX_PT_disc = [vOPEX_PT[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
    if 'FCR_revenue' in dict_vOPEX:
        vOPEX_FCR = [-dict_vOPEX['FCR_revenue']/10**6 for i in range(0,N+1)]
        vOPEX_FCR[0] = 0
        vOPEX_FCR_disc = [vOPEX_FCR[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
        vOPEX_aFRRup = [-dict_vOPEX['aFRRup_revenue']/10**6 for i in range(0,N+1)]
        vOPEX_aFRRup[0] = 0
        vOPEX_aFRRup_disc = [vOPEX_aFRRup[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
        vOPEX_aFRRdown = [-dict_vOPEX['aFRRdown_revenue']/10**6 for i in range(0,N+1)]
        vOPEX_aFRRdown[0] = 0
        vOPEX_aFRRdown_disc = [vOPEX_aFRRdown[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]
        vOPEX_mFRRup = [-dict_vOPEX['mFRRup_revenue']/10**6 for i in range(0,N+1)]
        vOPEX_mFRRup[0] = 0
        vOPEX_mFRRup_disc = [vOPEX_mFRRup[t]/(1+dfEconParam['discount rate'].iloc[0])**t for t in range(0,N+1)]

    # Summation and discounting
    CAPEX_sum = [CAPEX_PV[i] + CAPEX_PEM[i] + CAPEX_METHANOL[i] + CAPEX_CO2[i] for i in range(0,N+1)]
    fOPEX_sum = [fOPEX_PV[i] + fOPEX_PEM[i] + fOPEX_METHANOL[i] + fOPEX_CO2[i] for i in range(0,N+1)]
    fOPEX_disc_sum = [fOPEX_PV_disc[i] + fOPEX_PEM_disc[i] + fOPEX_METHANOL_disc[i] + fOPEX_CO2_disc[i] for i in range(0,N+1)]
    if 'FCR_revenue' in dict_vOPEX:
        vOPEX_sum = [vOPEX_DA_revenue[i] + vOPEX_DA_expenses[i] + vOPEX_CT[i] + vOPEX_PT[i] + vOPEX_FCR[i] + vOPEX_aFRRup[i] + vOPEX_aFRRdown[i] + vOPEX_mFRRup[i] for i in range(0,N+1)]
        vOPEX_sum_disc = [vOPEX_DA_revenue_disc[i] + vOPEX_DA_expenses_disc[i] + vOPEX_CT_disc[i] + vOPEX_PT_disc[i] + vOPEX_FCR_disc[i] + vOPEX_aFRRup_disc[i] + vOPEX_aFRRdown_disc[i] + vOPEX_mFRRup_disc[i] for i in range(0,N+1)]
    else:
        vOPEX_sum = [vOPEX_DA_revenue[i] + vOPEX_DA_expenses[i] + vOPEX_CT[i] + vOPEX_PT[i]  for i in range(0,N+1)]
        vOPEX_sum_disc = [vOPEX_DA_revenue_disc[i] + vOPEX_DA_expenses_disc[i] + vOPEX_CT_disc[i] + vOPEX_PT_disc[i]  for i in range(0,N+1)]
    
    if 'FCR_revenue' in dict_vOPEX:
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
                                #'vOPEX_DA_revenue_old' : vOPEX_DA_revenue_old,
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
    else:
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

# --------------- test loop, i.e. multiple settings -------------------
path = "Result_files/"  # the "/" is important!
#   define to different types of files to be analyzed
#find_year = ['2020','2021']
find_year = ['2020']
find_model = ['V2']
find_unique = 'eff_120' 
data_length = 'year' #'week' or 'year'
avoid = 'week'

# Reading Rep Weeks
if data_length == 'week': 
    file_to_open = Path("Result_files/") / 'RepWeeks.xlsx'
    df_RepWeeks = pd.read_excel(file_to_open)

All_Data, All_Pi = generate_vOPEX(path,find_model,find_year,find_unique,avoid)
vOPEX_year = calc_vOPEX_year(find_model, find_year,All_Data, All_Pi, data_length)
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")

#-------------WRITE CFA YEARLY RESULTS TO EXCEL SHEETS ----------------
writer = pd.ExcelWriter('DataAnalysis/EconResults_'+find_unique+'.xlsx')
for m in range(0,len(find_model)):
    for y in range(0,len(find_year)):
        dict_key = 'df_'+find_model[m]+'_'+find_year[y]
        # Write DataFrame to Excel file with sheet name
        df_Econ = Econ_Data_Constructor(dfEconParam,vOPEX_year[dict_key])
        df_Econ.to_excel(writer, sheet_name=dict_key+'_'+data_length)
writer.save()








#Bar plot example
#dfEcon_OPEX_disc = dfEcon[["PV_fOPEX_disc","PEM_fOPEX_disc","METH_fOPEX_disc","CO2_fOPEX_disc"]]
#dfEcon_OPEX_disc.plot(kind="bar")
#plt.title("CFA")
#plt.xlabel("year")
#plt.ylabel("million € (2021)}")
#plt.show()
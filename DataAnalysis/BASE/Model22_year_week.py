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
                if not f.endswith('12-31.xlsx'):# only necessary for weeks:  #OBS!
                    print('file: '+f)
                    i,π = findRepWeek(f,find_year)
                    list_f.append(f)
                    list_pi.append(π)
                    list_i.append(i)
                    data = pd.read_excel(path+f)
                    df_import = df_import.append(data)
    df_import.index = (np.arange(0,1680,1))
    return df_import, list_pi
    
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
    return data

def Append_Weeks(path,find_model,find_year,find_unique,avoid):
    All_Data = {}
    All_Pi = {}
    All_Param = {}

    for m in range(0,len(find_model)):
        for y in range(0,len(find_year)):
            dict_key = 'df_'+find_model[m]+'_'+find_year[y]
            All_Data[dict_key],All_Pi[dict_key] = import_w_model_results(path,find_year[y],find_model[m],find_unique, avoid)
            All_Data[dict_key]['weight'] = 0
            for w in range(0,int(len(All_Data[dict_key])/168)):
                All_Data[dict_key]['weight'].loc[(w*168):(w*168+167)] = All_Pi[dict_key][w]          
            All_Param[dict_key] = import_model_param(path,find_year[y],find_model[m], find_unique)
            
    return All_Data


path = "Result_files/"  # the "/" is important!

df_RepWeeks = pd.read_excel(Path("Result_files/") / 'RepWeeks.xlsx')
df_V2_2020_year = pd.read_excel(Path("Result_files/") / 'V2_year_2020-01-01_2020-12-31.xlsx')
df_V2_2021_year = pd.read_excel(Path("Result_files/") / 'V2_year_2021-01-01_2021-12-31.xlsx')



# Define what results files to look for / combine
find_year = ['2020']
find_model = ['V2']
find_unique = 'V' 
data_length = 'week' #'week' or 'year'
avoid = 'year'

All_Data = Append_Weeks(path,find_model,find_year,find_unique,avoid)
keys = list(All_Data.keys())

""" # Writing dataframe to excel for manual check of weight assignment
writer = pd.ExcelWriter('DataAnalysis/'+find_model[0]+'_'+find_year[0]+'_weeks.xlsx')
df = All_Data[keys[0]]
df.to_excel(writer)
writer.save() """

df_V2_2020_weeks = All_Data['df_V2_2020']
df_V2_2020_weeks['wP_PV']= df_V2_2020_weeks['P_PV']*df_V2_2020_weeks['weight']
sum(df_V2_2020_weeks['wP_PV'])*366/7
sum(df_V2_2020_weeks['P_PV'])*366/70
PV_2021_year = sum(df_V2_2020_year['P_PV'])

df_V2_2020_weeks['vOPEX_DA_cost'] = df_V2_2020_weeks['c_DA']*df_V2_2020_weeks['P_import']
df_V2_2020_weeks['vOPEX_DA_revenue'] = df_V2_2020_weeks['c_DA']*df_V2_2020_weeks['P_export']
df_V2_2020_weeks['vOPEX_CT'] = df_V2_2020_weeks['P_export']*CT
df_V2_2020_weeks['vOPEX_PT'] = df_V2_2020_weeks['P_export']*PT
df_V2_2020_weeks['vOPEX_FCR'] = df_V2_2020_weeks['r_FCR']*df_V2_2020_weeks['c_FCR']
df_V2_2020_weeks['vOPEX_aFRR_up'] = df_V2_2020_weeks['r_aFRR_up']*df_V2_2020_weeks['c_aFRR_up']
df_V2_2020_weeks['vOPEX_aFRR_down'] = df_V2_2020_weeks['r_aFRR_down']*df_V2_2020_weeks['c_aFRR_down']
df_V2_2020_weeks['vOPEX_mFRR_up'] = df_V2_2020_weeks['r_mFRR_up']*df_V2_2020_weeks['c_mFRRup']
df_V2_2020_weeks['vOPEX'] = -df_V2_2020_weeks['vOPEX_DA_cost']+df_V2_2020_weeks['vOPEX_DA_revenue']+df_V2_2020_weeks['vOPEX_CT']+df_V2_2020_weeks['vOPEX_PT']+df_V2_2020_weeks['vOPEX_FCR']+df_V2_2020_weeks['vOPEX_aFRR_up']+df_V2_2020_weeks['vOPEX_aFRR_down']+df_V2_2020_weeks['vOPEX_mFRR_up']
df_V2_2020_weeks['w_vOPEX_DA_cost'] = df_V2_2020_weeks['vOPEX_DA_cost']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX_DA_revenue'] = df_V2_2020_weeks['vOPEX_DA_revenue']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX_CT'] = df_V2_2020_weeks['vOPEX_CT']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX_PT'] = df_V2_2020_weeks['vOPEX_PT']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX_FCR'] = df_V2_2020_weeks['vOPEX_FCR']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX_aFRR_up'] = df_V2_2020_weeks['vOPEX_aFRR_up']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX_aFRR_down'] = df_V2_2020_weeks['vOPEX_aFRR_down']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX_mFRR_up'] = df_V2_2020_weeks['vOPEX_mFRR_up']*df_V2_2020_weeks['weight']
df_V2_2020_weeks['w_vOPEX'] = df_V2_2020_weeks['vOPEX']*df_V2_2020_weeks['weight']

V2_2020_w_vOPEX = (sum(df_V2_2020_weeks['w_vOPEX'])*366/7)/1000000
V2_2020_vOPEX = (sum(df_V2_2020_weeks['vOPEX'])*366/70)/1000000
(V2_2020_w_vOPEX-V2_2020_vOPEX)/V2_2020_vOPEX
#------------------------V3--------------------------------

# Define what results files to look for / combine
find_year = ['2020']
find_model = ['V3_SolX']
find_unique = 'V' 
data_length = 'week' #'week' or 'year'
avoid = 'year'

All_Data = Append_Weeks(path,find_model,find_year,find_unique,avoid)
keys = list(All_Data.keys())

df_V3_2020_weeks = All_Data['df_V3_SolX_2020']
df_V3_2020_weeks['wP_PV']= df_V3_2020_weeks['P_PV']*df_V3_2020_weeks['weight']
PV_difference = (sum(df_V3_2020_weeks['wP_PV'])*366/7 - sum(df_V3_2020_weeks['P_PV'])*366/70)/(sum(df_V3_2020_weeks['P_PV'])*366/70)


df_V3_2020_weeks['vOPEX_DA_cost'] = df_V3_2020_weeks['DA_clearing']*df_V3_2020_weeks['P_import']
df_V3_2020_weeks['vOPEX_DA_revenue'] = df_V3_2020_weeks['DA_clearing']*df_V3_2020_weeks['P_export']
df_V3_2020_weeks['vOPEX_CT'] = df_V3_2020_weeks['P_export']*CT
df_V3_2020_weeks['vOPEX_PT'] = df_V3_2020_weeks['P_export']*PT
df_V3_2020_weeks['vOPEX_FCR'] = df_V3_2020_weeks['r_FCR']*df_V3_2020_weeks['c_FCR']
df_V3_2020_weeks['vOPEX_aFRR_up'] = df_V3_2020_weeks['r_aFRR_up']*df_V3_2020_weeks['c_aFRR_up']
df_V3_2020_weeks['vOPEX_aFRR_down'] = df_V3_2020_weeks['r_aFRR_down']*df_V3_2020_weeks['c_aFRRdown']
df_V3_2020_weeks['vOPEX_mFRR_up'] = df_V3_2020_weeks['r_mFRR_up']*df_V3_2020_weeks['c_mFRR_up']
df_V3_2020_weeks['vOPEX'] = -df_V3_2020_weeks['vOPEX_DA_cost']+df_V3_2020_weeks['vOPEX_DA_revenue']+df_V3_2020_weeks['vOPEX_CT']+df_V3_2020_weeks['vOPEX_PT']+df_V3_2020_weeks['vOPEX_FCR']+df_V3_2020_weeks['vOPEX_aFRR_up']+df_V3_2020_weeks['vOPEX_aFRR_down']+df_V3_2020_weeks['vOPEX_mFRR_up']
df_V3_2020_weeks['w_vOPEX_DA_cost'] = df_V3_2020_weeks['vOPEX_DA_cost']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX_DA_revenue'] = df_V3_2020_weeks['vOPEX_DA_revenue']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX_CT'] = df_V3_2020_weeks['vOPEX_CT']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX_PT'] = df_V3_2020_weeks['vOPEX_PT']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX_FCR'] = df_V3_2020_weeks['vOPEX_FCR']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX_aFRR_up'] = df_V3_2020_weeks['vOPEX_aFRR_up']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX_aFRR_down'] = df_V3_2020_weeks['vOPEX_aFRR_down']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX_mFRR_up'] = df_V3_2020_weeks['vOPEX_mFRR_up']*df_V3_2020_weeks['weight']
df_V3_2020_weeks['w_vOPEX'] = df_V3_2020_weeks['vOPEX']*df_V3_2020_weeks['weight']

V3_2020_w_vOPEX = (sum(df_V3_2020_weeks['w_vOPEX'])*366/7)/1000000
V3_2020_vOPEX = (sum(df_V3_2020_weeks['vOPEX'])*366/70)/1000000

(V3_2020_w_vOPEX-V3_2020_vOPEX)/V3_2020_vOPEX

#-------------------------------2021---------------------------------------------------
#------------------------------------------------------------------------------------
find_year = ['2021']
find_model = ['V2']
find_unique = 'V' 
data_length = 'week' #'week' or 'year'
avoid = 'year'

All_Data = Append_Weeks(path,find_model,find_year,find_unique,avoid)
keys = list(All_Data.keys())

df_V2_2021_weeks = All_Data['df_V2_2021']
df_V2_2021_weeks['wP_PV']= df_V2_2021_weeks['P_PV']*df_V2_2021_weeks['weight']
PV_difference_2021_V2 = (sum(df_V2_2021_weeks['wP_PV'])*365/7 - sum(df_V2_2021_weeks['P_PV'])*365/70) / (sum(df_V2_2021_weeks['P_PV'])*365/70)


df_V2_2021_weeks['vOPEX_DA_cost'] = df_V2_2021_weeks['c_DA']*df_V2_2021_weeks['P_import']
df_V2_2021_weeks['vOPEX_DA_revenue'] = df_V2_2021_weeks['c_DA']*df_V2_2021_weeks['P_export']
df_V2_2021_weeks['vOPEX_CT'] = df_V2_2021_weeks['P_export']*CT
df_V2_2021_weeks['vOPEX_PT'] = df_V2_2021_weeks['P_export']*PT
df_V2_2021_weeks['vOPEX_FCR'] = df_V2_2021_weeks['r_FCR']*df_V2_2021_weeks['c_FCR']
df_V2_2021_weeks['vOPEX_aFRR_up'] = df_V2_2021_weeks['r_aFRR_up']*df_V2_2021_weeks['c_aFRR_up']
df_V2_2021_weeks['vOPEX_aFRR_down'] = df_V2_2021_weeks['r_aFRR_down']*df_V2_2021_weeks['c_aFRR_down']
df_V2_2021_weeks['vOPEX_mFRR_up'] = df_V2_2021_weeks['r_mFRR_up']*df_V2_2021_weeks['c_mFRRup']
df_V2_2021_weeks['vOPEX'] = -df_V2_2021_weeks['vOPEX_DA_cost']+df_V2_2021_weeks['vOPEX_DA_revenue']+df_V2_2021_weeks['vOPEX_CT']+df_V2_2021_weeks['vOPEX_PT']+df_V2_2021_weeks['vOPEX_FCR']+df_V2_2021_weeks['vOPEX_aFRR_up']+df_V2_2021_weeks['vOPEX_aFRR_down']+df_V2_2021_weeks['vOPEX_mFRR_up']
df_V2_2021_weeks['w_vOPEX_DA_cost'] = df_V2_2021_weeks['vOPEX_DA_cost']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX_DA_revenue'] = df_V2_2021_weeks['vOPEX_DA_revenue']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX_CT'] = df_V2_2021_weeks['vOPEX_CT']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX_PT'] = df_V2_2021_weeks['vOPEX_PT']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX_FCR'] = df_V2_2021_weeks['vOPEX_FCR']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX_aFRR_up'] = df_V2_2021_weeks['vOPEX_aFRR_up']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX_aFRR_down'] = df_V2_2021_weeks['vOPEX_aFRR_down']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX_mFRR_up'] = df_V2_2021_weeks['vOPEX_mFRR_up']*df_V2_2021_weeks['weight']
df_V2_2021_weeks['w_vOPEX'] = df_V2_2021_weeks['vOPEX']*df_V2_2021_weeks['weight']

V2_2021_w_vOPEX = (sum(df_V2_2021_weeks['w_vOPEX'])*366/7)/1000000
V2_2021_vOPEX = (sum(df_V2_2021_weeks['vOPEX'])*366/70)/1000000
(V2_2021_w_vOPEX-V2_2021_vOPEX)/V2_2021_vOPEX

#------------------------V3--------------------------------
find_year = ['2021']
find_model = ['V3_SolX']
find_unique = 'V' 
data_length = 'week' #'week' or 'year'
avoid = 'year'
All_Data = Append_Weeks(path,find_model,find_year,find_unique,avoid)
keys = list(All_Data.keys())

df_V3_2021_weeks = All_Data['df_V3_SolX_2021']
df_V3_2021_weeks['wP_PV']= df_V3_2021_weeks['P_PV']*df_V3_2021_weeks['weight']
PV_difference_V3_2021 = (sum(df_V3_2021_weeks['wP_PV'])*365/7 - sum(df_V3_2021_weeks['P_PV'])*365/70) / (sum(df_V3_2021_weeks['P_PV'])*365/70)


df_V3_2021_weeks['vOPEX_DA_cost'] = df_V3_2021_weeks['DA_clearing']*df_V3_2021_weeks['P_import']
df_V3_2021_weeks['vOPEX_DA_revenue'] = df_V3_2021_weeks['DA_clearing']*df_V3_2021_weeks['P_export']
df_V3_2021_weeks['vOPEX_CT'] = df_V3_2021_weeks['P_export']*CT
df_V3_2021_weeks['vOPEX_PT'] = df_V3_2021_weeks['P_export']*PT
df_V3_2021_weeks['vOPEX_FCR'] = df_V3_2021_weeks['r_FCR']*df_V3_2021_weeks['c_FCR']
df_V3_2021_weeks['vOPEX_aFRR_up'] = df_V3_2021_weeks['r_aFRR_up']*df_V3_2021_weeks['c_aFRR_up']
df_V3_2021_weeks['vOPEX_aFRR_down'] = df_V3_2021_weeks['r_aFRR_down']*df_V3_2021_weeks['c_aFRRdown']
df_V3_2021_weeks['vOPEX_mFRR_up'] = df_V3_2021_weeks['r_mFRR_up']*df_V3_2021_weeks['c_mFRR_up']
df_V3_2021_weeks['vOPEX'] = -df_V3_2021_weeks['vOPEX_DA_cost']+df_V3_2021_weeks['vOPEX_DA_revenue']+df_V3_2021_weeks['vOPEX_CT']+df_V3_2021_weeks['vOPEX_PT']+df_V3_2021_weeks['vOPEX_FCR']+df_V3_2021_weeks['vOPEX_aFRR_up']+df_V3_2021_weeks['vOPEX_aFRR_down']+df_V3_2021_weeks['vOPEX_mFRR_up']
df_V3_2021_weeks['w_vOPEX_DA_cost'] = df_V3_2021_weeks['vOPEX_DA_cost']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX_DA_revenue'] = df_V3_2021_weeks['vOPEX_DA_revenue']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX_CT'] = df_V3_2021_weeks['vOPEX_CT']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX_PT'] = df_V3_2021_weeks['vOPEX_PT']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX_FCR'] = df_V3_2021_weeks['vOPEX_FCR']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX_aFRR_up'] = df_V3_2021_weeks['vOPEX_aFRR_up']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX_aFRR_down'] = df_V3_2021_weeks['vOPEX_aFRR_down']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX_mFRR_up'] = df_V3_2021_weeks['vOPEX_mFRR_up']*df_V3_2021_weeks['weight']
df_V3_2021_weeks['w_vOPEX'] = df_V3_2021_weeks['vOPEX']*df_V3_2021_weeks['weight']

V3_2021_w_vOPEX = (sum(df_V3_2021_weeks['w_vOPEX'])*365/7)/1000000
V3_2021_vOPEX = (sum(df_V3_2021_weeks['vOPEX'])*365/70)/1000000

(V3_2021_w_vOPEX-V3_2021_vOPEX)/V3_2021_vOPEX

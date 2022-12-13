import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
#------------------------------------------------------

#Import Data
file_to_open = Path("DataAnalysis/BASE/") / 'BASE_EconResults_All.xlsx'
DataX = pd.ExcelFile(file_to_open)
dict_BASE = pd.read_excel(file_to_open, sheet_name=DataX.sheet_names)
keys = list(dict_BASE)

#--------------------Calculate vOPEX between the different models--------------------------------------
for n in range(0,len(keys)):
    if n < 4:
        dict_BASE[keys[n]]['vOPEX_sum'] = dict_BASE[keys[n]]['vOPEX_DA_revenue']+dict_BASE[keys[n]]['vOPEX_DA_expenses']+dict_BASE[keys[n]]['vOPEX_CT']+dict_BASE[keys[n]]['vOPEX_PT']
    else:
        dict_BASE[keys[n]]['vOPEX_sum'] = dict_BASE[keys[n]]['vOPEX_DA_revenue']+dict_BASE[keys[n]]['vOPEX_DA_expenses']+dict_BASE[keys[n]]['vOPEX_CT']+dict_BASE[keys[n]]['vOPEX_PT']+dict_BASE[keys[n]]['vOPEX_FCR']+dict_BASE[keys[n]]['vOPEX_aFRRup']+dict_BASE[keys[n]]['vOPEX_aFRRdown']+dict_BASE[keys[n]]['vOPEX_mFRRup'][1]        


#---------------------PLOT CAPEX and Fixed OPEX as pie charts---------------------------------
def PieChart_CAPEX(df):
    Names = ['PV_CAPEX','PEM_CAPEX','METH_CAPEX','CO2_CAPEX','GRID_CAPEX']
    #Names = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    System_Component =['PV','Electrolyzer','Methanol Reactor','CO2 Storage','Grid Connection']
    LIST = []
    # Sum the values
    for i in Names: 
        LIST.append(sum(df[i]))


    #colors = ['#008fd5','#fc4f30','#e5ae37','#6d904f']
    #explode = [0,0.1,0,0]
    plt.pie(LIST,labels=System_Component,startangle=0,autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

    plt.title('HRES CAPEX')
    plt.tight_layout()
    plt.show()

PieChart_CAPEX(dict_BASE[keys[0]])

def PieChart_OPEX(df):
    Names = ['PV_fOPEX','PEM_fOPEX','METH_fOPEX','CO2_fOPEX']
    #Names = ['r_FCR','r_aFRR_up','r_aFRR_down','r_mFRR_up']
    System_Component =['PV','Electrolyzer','Methanol Reactor','CO2 Storage']
    LIST = []
    # Sum the values
    for i in Names: 
        LIST.append(sum(df[i]))


    #colors = ['#008fd5','#fc4f30','#e5ae37','#6d904f']
    #explode = [0,0.1,0,0]
    plt.pie(LIST,labels=System_Component,startangle=0,autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

    plt.title('HRES fOPEX')
    plt.tight_layout()
    plt.show()

PieChart_OPEX(dict_BASE[keys[0]])


#V1_2020 = dict[keys[0]]['vOPEX_DA_revenue'][1]+dict[keys[0]]['vOPEX_DA_expenses'][1]+dict[keys[0]]['vOPEX_CT'][1]+dict[keys[0]]['vOPEX_PT'][1]
#V1_2021 = dict[keys[1]]['vOPEX_DA_revenue'][1]+dict[keys[1]]['vOPEX_DA_expenses'][1]+dict[keys[1]]['vOPEX_CT'][1]+dict[keys[1]]['vOPEX_PT'][1]
#V2_2020 = dict[keys[4]]['vOPEX_DA_revenue'][1]+dict[keys[4]]['vOPEX_DA_expenses'][1]+dict[keys[4]]['vOPEX_CT'][1]+dict[keys[4]]['vOPEX_PT'][1]+dict[keys[4]]['vOPEX_FCR'][1]+dict[keys[4]]['vOPEX_aFRRup'][1]+dict[keys[4]]['vOPEX_aFRRdown'][1]+dict[keys[4]]['vOPEX_mFRRup'][1]
#V2_2021 = dict[keys[5]]['vOPEX_DA_revenue'][1]+dict[keys[5]]['vOPEX_DA_expenses'][1]+dict[keys[5]]['vOPEX_CT'][1]+dict[keys[5]]['vOPEX_PT'][1]+dict[keys[4]]['vOPEX_FCR'][1]+dict[keys[4]]['vOPEX_aFRRup'][1]+dict[keys[4]]['vOPEX_aFRRdown'][1]+dict[keys[4]]['vOPEX_mFRRup'][1]

def Bar_vOPEX_V1_V2(dict,keys):
    #keys[0] = V1_2020_year
    #keys[1] = V1_2021_year
    #keys[4] = V2_2020_year
    #keys[5] = V2_2021_year
# create dataset
    vOPEX = [dict[keys[0]]['vOPEX_sum'][1], dict[keys[1]]['vOPEX_sum'][1], dict[keys[2]]['vOPEX_sum'][1], dict[keys[3]]['vOPEX_sum'][1], dict[keys[4]]['vOPEX_sum'][1], dict[keys[5]]['vOPEX_sum'][1], dict[keys[6]]['vOPEX_sum'][1], dict[keys[7]]['vOPEX_sum'][1], dict[keys[8]]['vOPEX_sum'][1], dict[keys[9]]['vOPEX_sum'][1]]
    bars = (keys[0], keys[1], keys[2], keys[3], keys[4], keys[5], keys[6], keys[7], keys[8], keys[9])
    x_pos = np.arange(len(bars))
    
    # Create bars and choose color
    plt.bar(x_pos, vOPEX, color = (0.5,0.1,0.5,0.6))

    # Add title and axis names
    plt.title('vOPEX')
    #plt.xlabel('categories')
    plt.ylabel('m€/year')
    
    # Create names on the x axis
    plt.xticks(x_pos, bars)
    plt.grid()
    plt.show()
    return vOPEX
vOPEX = Bar_vOPEX_V1_V2(dict_BASE,keys)


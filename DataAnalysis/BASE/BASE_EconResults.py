import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
#------------------------------------------------------

#Import Data
file_to_open = Path("DataAnalysis/") / 'BASE_EconResults_All.xlsx'
DataX = pd.ExcelFile(file_to_open)
dict_BASE = pd.read_excel(file_to_open, sheet_name=DataX.sheet_names)
keys = list(dict_BASE)

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

    plt.title('HRES CAPEX')
    plt.tight_layout()
    plt.show()
PieChart_OPEX(dict_BASE[keys[0]])
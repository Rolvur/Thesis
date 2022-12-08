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
path = "Results_V3/"  # the "/" is important!

# Import files representing a week in year 'string_to_match'
string_to_match = "2020"

#---FUNCTION DEFINITIONS--------------------------------------------------------

#Import and combine SolX files
def import_SolX(path,string_to_match):
    files = os.listdir(path)
    files_xls = [f for f in files if f[-4:] == 'xlsx']
    df_import = pd.DataFrame()
    for f in files_xls:
        if string_to_match in f:
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
    fOPEX_CO2[1] = 0
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




#
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")
dfEcon = Econ_Data_Constructor(dfEconParam)



#Import SolX data
df_import = import_SolX(path, string_to_match)
sum(df_import['vOPEX'])/(len(df_import)/(365*24))

#Bar plot example
dfEcon_OPEX_disc = dfEcon[["PV_fOPEX_disc","PEM_fOPEX_disc","METH_fOPEX_disc","CO2_fOPEX_disc"]]
dfEcon_OPEX_disc.plot(kind="bar")
plt.title("CFA")
plt.xlabel("year")
plt.ylabel("million € (2021)}")
plt.show()

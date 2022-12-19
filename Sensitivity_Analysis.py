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

#Import CAPEX and fixed OPEX values 
dfEconParam = pd.read_excel("Data/Economics_Data.xlsx")

CAPEX_PV = dfEconParam['PV_CAPEX'].iloc[0]
CAPEX_PEM = dfEconParam['PEM_CAPEX'].iloc[0]
CAPEX_METHANOL = dfEconParam['METHANOL_CAPEX'].iloc[0]
CAPEX_CO2 = dfEconParam['CO2_CAPEX'].iloc[0]
CAPEX_GRID = dfEconParam['Grid_Connection'].iloc[0]

fOPEX_PV = dfEconParam['PV_OPEX'].iloc[0]
fOPEX_PEM = dfEconParam['PEM_OPEX'].iloc[0]
fOPEX_METHANOL = dfEconParam['METHANOL_OPEX'].iloc[0]
fOPEX_CO2 = dfEconParam['CO2_OPEX'].iloc[0]

dict_CAPEX = {}
dict_CAPEX = {'PV':CAPEX_PV}


def CAPEX_calc(values,factors):
    CAPEX = sum(values[i]*factors[i] for i in len(values))
    return CAPEX


#def LCOMe_calc(CAPEX,fOPEX,vOPEX):


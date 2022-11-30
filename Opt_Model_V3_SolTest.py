import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
import numpy as np
from Opt_Constants import *
from Data_process import Start_date,End_date, P_PV_max, DA, Demand, c_FCR, c_aFRR_up, c_aFRR_down, c_mFRR_up, π, c_FCRs, c_aFRR_ups, c_aFRR_downs, c_mFRR_ups, Ω, DateRange, pem_setpoint, hydrogen_mass_flow
from Settings import sEfficiency

def ReadResults(Start_date, End_date):
    df_results = pd.read_excel("Result_files/Model3_"+Start_date+"_"+End_date+".xlsx")    

    list_b_FCR = df_results["bidVol_FCR"].tolist();
    list_β_FCR = df_results["bidPrice_FCR"].tolist();
    b_FCR = dict(zip(np.arange(1,len(list_b_FCR)+1),list_b_FCR));
    β_FCR = dict(zip(np.arange(1,len(list_β_FCR)+1),list_β_FCR));

    list_b_aFRR_up = df_results["bidVol_aFRR_up"].tolist();
    list_β_aFRR_up = df_results["bidPrice_aFRR_up"].tolist();
    b_aFRR_up = dict(zip(np.arange(1,len(list_b_aFRR_up)+1),list_b_aFRR_up));
    β_aFRR_up = dict(zip(np.arange(1,len(list_β_aFRR_up)+1),list_β_aFRR_up));

    list_b_aFRR_down = df_results["bidVol_aFRR_down"].tolist();
    list_β_aFRR_down = df_results["bidPrice_aFRR_down"].tolist();
    b_aFRR_down = dict(zip(np.arange(1,len(list_b_aFRR_down)+1),list_b_aFRR_down));
    β_aFRR_down = dict(zip(np.arange(1,len(list_β_aFRR_down)+1),list_β_aFRR_down));

    list_b_mFRR_up = df_results["bidVol_mFRR_up"].tolist();
    list_β_mFRR_up = df_results["bidPrice_mFRR_up"].tolist();
    b_mFRR_up = dict(zip(np.arange(1,len(list_b_mFRR_up)+1),list_b_mFRR_up));
    β_mFRR_up = dict(zip(np.arange(1,len(list_β_mFRR_up)+1),list_β_mFRR_up));
    return b_FCR, b_aFRR_up, b_aFRR_down, b_mFRR_up, β_FCR, β_aFRR_up, β_aFRR_down, β_mFRR_up;

b_FCR, b_aFRR_up, b_aFRR_down, b_mFRR_up, β_FCR, β_aFRR_up, β_aFRR_down, β_mFRR_up = ReadResults(Start_date, End_date);



from turtle import width
from Opt_Constants import *
import scipy
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as md
from statistics import mean
from pathlib import Path

file_to_open1 = Path("Result_files/") / "Model2_All2020.xlsx"
file_to_open2 = Path("Result_files/") / "Model2_All2021.xlsx"
df_resultsM2_2020 = pd.read_excel(file_to_open1)
df_resultsM2_2021 = pd.read_excel(file_to_open2)





######### Stacked BAR Chart ########### Hourly reserved capacities 
def StackBar(Results,P_pem_cap,P_pem_min):
    x = Results.index
    d = scipy.zeros(len(x))

    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)

    FCR_up = Results['FCR "up"']
    aFRR_up = Results['aFRR_up']
    mFRR_up = Results['mFRR_up']
    aFRR_down = Results['aFRR_down']
    FCR_down = Results['FCR "down"']


    sum_cols = ['FCR "up"','aFRR_up','mFRR_up','aFRR_down','FCR "down"']
    sum_act = Results[sum_cols].sum(axis=1)
    PEM_deviate = P_pem_cap - sum_act - P_pem_min




    ax.bar(x, P_pem_min, color='lightsteelblue',linestyle = 'solid', label ='PEM_min',width=0.02)
    ax.bar(x, FCR_up, bottom = P_pem_min,  color='darkorange',linestyle = 'solid', label ='FCR "up"',width=0.02)
    ax.bar(x, aFRR_up, bottom = P_pem_min+FCR_up,  color='firebrick',linestyle = 'solid', label ='aFRR up',width=0.02)
    ax.bar(x, mFRR_up, bottom = P_pem_min+FCR_up+aFRR_up,  color='goldenrod',linestyle = 'solid', label ='mFRR up',width=0.02)
    ax.bar(x, PEM_deviate, bottom = P_pem_min+FCR_up+aFRR_up+mFRR_up,  color='steelblue',linestyle = 'solid', label ='PEM_Free"',width=0.02)
    ax.bar(x, aFRR_down, bottom = P_pem_min+FCR_up+aFRR_up+mFRR_up+PEM_deviate,  color='maroon',linestyle = 'solid', label ='aFRR down',width=0.02)
    ax.bar(x, FCR_down, bottom = P_pem_min+FCR_up+aFRR_up+mFRR_up+PEM_deviate+aFRR_down,  color='darkorange',linestyle = 'solid', label ='FCR "down"',width=0.02)

    ax.plot(x, Results['P_PEM'], color='fuchsia',linestyle = '-', label ='PEM_Setpoint')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.bar(x, df_results['FCR "up"'], color='blue',linestyle = 'solid', label ='FCR "up"',width=0.02)





    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 57])
    #ax.title('')
    plt.tight_layout()
    plt.show()
StackBar(df_resultsM2_2020,P_pem_cap,P_pem_min)













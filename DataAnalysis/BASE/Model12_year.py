from pathlib import Path
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Import all relevant files for yearly results (Model V1 and V2)
file_to_open1 = Path("Result_files/") / "V1_year_2020-01-01_2020-12-31_k.xlsx"
file_to_open2 = Path("Result_files/") / "V1_year_2021-01-01_2021-12-31_k.xlsx"
df_V1_2020 = pd.read_excel(file_to_open1)
df_V1_2021 = pd.read_excel(file_to_open2)
file_to_open1_pw = Path("Result_files/") / "V1_year_2020-01-01_2020-12-31_pw.xlsx"
file_to_open2_pw = Path("Result_files/") / "V1_year_2021-01-01_2021-12-31_pw.xlsx"
df_V1_2020_pw = pd.read_excel(file_to_open1_pw)
df_V1_2021_pw = pd.read_excel(file_to_open2_pw)
file_to_open3 = Path("Result_files/") / "V2_year_2020-01-01_2020-12-31.xlsx"
file_to_open4 = Path("Result_files/") / "V2_year_2021-01-01_2021-12-31.xlsx"
df_V2_2020 = pd.read_excel(file_to_open3)
df_V2_2021 = pd.read_excel(file_to_open4)

#-----------------------------PEM Setpoint distribution-----------------------------

#--------------Setpoint histogram for full year
#Speciying bin
def createBinList(b_start, b_end):
    return [item for item in range(b_start, b_end+1)]
r1, r2 = 0, 53
bin_list = createBinList(r1, r2)
#Plotting histogram for V1 and V2  - 2020
counts, bins, bars = plt.hist([df_V1_2020['P_PEM'],df_V2_2020['P_PEM']],density=True, bins=bin_list ,label=['V1-2020: PEM Setpoint','V2-2020: PEM Setpoint'],width=0.45, color = ['#a83232','#3255a8'])
plt.xlabel('MW')
plt.legend()
plt.ylabel('Density')
plt.show()
#Plotting histogram for V1 and V2  - 2021
counts, bins, bars = plt.hist([df_V1_2021['P_PEM'],df_V2_2021['P_PEM']],density=True, bins=bin_list ,label=['V1-2021: PEM Setpoint','V2-2021: PEM Setpoint'],width=0.45, color = ['#a83232','#3255a8'])
plt.xlabel('MW')
plt.legend()
plt.ylabel('Density')
plt.show()

#Plotting histogram for V1_k and V1_pw - 2020
counts, bins, bars  = plt.hist([df_V1_2020['P_PEM'],df_V1_2020_pw['P_PEM']],density=True, bins=bin_list ,label=['V1-2020-k: PEM Setpoint','V1-2020-pw: PEM Setpoint'],width=0.45, color = ['#a83232','#3255a8'])
plt.xlabel('MW')
plt.legend()
plt.ylabel('Density')
plt.show()

#Plotting histogram for V1_k and V1_pw - 2021
counts, bins, bars = plt.hist([df_V1_2021['P_PEM'],df_V1_2021_pw['P_PEM']],density=True, bins=bin_list ,label=['V1-2021-k: PEM Setpoint','V1-2021-pw: PEM Setpoint'],width=0.45, color = ['#a83232','#3255a8'])
plt.xlabel('MW')
plt.legend()
plt.ylabel('Density')
plt.show()

#-----------------------------------average PEM setpoint count for every hour in the day over a full year--------------------------------------------

#count observations and calculate average setpoint throughout the year
def check_hours(df,series):
    hours = {}
    P_sum = {}
    P_avg = {}
    for i in range(0,24):
        hours[str(i)] = 0
        P_sum[str(i)] = 0
        P_avg[str(i)] = 0
    for j in range(0,len(df)):
        hours[str(df['HourDK'][j].hour)] += 1
        P_sum[str(df['HourDK'][j].hour)] = P_sum[str(df['HourDK'][j].hour)] + df['P_PEM'][j]
#        hours[df['HourDK'][j].hour] += 1
    for i in range(0,24):
        P_avg[str(i)] = P_sum[str(i)]/hours[str(i)]
    return P_avg

#Plotting average daily pem curve for V1_2020_k and V1_2020_pw
P_PEM_avg24_V1_2020_k = check_hours(df_V1_2020,'P_PEM')
P_PEM_avg24_V1_2020_pw = check_hours(df_V1_2020_pw,'P_PEM')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_k),1)-0.25, P_PEM_avg24_V1_2020_k.values(), width = 0.45, label = 'V1_2020_k',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_pw),1)+0.25, P_PEM_avg24_V1_2020_pw.values(), width = 0.45, label = 'V1_2020_pw',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2020_k),1), P_PEM_avg24_V1_2020_k.values(), label = 'V1_2020_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2020_pw),1), P_PEM_avg24_V1_2020_pw.values(), label = 'V1_2020_pw',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2020_k)), list(P_PEM_avg24_V1_2020_k.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

#Plotting average daily pem curve for V1_2021_k and V1_2021_pw
P_PEM_avg24_V1_2021_k = check_hours(df_V1_2021,'P_PEM')
P_PEM_avg24_V1_2021_pw = check_hours(df_V1_2021_pw,'P_PEM')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021_k),1)-0.25, P_PEM_avg24_V1_2021_k.values(), width = 0.45, label = 'V1_2021_k',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1)+0.25, P_PEM_avg24_V1_2021_pw.values(), width = 0.45, label = 'V1_2021_pw',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2021_k)), list(P_PEM_avg24_V1_2021_k.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

#Plotting average daily pem curve for V1_2020 and V2_2020
P_PEM_avg24_V1_2020 = check_hours(df_V1_2020,'P_PEM')
P_PEM_avg24_V2_2020 = check_hours(df_V2_2020,'P_PEM')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020),1)-0.25, P_PEM_avg24_V1_2020.values(), width = 0.45, label = 'V1_2020',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V2_2020),1)+0.25, P_PEM_avg24_V2_2020.values(), width = 0.45, label = 'V2_2020',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2020)), list(P_PEM_avg24_V1_2020.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

#Plotting average daily pem curve for V1_2021 and V2_2021
P_PEM_avg24_V1_2021 = check_hours(df_V1_2021,'P_PEM')
P_PEM_avg24_V2_2021 = check_hours(df_V2_2021,'P_PEM')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021),1)-0.25, P_PEM_avg24_V1_2021.values(), width = 0.45, label = 'V1_2020',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V2_2021),1)+0.25, P_PEM_avg24_V2_2021.values(), width = 0.45, label = 'V2_2020',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2021)), list(P_PEM_avg24_V1_2021.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

#----------------------------------- Seasonal 24-hour PEM curves -----------------------------
def find_january_start(df):
    for n in range(0,len(df)):
        if df['HourDK'][n].month == 1:
            return n
def find_february_end(df):
    for n in range(0,len(df)):
        if df['HourDK'][n].month == 3:
            return n
def find_july_start(df):
    for n in range(0,len(df)):
        if df['HourDK'][n].month == 7:
            return n
def find_august_end(df):
    for n in range(0,len(df)):
        if df['HourDK'][n].month == 9:
            return n
def check_hours_season(df,series,season):
    hours = {}
    P_sum = {}
    P_avg = {}

    if season == 'summer':
        r1 = find_july_start(df)
        r2 = find_august_end(df)
    elif season == 'winter':
        r1 = find_january_start(df)
        r2 = find_february_end(df)

    for i in range(0,24):
        hours[str(i)] = 0
        P_sum[str(i)] = 0
        P_avg[str(i)] = 0
    for j in range(r1,r2):
        hours[str(df['HourDK'][j].hour)] += 1
        P_sum[str(df['HourDK'][j].hour)] = P_sum[str(df['HourDK'][j].hour)] + df['P_PEM'][j]
#        hours[df['HourDK'][j].hour] += 1
    for i in range(0,24):
        P_avg[str(i)] = P_sum[str(i)]/hours[str(i)]
    return P_avg

P_PEM_avg24_V1_2020_summer = check_hours_season(df_V1_2020,'P_PEM','summer')
P_PEM_avg24_V1_2020_winter = check_hours_season(df_V1_2020,'P_PEM','winter')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_summer),1)-0.25, P_PEM_avg24_V1_2020_summer.values(), width = 0.45, label = 'V1_2020_summer',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1)+0.25, P_PEM_avg24_V1_2020_winter.values(), width = 0.45, label = 'V1_2020_winter',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2020_summer)), list(P_PEM_avg24_V1_2020_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

P_PEM_avg24_V1_2021_summer = check_hours_season(df_V1_2021,'P_PEM','summer')
P_PEM_avg24_V1_2021_winter = check_hours_season(df_V1_2021,'P_PEM','winter')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021_summer),1)-0.25, P_PEM_avg24_V1_2021_summer.values(), width = 0.45, label = 'V1_2020_summer',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021_winter),1)+0.25, P_PEM_avg24_V1_2021_winter.values(), width = 0.45, label = 'V1_2020_winter',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2021_summer)), list(P_PEM_avg24_V1_2021_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

#----------------------------------dPEM/dt---------------------------------------
def dPEM_dt(df):
    ddf = df.diff()      
    df = pd.DataFrame(columns = ['Hour', 'Articles', 'Improved'])
    len(df)    
ddf = df.diff()
plt.plot(ddf.index,ddf['P_PEM'])
plt.show()

#ddf['abs_P_PEM']=
#ddf['P_PEM'].abs().mean()
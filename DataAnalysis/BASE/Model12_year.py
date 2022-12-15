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

fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios': [1, 1]})
ax1.grid()
ax2.grid()
ax1.hist([df_V1_2020['P_PEM'],df_V1_2020_pw['P_PEM']],density=True, bins=bin_list ,label=['2020-k','2020-pw'],width=0.45, color = ['#a83232','#3255a8'])
ax2.hist([df_V1_2021['P_PEM'],df_V1_2021_pw['P_PEM']],density=True, bins=bin_list ,label=['2021-k','2021-pw'],width=0.45, color = ['#a83232','#3255a8'])
ax2.tick_params(axis='MW', rotation=0)
ax1.set_ylabel('Density')
ax2.set_ylabel('Density')
plt.tight_layout()
ax1.legend()
ax2.legend()
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

#, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True,gridspec_kw={'height_ratios': [1, 1]})
figaro, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, sharey = True)
ax1.grid()
ax2.grid()
ax1.bar(np.arange(0,len(P_PEM_avg24_V1_2020_k),1)-0.25, P_PEM_avg24_V1_2020_k.values(), width = 0.45, label = 'V1_2020_k',color = '#a83232')
ax1.bar(np.arange(0,len(P_PEM_avg24_V1_2020_pw),1)+0.25, P_PEM_avg24_V1_2020_pw.values(), width = 0.45, label = 'V1_2020_pw',color = '#3255a8')
ax2.bar(np.arange(0,len(P_PEM_avg24_V1_2021_k),1)-0.25, P_PEM_avg24_V1_2021_k.values(), width = 0.45, label = 'V1_2021_k',color = '#a83232')
ax2.bar(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1)+0.25, P_PEM_avg24_V1_2021_pw.values(), width = 0.45, label = 'V1_2021_pw',color = '#3255a8')
ax1.xticks(range(len(P_PEM_avg24_V1_2020_k)), list(P_PEM_avg24_V1_2020_k.keys()))
ax2.xticks(range(len(P_PEM_avg24_V1_2021_k)), list(P_PEM_avg24_V1_2021_k.keys()))
ax2.tick_params(axis='MW', rotation=0)
ax1.set_ylabel('MW')
#ax2.set_ylabel('MW')
plt.tight_layout()
ax1.legend()
ax2.legend()
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
        P_sum[str(df['HourDK'][j].hour)] = P_sum[str(df['HourDK'][j].hour)] + df[series][j]
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


#------------ V1-k and V1-pw summer 2020 -----------------
P_PEM_avg24_V1_2020_k_summer = check_hours_season(df_V1_2020,'P_PEM','summer')
P_PEM_avg24_V1_2020_pw_summer = check_hours_season(df_V1_2020_pw,'P_PEM','summer')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_summer),1)-0.25, P_PEM_avg24_V1_2020_k_summer.values(), width = 0.45, label = 'V1_2020_summer',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_pw_summer),1)+0.25, P_PEM_avg24_V1_2020_pw_summer.values(), width = 0.45, label = 'V1_2020_summer',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2020_k_summer)), list(P_PEM_avg24_V1_2020_k_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

P_PEM_avg24_V1_2020_k_winter = check_hours_season(df_V1_2020,'P_PEM','winter')
P_PEM_avg24_V1_2020_pw_winter = check_hours_season(df_V1_2020_pw,'P_PEM','winter')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1)-0.25, P_PEM_avg24_V1_2020_k_winter.values(), width = 0.45, label = 'V1-k: 2020, jan-feb',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2020_pw_winter),1)+0.25, P_PEM_avg24_V1_2020_pw_winter.values(), width = 0.45, label = 'V1-pw: 2020, jan-feb',color = '#3255a8')
plt.xticks(range(len(P_PEM_avg24_V1_2020_k_winter)), list(P_PEM_avg24_V1_2020_k_winter.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

P_PEM_avg24_V1_2020_k_winter = check_hours_season(df_V1_2020,'P_PEM','winter')
P_PEM_avg24_V1_2020_pw_winter = check_hours_season(df_V1_2020_pw,'P_PEM','winter')
c_DA_avg24_2020_winter = check_hours_season(df_V1_2020,'DA','winter')
c_DA_avg24_2020_summer = check_hours_season(df_V1_2020,'DA','summer')
P_PEM_avg24_V1_2020_k_summer = check_hours_season(df_V1_2020,'P_PEM','summer')
P_PEM_avg24_V1_2020_pw_summer = check_hours_season(df_V1_2020_pw,'P_PEM','summer')
P_PV_avg24_2020_summer = check_hours_season(df_V1_2020_pw,'P_PV','summer')
for key in P_PV_avg24_2020_summer:    
    P_PV_avg24_2020_summer[key] /=  3
figaro, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, sharey = True)
ax1.grid()
ax2.grid()
ax1.bar(np.arange(0,len(P_PEM_avg24_V1_2020_k_winter),1)-0.25, P_PEM_avg24_V1_2020_k_winter.values(), width = 0.45, label = 'V1-k: jan-feb 2020',color = '#a83232')
ax1.bar(np.arange(0,len(P_PEM_avg24_V1_2020_pw_winter),1)+0.25, P_PEM_avg24_V1_2020_pw_winter.values(), width = 0.45, label = 'V1-pw: jan-feb 2020',color = '#3255a8')
ax1.plot(np.arange(0,len(P_PEM_avg24_V1_2020_pw_winter),1), c_DA_avg24_2020_winter.values(), label = 'SPOT price: jan-feb 2020',color = '#04911c')
ax2.bar(np.arange(0,len(P_PEM_avg24_V1_2020_k_summer),1)-0.25, P_PEM_avg24_V1_2020_k_summer.values(), width = 0.45, label = 'V1-k: july-aug 2020',color = '#a83232')
ax2.bar(np.arange(0,len(P_PEM_avg24_V1_2020_pw_summer),1)+0.25, P_PEM_avg24_V1_2020_pw_summer.values(), width = 0.45, label = 'V1-pw: july-aug 2020',color = '#3255a8')
ax2.plot(np.arange(0,len(P_PEM_avg24_V1_2020_pw_winter),1), P_PV_avg24_2020_summer.values(), label = 'V1-: PV, jul-aug 2020',color = '#ebc034')
ax2.plot(np.arange(0,len(P_PEM_avg24_V1_2020_pw_winter),1), c_DA_avg24_2020_summer.values(), label = 'SPOT price: jul-aug 2020',color = '#04911c')
ax1.xticks(range(len(P_PEM_avg24_V1_2020_k_winter)), list(P_PEM_avg24_V1_2020_k_winter.keys()))
ax2.xticks(range(len(P_PEM_avg24_V1_2020_k_winter)), list(P_PEM_avg24_V1_2020_k_winter.keys()))
ax2.tick_params(axis='MW', rotation=0)
ax1.set_ylabel('MW')
ax1.set_xlabel('hour')
ax2.set_xlabel('hour')
#ax2.set_ylabel('MW')
#plt.tight_layout()
# Add a legend
pos = ax1.get_position()
ax1.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
ax1.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.32),
    ncol=1, 
)
pos = ax2.get_position()
ax2.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
ax2.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.32),
    ncol=1, 
)
#ax1.legend(loc='upper center', bbox_to_anchor=(1.05,1))
#ax2.legend(loc='upper left', bbox_to_anchor=(1.05,1))
#ax2.legend()
plt.show()

#----------------------------------dPEM/dt---------------------------------------
ddf_V1_2020 = df_V1_2020.diff().tail(-1)
ddf_V1_2021 = df_V1_2021.diff().tail(-1)
ddf_V1_2020_pw = df_V1_2020_pw.diff().tail(-1)
ddf_V1_2021_pw = df_V1_2021_pw.diff().tail(-1)
ddf_V2_2020 = df_V2_2020.diff().tail(-1)
ddf_V2_2021 = df_V2_2021.diff().tail(-1)


dPEM_dt = {}
dPEM_dt['V1_2020'] = ddf_V1_2020['P_PEM'].abs().mean()
dPEM_dt['V1_2021'] = ddf_V1_2021['P_PEM'].abs().mean()
dPEM_dt['V1_2020_pw'] = ddf_V1_2020_pw['P_PEM'].abs().mean()
dPEM_dt['V1_2021_pw'] = ddf_V1_2021_pw['P_PEM'].abs().mean()
dPEM_dt['V2_2020'] = ddf_V2_2020['P_PEM'].abs().mean()
dPEM_dt['V2_2021'] = ddf_V2_2021['P_PEM'].abs().mean()


plt.bar(range(len(dPEM_dt)), list(dPEM_dt.values()), align='center')
plt.xticks(range(len(dPEM_dt)), list(dPEM_dt.keys()))
plt.show()

def createBinList(b_start, b_end):
    return [item for item in range(b_start, b_end+1)]
r1, r2 = 0, 53
bin_list = createBinList(r1, r2)
fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios': [1, 1]})
ax1.grid()
ax2.grid()
ax1.hist([ddf_V1_2020['P_PEM'],ddf_V1_2020_pw['P_PEM']],density=True, bins=bin_list ,label=['2020-k','2020-pw'],width=0.45, color = ['#a83232','#3255a8'])
ax1.axvline(x = dPEM_dt['V1_2020'], linestyle = 'dashed',color = '#a83232', label = '2020-k: mean')
ax1.axvline(x = dPEM_dt['V1_2020_pw'],linestyle = 'dashed', color = '#3255a8', label = '2020-pw: mean')
ax2.hist([ddf_V1_2021['P_PEM'],ddf_V1_2021_pw['P_PEM']],density=True, bins=bin_list ,label=['2021-k','2021-pw'],width=0.45, color = ['#a83232','#3255a8'])
ax2.axvline(x = dPEM_dt['V1_2021'],linestyle = 'dashed', color = '#a83232', label = '2021-k: mean')
ax2.axvline(x = dPEM_dt['V1_2021_pw'],linestyle = 'dashed', color = '#3255a8', label = '2021-pw: mean')
ax2.tick_params(axis='MW/h', rotation=0)
ax1.set_ylabel('Density')
ax2.set_ylabel('Density')
plt.tight_layout()
ax1.legend()
ax2.legend()
plt.show()


#----------------------

df_V1_2020['P_grid']

plt.plot(df_V1_2020['zT']) #zT positive in summer
plt.show()
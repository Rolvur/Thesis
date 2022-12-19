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

#import V3 results
file_to_open = Path("Result_files/") / "V2_2020-01-13_2020-01-19.xlsx"
df_V2_w = pd.read_excel(file_to_open)
file_to_open = Path("Result_files/") / "V3_SolX_2020-01-13_2020-01-19.xlsx"
df_V3_w = pd.read_excel(file_to_open)

#-----------------------------PEM Setpoint distribution-----------------------------

#--------------Setpoint histogram for full year
#Speciying bin
def createBinList(b_start, b_end):
    return [item for item in range(b_start, b_end+1)]
r1, r2 = 0, 53
bin_list = createBinList(r1, r2)
#Plotting histogram for V1 and V2  - 2020
counts, bins, bars = plt.hist([df_V1_2020['P_PEM'],df_V2_2020['P_PEM'],],density=True, bins=bin_list ,label=['V1-2020: PEM Setpoint','V2-2020: PEM Setpoint'],width=0.45, color = ['#a83232','#3255a8'])
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

# PEM setpoint histogram - V1-k and V1-pw , 2020 and 2021
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

# PEM setpoint histogram - V1 and V2 , 2020 and 2021
fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios': [1, 1]})
ax1.grid()
ax2.grid()
ax1.hist([df_V1_2020['P_PEM'],df_V2_2020['P_PEM']],density=True, bins=bin_list ,label=['V1-2020','V2-2020'],width=0.45, color = ['#a83232','#3255a8'])
ax2.hist([df_V1_2021['P_PEM'],df_V2_2021['P_PEM']],density=True, bins=bin_list ,label=['V1-2021','V2-2021'],width=0.45, color = ['#a83232','#3255a8'])
ax2.tick_params(axis='MW', rotation=0)
ax1.set_ylabel('Density')
ax2.set_ylabel('Density')
plt.tight_layout()
ax1.legend()
ax2.legend()
plt.show()

#Plotting histogram for V2_w and V3_w - 2020
counts, bins, bars = plt.hist([df_V2_w['P_PEM'],df_V3_w['P_PEM']],density=True, bins=bin_list ,label=['V2' ,'V3'],width=0.45, color = ['#a83232','#3255a8'])
plt.xlabel('MW')
plt.legend()
plt.ylabel('Density')
plt.show()

plt.scatter(df_V3_w, df.y)

#count number of hours with reserves
f_FCR_2020 = np.count_nonzero(df_V2_2020['r_FCR'])/len(df_V2_2020)
f_FCR_2021 = np.count_nonzero(df_V2_2021['r_FCR'])/len(df_V2_2021)
mean_FCR_2020 = df_V2_2020['c_FCR'].mean()
mean_FCR_2021 = df_V2_2021['c_FCR'].mean()
mean__cDA_2020 = df_V2_2020['c_DA'].mean()
mean_c_DA_2021 = df_V2_2021['c_DA'].mean()


f_aFRRup_2020 = np.count_nonzero(df_V2_2020['r_aFRR_up'])/len(df_V2_2020)
f_aFRRup_2021 = np.count_nonzero(df_V2_2021['r_aFRR_up'])/len(df_V2_2021)
f_aFRRdown_2020 = np.count_nonzero(df_V2_2020['r_aFRR_down'])/len(df_V2_2020)
f_aFRRdown_2021 = np.count_nonzero(df_V2_2021['r_aFRR_down'])/len(df_V2_2021)
f_mFRRup_2020 = np.count_nonzero(df_V2_2020['r_mFRR_up'])/len(df_V2_2020)
f_mFRRup_2021 = np.count_nonzero(df_V2_2021['r_mFRR_up'])/len(df_V2_2021)


np.count_nonzero(df_V2_2020['r_aFRR_up'])
np.count_nonzero(df_V2_2020['r_aFRR_down'])
np.count_nonzero(df_V2_2020['r_mFRR_up']) 
(df_V2_2021['r_FCR']+df_V2_2020['r_aFRR_up']++df_V2_2020['r_aFRR_down']+df_V2_2020['r_mFRR_up']).eq(0).sum()
(df_V2_2021['r_FCR']+df_V2_2020['r_aFRR_up']+df_V2_2020['r_mFRR_up']).eq(0).sum()
(df_V2_2021['r_FCR']+df_V2_2020['r_aFRR_down']).eq(0).sum() 



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
        P_sum[str(df['HourDK'][j].hour)] = P_sum[str(df['HourDK'][j].hour)] + df[series][j]
#        hours[df['HourDK'][j].hour] += 1
    for i in range(0,24):
        P_avg[str(i)] = P_sum[str(i)]/hours[str(i)]
    return P_avg
def check_hours_notime(df,series):
    getsum = {}
    avg = {}
    for i in range(0,24):
        getsum[str(i)] = 0
        avg[str(i)] = 0
    
    for d in range(0,int(len(df)/24)):
        for h in range(0,24):
            getsum[str(h)] = getsum[str(h)] + df[series][24*d+h]
    for key in avg:    
        avg[key] =  getsum[key]/(len(df)/24)
    return avg


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

#, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True,gridspec_kw={'height_ratios': [1, 1]})


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

#Plotting average daily FCR curve for V2_2020 and V2_2021
#r_FCR_avg24_V2_2020 = check_hours(df_V2_2020,'r_FCR')
r_FCR_avg24_V2_2020 = check_hours_notime(df_V2_2020,'r_FCR')
#r_FCR_avg24_V2_2021 = check_hours(df_V2_2021,'r_FCR')
r_FCR_avg24_V2_2021 = check_hours_notime(df_V2_2021,'r_FCR')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021),1)-0.25, r_FCR_avg24_V2_2020.values(), width = 0.45, label = 'FCR volume 2020',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V2_2021),1)+0.25, r_FCR_avg24_V2_2021.values(), width = 0.45, label = 'FCR volume 2021',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(r_FCR_avg24_V2_2020)), list(r_FCR_avg24_V2_2020.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

#for i in range(0,len(df_V2_2020)):
#    if df_V2_2020['r_aFRR_up'][i] != 0:
#        print(df_V2_2020['HourDK'][i])

r_aFRR_up_avg24_V2_2020 = check_hours_notime(df_V2_2020,'r_aFRR_up')
r_aFRR_up_avg24_V2_2021 = check_hours_notime(df_V2_2021,'r_aFRR_up')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021),1)-0.25, r_aFRR_up_avg24_V2_2020.values(), width = 0.45, label = 'aFRR_up volume 2020',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V2_2021),1)+0.25, r_aFRR_up_avg24_V2_2021.values(), width = 0.45, label = 'aFRR_up volume 2021',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(r_aFRR_up_avg24_V2_2020)), list(r_aFRR_up_avg24_V2_2020.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

r_aFRR_down_avg24_V2_2020 = check_hours_notime(df_V2_2020,'r_aFRR_down')
r_aFRR_down_avg24_V2_2021 = check_hours_notime(df_V2_2021,'r_aFRR_down')
plt.bar(np.arange(0,len(P_PEM_avg24_V1_2021),1)-0.25, r_aFRR_down_avg24_V2_2020.values(), width = 0.45, label = 'aFRR_down volume 2020',color = '#a83232')
plt.bar(np.arange(0,len(P_PEM_avg24_V2_2021),1)+0.25, r_aFRR_down_avg24_V2_2021.values(), width = 0.45, label = 'aFRR_down volume 2021',color = '#3255a8')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_k),1), P_PEM_avg24_V1_2021_k.values(), label = 'V1_2021_k',color = '#a83232')
#plt.plot(np.arange(0,len(P_PEM_avg24_V1_2021_pw),1), P_PEM_avg24_V1_2021_pw.values(), label = 'V1_2021_pw',color = '#3255a8')
plt.xticks(range(len(r_aFRR_down_avg24_V2_2020)), list(r_aFRR_down_avg24_V2_2020.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

P_PEM_avg24_V1_2020 = check_hours_notime(df_V1_2020,'P_PEM')
P_PEM_avg24_V2_2020 = check_hours_notime(df_V2_2020,'P_PEM')
P_PEM_avg24_V2_2021 = check_hours_notime(df_V1_2021,'P_PEM')
P_PEM_avg24_V2_2021 = check_hours_notime(df_V2_2021,'P_PEM')
figaro, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, sharey = True)
ax1.grid()
ax2.grid()
ax1.bar(np.arange(0,len(P_PEM_avg24_V1_2020),1)-0.25, P_PEM_avg24_V1_2020.values(), width = 0.45, label = 'V1_2020_k',color = '#a83232')
ax1.bar(np.arange(0,len(P_PEM_avg24_V2_2020),1)+0.25, P_PEM_avg24_V2_2020.values(), width = 0.45, label = 'V2_2020',color = '#3255a8')
ax2.bar(np.arange(0,len(P_PEM_avg24_V1_2021),1)-0.25, P_PEM_avg24_V1_2021.values(), width = 0.45, label = 'V1_2021',color = '#a83232')
ax2.bar(np.arange(0,len(P_PEM_avg24_V2_2021),1)+0.25, P_PEM_avg24_V2_2021.values(), width = 0.45, label = 'V2_2021',color = '#3255a8')
ax1.xticks(range(len(P_PEM_avg24_V1_2020)), list(P_PEM_avg24_V1_2020.keys()))
ax2.xticks(range(len(P_PEM_avg24_V1_2021)), list(P_PEM_avg24_V1_2021.keys()))
ax2.tick_params(axis='MW', rotation=0)
ax1.set_ylabel('MW')
#ax2.set_ylabel('MW')
plt.tight_layout()
ax1.legend()
ax2.legend()
ax1.set_xlabel('hour')
ax2.set_xlabel('hour')
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
plt.show()



r_FCR_avg24_V2_2020_summer = check_hours_season(df_V2_2020,'r_FCR','summer')
r_FCR_avg24_V2_2020_winter = check_hours_season(df_V2_2020,'r_FCR','winter')
plt.bar(np.arange(0,len(r_FCR_avg24_V2_2020_summer),1)-0.25, r_FCR_avg24_V2_2020_summer.values(), width = 0.45, label = 'FCR - V2 - 2020 - summer',color = '#a83232')
plt.bar(np.arange(0,len(r_FCR_avg24_V2_2020_winter),1)+0.25, r_FCR_avg24_V2_2020_winter.values(), width = 0.45, label = 'FCR - V2 - 2020 - winter',color = '#3255a8')
plt.xticks(range(len(r_FCR_avg24_V2_2020_summer)), list(r_FCR_avg24_V2_2020_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()
r_FCR_avg24_V2_2021_summer = check_hours_season(df_V2_2021,'r_FCR','summer')
r_FCR_avg24_V2_2021_winter = check_hours_season(df_V2_2021,'r_FCR','winter')
plt.bar(np.arange(0,len(r_FCR_avg24_V2_2021_summer),1)-0.25, r_FCR_avg24_V2_2021_summer.values(), width = 0.45, label = 'FCR - V2 - 2021 - summer',color = '#a83232')
plt.bar(np.arange(0,len(r_FCR_avg24_V2_2021_winter),1)+0.25, r_FCR_avg24_V2_2021_winter.values(), width = 0.45, label = 'FCR - V2 - 2021 - winter',color = '#3255a8')
plt.xticks(range(len(r_FCR_avg24_V2_2021_summer)), list(r_FCR_avg24_V2_2021_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()


r_aFRR_up_avg24_V2_2020_summer = check_hours_season(df_V2_2020,'r_aFRR_up','summer')
r_aFRR_up_avg24_V2_2020_winter = check_hours_season(df_V2_2020,'r_aFRR_up','winter')
plt.bar(np.arange(0,len(r_aFRR_up_avg24_V2_2020_summer),1)-0.25, r_aFRR_up_avg24_V2_2020_summer.values(), width = 0.45, label = 'aFRR_up - V2 - 2020 - summer',color = '#a83232')
plt.bar(np.arange(0,len(r_aFRR_up_avg24_V2_2020_winter),1)+0.25, r_aFRR_up_avg24_V2_2020_winter.values(), width = 0.45, label = 'aFRR_up - V2 - 2020 - winter',color = '#3255a8')
plt.xticks(range(len(r_aFRR_up_avg24_V2_2020_summer)), list(r_aFRR_up_avg24_V2_2020_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

r_aFRR_down_avg24_V2_2020_summer = check_hours_season(df_V2_2020,'r_aFRR_down','summer')
r_aFRR_down_avg24_V2_2020_winter = check_hours_season(df_V2_2020,'r_aFRR_down','winter')
plt.bar(np.arange(0,len(r_aFRR_down_avg24_V2_2020_summer),1)-0.25, r_aFRR_down_avg24_V2_2020_summer.values(), width = 0.45, label = 'aFRR_down - V2 - 2020 - summer',color = '#a83232')
plt.bar(np.arange(0,len(r_aFRR_down_avg24_V2_2020_winter),1)+0.25, r_aFRR_down_avg24_V2_2020_winter.values(), width = 0.45, label = 'aFRR_down - V2 - 2020 - winter',color = '#3255a8')
plt.xticks(range(len(r_aFRR_down_avg24_V2_2020_summer)), list(r_aFRR_down_avg24_V2_2020_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

r_mFRR_up_avg24_V2_2020_summer = check_hours_season(df_V2_2020,'r_mFRR_up','summer')
r_mFRR_up_avg24_V2_2020_winter = check_hours_season(df_V2_2020,'r_mFRR_up','winter')
plt.bar(np.arange(0,len(r_mFRR_up_avg24_V2_2020_summer),1)-0.25, r_mFRR_up_avg24_V2_2020_summer.values(), width = 0.45, label = 'mFRR_up - V2 - 2020 - summer',color = '#a83232')
plt.bar(np.arange(0,len(r_mFRR_up_avg24_V2_2020_winter),1)+0.25, r_mFRR_up_avg24_V2_2020_winter.values(), width = 0.45, label = 'mFRR_up - V2 - 2020 - winter',color = '#3255a8')
plt.xticks(range(len(r_mFRR_up_avg24_V2_2020_summer)), list(r_mFRR_up_avg24_V2_2020_summer.keys()))
plt.xlabel('hour')
plt.legend()
plt.ylabel('MW')
plt.show()

#Plotting P_PEM for V1 vs V2 for summer and Winter
P_PEM_avg24_V1_2020_winter = check_hours_season(df_V1_2020,'P_PEM','winter')
P_PEM_avg24_V2_2020_winter = check_hours_season(df_V2_2020,'P_PEM','winter')
c_DA_avg24_2020_winter = check_hours_season(df_V1_2020,'DA','winter')
c_DA_avg24_2020_summer = check_hours_season(df_V1_2020,'DA','summer')
P_PEM_avg24_V1_2020_summer = check_hours_season(df_V1_2020,'P_PEM','summer')
P_PEM_avg24_V2_2020_summer = check_hours_season(df_V2_2020,'P_PEM','summer')
P_PV_avg24_2020_summer = check_hours_season(df_V1_2020_pw,'P_PV','summer')
for key in P_PV_avg24_2020_summer:    
    P_PV_avg24_2020_summer[key] /=  3
figaro, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, sharey = True)
ax1.grid()
ax2.grid()
ax1.bar(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1)-0.25, P_PEM_avg24_V1_2020_winter.values(), width = 0.45, label = 'V1: jan-feb 2020',color = '#a83232')
ax1.bar(np.arange(0,len(P_PEM_avg24_V2_2020_winter),1)+0.25, P_PEM_avg24_V2_2020_winter.values(), width = 0.45, label = 'V2: jan-feb 2020',color = '#3255a8')
ax1.plot(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1), c_DA_avg24_2020_winter.values(), label = 'SPOT price: jan-feb 2020',color = '#04911c')
ax2.bar(np.arange(0,len(P_PEM_avg24_V1_2020_summer),1)-0.25, P_PEM_avg24_V1_2020_summer.values(), width = 0.45, label = 'V1: july-aug 2020',color = '#a83232')
ax2.bar(np.arange(0,len(P_PEM_avg24_V1_2020_summer),1)+0.25, P_PEM_avg24_V2_2020_summer.values(), width = 0.45, label = 'V2: july-aug 2020',color = '#3255a8')
ax2.plot(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1), P_PV_avg24_2020_summer.values(), label = 'V1: PV, jul-aug 2020',color = '#ebc034')
ax2.plot(np.arange(0,len(P_PEM_avg24_V1_2020_winter),1), c_DA_avg24_2020_summer.values(), label = 'SPOT price: jul-aug 2020',color = '#04911c')
ax1.xticks(range(len(P_PEM_avg24_V1_2020_winter)), list(P_PEM_avg24_V1_2020_winter.keys()))
ax2.xticks(range(len(P_PEM_avg24_V1_2020_winter)), list(P_PEM_avg24_V1_2020_winter.keys()))
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

# plotting dPEM/dt for V1_k and V1-pw
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

# plotting dPEM/dt for V1 and V2
fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios': [1, 1]})
ax1.grid()
ax2.grid()
ax1.hist([ddf_V1_2020['P_PEM'],ddf_V2_2020['P_PEM']],density=True, bins=bin_list ,label=['2020-V1','2020-V2'],width=0.45, color = ['#a83232','#3255a8'])
ax1.axvline(x = dPEM_dt['V1_2020'], linestyle = 'dashed',color = '#a83232', label = '2020-V1: mean')
ax1.axvline(x = dPEM_dt['V2_2020'],linestyle = 'dashed', color = '#3255a8', label = '2020-V2: mean')
ax2.hist([ddf_V1_2021['P_PEM'],ddf_V2_2021['P_PEM']],density=True, bins=bin_list ,label=['2021-V1','2021-V2'],width=0.45, color = ['#a83232','#3255a8'])
ax2.axvline(x = dPEM_dt['V1_2021'],linestyle = 'dashed', color = '#a83232', label = '2021-V1: mean')
ax2.axvline(x = dPEM_dt['V2_2021'],linestyle = 'dashed', color = '#3255a8', label = '2021-V2: mean')
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
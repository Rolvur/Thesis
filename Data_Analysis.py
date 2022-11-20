from turtle import width
from Data_process import DA,df_DKDA_raw,df_aFRR,df_DKmFRR_raw,df_mFRR,df_FCR_DE
#from Opt_Model_V2 import df_results
from Opt_Constants import *
import scipy
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as md
from statistics import mean
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')


####################### Plot Model Results (Ready to use) #######################

file_to_open1 = Path("Result_files/") / "Model1_All2020.xlsx"
file_to_open2 = Path("Result_files/") / "Model1_All2021.xlsx"
df_resultsM1_2020 = pd.read_excel(file_to_open1)
df_resultsM1_2021 = pd.read_excel(file_to_open2)


## Subplot exa½ple
def SubPlot1(Results):


    #Defingin x axes
    x = Results.index


    ## Subplot example
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,sharex=True) 
    #nrows and ncols is how to show the plots in the frame. Sharex=True(both plots share xakis)

    #ax1 & ax2 two subplots in fig
    ax1.plot(x, Results['DA'], color='b',linestyle = '-', label ='Day Ahead Price')
    ax1.plot(x, Results['cFCR'], color='g',linestyle = '-', label ='FCR Reserve Price')
    ax1.plot(x, Results['caFRRup'], color='firebrick',linestyle = '-', label ='aFRR up-reserve Price')
    ax1.plot(x, Results['caFRRdown'], color='maroon',linestyle = '-', label ='aFRR down-reserve Price')
    ax1.plot(x, Results['cmFRRup'], color='goldenrod',linestyle = '-', label ='mFRR up-Reserve Price')
    
    ax2.plot(x, Results['PEM'],color='b', label ='PEM setpoint')
    ax2.plot(x, Results['FCR "up"'],color='g', label ='FCR "up/down"')
    ax2.plot(x, Results['aFRR_up'],color='firebrick', label ='aFRR up-reserve')
    ax2.plot(x, Results['aFRR_down'],color='maroon', label ='aFRR down-reserve')
    ax2.plot(x, Results['mFRR_up'],color='goldenrod', label ='mFRR up-reserve')
    ax2.plot(x, df_results['P_PV'], color='orange',label ='PV Production')

    ax3.plot(x, df_results['s_raw'], color='dodgerblue',label ='raw methanol storage')
    ax3.plot(x, df_results['s_Pu'], color='blueviolet',label ='pure methanol storage')
    


    ax1.legend()
    ax1.set_title('MONS')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('EUR/MWh')
    

    ax2.legend(loc='upper left')
    ax2.set_ylim([0, 70])
    #ax2.set_title('TONS')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('MW')
    ax2.tick_params(axis='x', rotation=45)

    ax3.legend()
    ax3.set_title('Storage levels')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('kg')

#    ax3 = ax2.twinx()
#    ax3.plot(x, Results['P_PV'],color='g', label ='PV Production')

#    ax3.legend()
#    ax3.set_ylim([0, 300])
    #ax3.set_title('TONS')
    #ax3.set_xlabel('Time')
#    ax3.set_ylabel('MW')


    plt.tight_layout()
    #ax1.grid()
    plt.show()
SubPlot1(df_results)

#Subplot of Storage level(Shaded region), stack and line
def SubPlot2(Results):

    ##Two line(shaded under line) plot also subplots
    x = df_results.index
    d = scipy.zeros(len(x))

    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)

    #1st Subplot 
    ax4 = ax1.twinx()
    ax1.fill_between(x, Results['s_Pu'], where=Results['s_Pu']>=d, interpolate=True, color='lightblue',label='Pure Storage')
    ax1.bar(x, Results['Demand'], color='red',linestyle = 'solid', label ='Demand',width=0.05)

    #ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylabel('kg')
    ax1.set_ylim([0, 90000])
    ax1.legend(loc='upper left')
    ax1.set_title('Pure Methanol')

    ax4.plot(x, Results['Pure_In'], color='midnightblue',linestyle = '--', label ='Pure In')
    ax4.set_ylabel('kg/s')
    ax4.set_ylim([0, 5000])
    ax4.legend(loc='upper right')

    #2nd Subplot
    ax3 = ax2.twinx()

    ax2.fill_between(x, Results['s_raw'], where=Results['s_raw']>=d, interpolate=True, color='lightgrey',label='Raw Storage')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel('kg')
    ax2.set_ylim([0, 85000])
    ax2.legend(loc='upper left')


    ax3.plot(x, Results['Raw_In'], color='forestgreen',linestyle = 'solid', label ='Raw In')
    ax3.plot(x, Results['Raw_Out'], color='midnightblue',linestyle = '--', label ='Raw Out')
    ax3.set_title('Raw Methanol')
    ax3.set_ylabel('kg/s')
    ax3.set_ylim([0, 11000])
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
SubPlot2(df_results)


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

    ax.plot(x, Results['PEM'], color='fuchsia',linestyle = '-', label ='PEM_Setpoint')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.bar(x, df_results['FCR "up"'], color='blue',linestyle = 'solid', label ='FCR "up"',width=0.02)





    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 57])
    #ax.title('')
    plt.tight_layout()
    plt.show()
StackBar(df_results,P_pem_cap,P_pem_min)

#### Scatter Plots ### 
def ScatterPgridvsDA(Results):

    x = Results['P_grid']
    y = Results['DA']

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)


    #plt.scatter(Results['P_grid'], Results['DA'], s=2, color = 'teal')
    plt.scatter(x, y, s=2, color = 'teal')
    #plt.legend(loc='best', fontsize=16)
    plt.plot(x,p(x),"r--")
    plt.xlabel('Grid [MW]')
    plt.ylabel('Day Ahead Price [€/MWh]')


    plt.show()
ScatterPgridvsDA(df_resultsM1)

def ScatterPEMvsDA(Results):

    x = Results['P_PEM']
    y = Results['DA']

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)


    #plt.scatter(Results['P_grid'], Results['DA'], s=2, color = 'teal')
    plt.scatter(x, y, s=2, color = 'teal')
    #plt.legend(loc='best', fontsize=16)
    plt.plot(x,p(x),"r--")
    plt.xlabel('PEM [MW]')
    plt.ylabel('Day Ahead Price [€/MWh]')


    plt.show()
ScatterPEMvsDA(df_resultsM1)

#Subplot of Storage level(Shaded region), stack and line
def Model1StorageFlow(Results):

    ##Two line(shaded under line) plot also subplots
    x = Results['HourDK']
    d = scipy.zeros(len(x))

    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)

    #1st Subplot 
    ax4 = ax1.twinx()
    ax1.fill_between(x, Results['Pure Storage'], where=Results['Pure Storage']>=d, interpolate=True, color='lightblue',label='Pure Storage')
    ax1.bar(x, Results['Demand'], color='red',linestyle = 'solid', label ='Demand',width=0.05)

    #ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylabel('kg')
    ax1.set_ylim([0, 650000])
    ax1.legend(loc='upper left')
    ax1.set_title('Pure Methanol')

    ax4.plot(x, Results['Pure_In'], color='midnightblue',linestyle = '--', label ='Pure In')
    ax4.set_ylabel('kg/s')
    ax4.set_ylim([0, 5000])
    ax4.legend(loc='upper right')

    #2nd Subplot
    ax3 = ax2.twinx()

    ax2.fill_between(x, Results['Raw Storage'], where=Results['Raw Storage']>=d, interpolate=True, color='lightgrey',label='Raw Storage')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel('kg')
    ax2.set_ylim([0, 110000])
    ax2.legend(loc='upper left')


    ax3.plot(x, Results['Raw_In'], color='forestgreen',linestyle = 'solid', label ='Raw In')
    ax3.plot(x, Results['Raw Out'], color='midnightblue',linestyle = '--', label ='Raw Out')
    ax3.set_title('Raw Methanol')
    ax3.set_ylabel('kg/s')
    ax3.set_ylim([0, 11000])
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
Model1StorageFlow(df_resultsM1)

### ECO ##########################3 

df_resultsM1_2020['P_PEM'].mean()


#Calculate Cost, Revenue and Profit from DA market 


a = df_resultsM1_2020.loc[df_resultsM1_2020['DA'] < 0-18.4815]
a['P_PV'].mean()


b = df_resultsM1_2020.loc[df_resultsM1_2020['zT'] == 1]
a['PV'].mean()
b['DA'].mean()


def DACostRevProf(df):

    consumer = df.loc[df['zT'] == 0]
    producer = df.loc[df['zT'] == 1]

    Cost_DA = sum((CT + consumer['DA'])*consumer['P_grid'])

    Rev_DA = sum((producer['DA']-PT)* (-producer['P_grid']))

    Profit = Rev_DA - Cost_DA

    return Cost_DA,Rev_DA,Profit


DA_Cost2020,DA_Rev_2020,DA_Profit_2020 = DACostRevProf(df_resultsM1_2020)
DA_Cost2021,DA_Rev_2021,DA_Profit_2021 = DACostRevProf(df_resultsM1_2021)


x = ['2020','2021']
d = scipy.zeros(len(x))



Revenue = np.transpose(np.array([DA_Rev_2020,DA_Rev_2021]))
Cost = np.array([-DA_Cost2020,-DA_Cost2021])
Profit = np.array([DA_Profit_2020,DA_Profit_2021])

## DA Cost plot ## 


fig , ax = plt.subplots(nrows=1,ncols=1,sharex=True)

ax.bar(x, [DA_Rev_2020,DA_Rev_2021], color='darkgreen',linestyle = 'solid', label ='DA Revenue')

ax.bar(x , [-DA_Cost2020,-DA_Cost2021], color='maroon',linestyle = '-', label ='DA Cost')

ax.set_ylabel('€')
ax.legend(loc='upper left')



ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()  


df_DA = pd.DataFrame({'Revenue' : Revenue, 'Cost':Cost , 'Profit':Profit}, index=['2020','2021'] )


fig, ax = plt.subplots(nrows=1,ncols=1)

#ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
ax = df_DA.plot.bar(color=['firebrick','navy','darkseagreen'],rot=20, width = 0.5)


ax.set_ylabel('€')
ax.legend(loc='upper right')
ax.set_ylim[300000,-600000]


ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()   





########## PEM vs DA and PV ######### 


df_resultsM1_2020

fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)


x = df_resultsM1_2020['HourDK']

ax1.plot(x, df_resultsM1_2020['DA'], color='g',linestyle = '-', label ='DA')
ax2.plot(x, df_resultsM1_2020['P_PV'], color='r',linestyle = '-', label ='PV')
ax2.plot(x, df_resultsM1_2020['P_PEM'], color='b',linestyle = '-', label ='PEM')
ax2.plot(x, df_resultsM1_2020['P_grid'], color='purple',linestyle = '-', label ='Grid')


ax1.set_ylabel('€')
ax2.set_ylabel('MW')
ax1.legend(loc='best')
ax2.legend(loc='best')
#ax1.set_ylim[300000,-600000]


ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()   





#########SCATTER ZOOM ##########  TRY THIS ON line plot 






####################### Plot Input Data #######################


### PV ### 
file_to_open = Path("Data/") / "PV_data.xlsx"
PV_data = pd.read_excel(file_to_open)

def PlotPVdata(PV_data):
    x = PV_data['Hour UTC']

    PV_data_month = PV_data.groupby(pd.PeriodIndex(PV_data['Hour UTC'], freq='M'))['Power [MW]'].mean()

    PV_data_month = PV_data_month.tolist() 

    days = [31,29,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31]
    hours = 24 
    PV_avg = []
    y = 0
    for i in PV_data_month:
        for j in range(0,days[y]):

            for z in range(0,hours):
                PV_avg.append(i)
        y = y+1



    #Counting occurances in array 
    PV_avg_arr = np.array(PV_avg)
    np.count_nonzero(PV_avg_arr == PV_data_month[3])

    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True)


    ax.bar(x, PV_data['Power [MW]'], color='red', label ='Hourly Production')
    ax.plot(x,PV_avg, color='green',linewidth=2 , label ='Monthly average')

    ax.set_ylabel('[MW]')
    #ax.set_ylim([-60, 170])
    ax.legend(loc='upper left')

    #ax.set_title('Day-Ahead Price')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
PlotPVdata(PV_data)

#Plotting Day ahead prices
def DayAhead(df_DKDA_raw):
    TimeRangePlot = (df_DKDA_raw['HourDK'] >= '2020-01-01 00:00') & (df_DKDA_raw['HourDK']  <= '2021-12-31 23:59')

    df_Data_plot = df_DKDA_raw[TimeRangePlot] 


    x = df_Data_plot['HourDK']


    fig, ax = plt.subplots(nrows=1,ncols=1)

    #ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
    ax.plot(x, df_Data_plot['SpotPriceEUR,,'], color='teal',linestyle = '-', label ='Day-Ahead Price', linewidth=1)
    ax.set_ylabel('[€/MWh]')
    #ax.set_ylim([-60, 170])
    ax.legend(loc='upper left')
    #ax.set_title('Day-Ahead Price')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
DayAhead(df_DKDA_raw)

## Plottting FCR(Germany) Capacity prices 
def FCR(df_FCR):
    
    TimeRangePlot = (df_FCR['DATE_FROM'] >= '2020-01-01 00:00') & (df_FCR['DATE_FROM']  <= '2021-12-31 23:59')#This should be changed in settings.py
    df_FCR = df_FCR[TimeRangePlot]

    #converting string values to float
    df_FCR['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'] = df_FCR['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].astype(float)

    #Converting date from string to datetime 
    x = df_FCR['DATE_FROM'].apply(pd.to_datetime)


    #Begining figure
    fig, ax = plt.subplots(nrows=1,ncols=1)

    #ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
    ax.plot(x, df_FCR['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'], color='royalblue',linestyle = '-', label ='FCR Capacity', linewidth=1)
    ax.set_ylabel('[€/MW]')
    #ax.set_ylim([-60, 170])
    ax.legend(loc='upper left')
    #ax.set_title('Day-Ahead Price')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
FCR(df_FCR_DE)

#Plotting aFRR (Sweden) 
def aFRR(df_aFRR):

    TimeRangePlot = (df_aFRR['Period'] >= '2020-01-01 00:00') & (df_aFRR['Period']  <= '2021-12-31 23:59')  #This should be changed in settings.py
    df_aFRR = df_aFRR[TimeRangePlot]

    x = df_aFRR['Period'].apply(pd.to_datetime)

    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)

    #ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
    ax1.plot(x, df_aFRR['aFRR Upp Pris (EUR/MW)'], color='navy',linestyle = '-', label ='aFRR Up Price', linewidth=1)
    ax2.plot(x, df_aFRR['aFRR Ned Pris (EUR/MW)'], color='firebrick',linestyle = '-', label ='aFRR Down Price', linewidth=1)
    ax1.set_ylabel('[€/MW]')
    #ax.set_ylim([-60, 170])
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    #ax.set_title('Day-Ahead Price')
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()   
aFRR(df_aFRR)

#Plotting mFRR

x = df_mFRR['HourDK']

df_mFRR_avg = df_mFRR.groupby(pd.PeriodIndex(df_mFRR['HourDK'], freq='M'))['mFRR_UpPriceEUR'].mean()

mFRR_avg = df_mFRR_avg.tolist() 

days = [31,29,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31]
hours = 24 
mFRR = []
y = 0
for i in mFRR_avg:
    for j in range(0,days[y]):

        for z in range(0,hours):
            mFRR.append(i)
    y = y+1


#Begining figure
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

#fig = plt.figure()
#ax = fig.add_subplot(111)




#ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
ax1.bar(x, df_mFRR['mFRR_UpPriceEUR'], color='saddlebrown', label ='mFRR Hourly')
#ax2 = ax.twinx()
ax1.plot(x,mFRR, color='navy',linestyle = '-', label ='mFRR Monthly Average ', linewidth=2)

ax2.bar(x, df_mFRR['mFRR_UpPriceEUR'], color='saddlebrown', label ='mFRR Hourly')
#ax2 = ax.twinx()
ax2.plot(x,mFRR, color='navy',linestyle = '-', label ='mFRR Monthly Average ', linewidth=2)



#lns = lns1+lns2
#labs = [l.get_label() for l in lns]
#fig.legend(loc='upper right')

#ax1.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)
ax1.legend(loc="upper left")
ax2.legend(loc="upper left")
#ax.set_title('Day-Ahead Price')
ax1.set_ylabel('[€/MW]')
ax2.set_ylabel('[€/MW]')

ax2.set_ylim([0,50])
ax1.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()



####################### Plot  NOT DONE!!! #######################

## Pie Chart Example## 
def PieChart():


    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    recipe = ["225 g flour",
            "90 g sugar",
            "1 egg",
            "60 g butter",
            "100 ml milk",
            "1/2 package of yeast"]

    data = [225, 90, 50, 60, 100, 5]

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title("Matplotlib bakery: A donut")

    plt.show()




## Multiple bar and line plot together

















#Plot style
plt.style.use('fivethirtyeight') ## 'fivethirtyeight' 'seaborn' etc

#Defingin x axes
x = df_results.index

plt.bar(x_indexes-width,df_results['P_PEM'],width=width, color='r', label = 'PEM') #-width  = shift bar to left
plt.bar(x_indexes,df_results['P_PV'],width=width, color='b', label = 'PV')


#plt.xticks(ticks=x_indexes, labels=x)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()


#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.legend()
plt.tight_layout()
plt.show()













plt.plot(x, df_results.DA, label = "line 1", linestyle="-")
plt.plot(x, df_results.DA, label = "line 2", linestyle="--")
plt.plot(x, df_results.P_sRaw, label = "curve 1")
plt.xticks(rotation=20)
plt.legend()
plt.show()

df_results[['P_sRaw','P_sPu']].plot(kind='bar', width = width)
df_results['DA'].plot(secondary_y=True)

ax = plt.gca()
plt.xlim([-width, len(df_results['P_sRaw'])-width])
ax.set_xticklabels(x)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.show()















labels = df_results.columns #Getting column names


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(df_results.index - width/2, df_results['P_sRaw'], width, label='P_sRaw')
#rects1 = ax.bar(x - width/2, df_results['P_sRaw'], width, label='P_sRaw')
rects2 = ax.bar(x + width/2, df_results['P_sPu'], width, label='P_sPu')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MW')
ax.set_title('Time')
#ax.set_xticks(df_results.index)
ax.legend()

#Shows value on top of bars 
ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()







####################### Plot ####################### 

#Average grouped bar chart FCR
df_FCRR2019_raw
df_FCR20_21
TimeRangePlot2020 = (df_FCR20_21['DATE_FROM'] >= '2020-01-01 00:00') & (df_FCR20_21['DATE_FROM']  <= '2020-12-31 23:59')  
TimeRangePlot2021 = (df_FCR20_21['DATE_FROM'] >= '2021-01-01 00:00') & (df_FCR20_21['DATE_FROM']  <= '2021-12-31 23:59') 
df_FCR20 = df_FCR20_21[TimeRangePlot2020]
df_FCR21 = df_FCR20_21[TimeRangePlot2021]
df_FCRR2022_raw


### FCR ###

def FCR2019(df_FCRR2019_raw):
    #Average yearly and converting resolution to hourly price 
    Avg2019 = df_FCRR2019_raw.groupby(pd.PeriodIndex(df_FCRR2019_raw['DATE_FROM'],freq="Y"))['AT_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','BE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','CH_SETTLEMENTCAPACITY_PRICE_[EUR/MW]',
                                                                                                'DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]', 'FR_SETTLEMENTCAPACITY_PRICE_[EUR/MW]','NL_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].mean().reset_index()
    #Hourly values
    Avg2019 = Avg2019.iloc[0,1:7]/24
    Avg2019 = Avg2019.values.tolist()


    return Avg2019
Avg2019 = FCR2019(df_FCRR2019_raw) + [0,0] #missing data for last two countries 

def FCR2020(df_FCR20):

    #AUSTRIA 
    list_FCR_AT = df_FCR20['AT_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    del list_FCR_AT[4944:4968]
    del list_FCR_AT[7632:7656]

    list_FCR_AT = [float(i) for i in list_FCR_AT]

    AVG_AT = mean(list_FCR_AT)/24

    #Belgium 

    list_FCR_BE = df_FCR20['BE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    list_FCR_BE = [float(i) for i in list_FCR_BE]
    AVG_BE = mean(list_FCR_BE)/24

    #Chech
    list_FCR_CH = df_FCR20['CH_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    del list_FCR_CH[4944:4968]
    list_FCR_CH = [float(i) for i in list_FCR_CH] 
    AVG_CH = mean(list_FCR_CH)/24

    #Germany 
    list_FCR_DE = df_FCR20['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    list_FCR_DE = [float(i) for i in list_FCR_DE] 
    AVG_DE = mean(list_FCR_DE)/24

    #France
    list_FCR_FR = df_FCR20['FR_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    del list_FCR_FR[4488:4512]
    list_FCR_FR = [float(i) for i in list_FCR_FR] 
    AVG_FR = mean(list_FCR_FR)/24

    #Netherlands

    list_FCR_NL = df_FCR20['NL_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    list_FCR_NL = [float(i) for i in list_FCR_NL] 
    AVG_NL = mean(list_FCR_NL)/24

    Avg2020 = [AVG_AT,AVG_BE,AVG_CH,AVG_DE,AVG_FR,AVG_NL,0,0]

    return Avg2020
Avg2020 = FCR2020(df_FCR20)


def FCR2021(df_FCR21):

    #AUSTRIA 
    list_FCR_AT = df_FCR21['AT_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    del list_FCR_AT[6960:6984]
    del list_FCR_AT[2856:2880] 
    list_FCR_AT = [float(i) for i in list_FCR_AT]

    AVG_AT = mean(list_FCR_AT)/24

    #Belgium 

    list_FCR_BE = df_FCR21['BE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    list_FCR_BE = [float(i) for i in list_FCR_BE]
    AVG_BE = mean(list_FCR_BE)/24

    #Chech
    list_FCR_CH = df_FCR21['CH_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    del list_FCR_CH[6408:6432]
    del list_FCR_CH[2856:2880]
    del list_FCR_CH[1176:1200]
    list_FCR_CH = [float(i) for i in list_FCR_CH] 
    AVG_CH = mean(list_FCR_CH)/24

    #Germany 
    list_FCR_DE = df_FCR21['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    list_FCR_DE = [float(i) for i in list_FCR_DE] 
    AVG_DE = mean(list_FCR_DE)/24

    #France
    list_FCR_FR = df_FCR21['FR_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    list_FCR_FR = [float(i) for i in list_FCR_FR] 
    AVG_FR = mean(list_FCR_FR)/24

    #Netherlands

    list_FCR_NL = df_FCR21['NL_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    list_FCR_NL = [float(i) for i in list_FCR_NL] 
    AVG_NL = mean(list_FCR_NL)/24

    #Switzerland 
    list_FCR_SI = df_FCR21['SI_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    del list_FCR_SI[2856:2880]
    del list_FCR_SI[0:432]
    list_FCR_SI = [float(i) for i in list_FCR_SI] 
    AVG_SI = mean(list_FCR_SI)/24

    #Denmark 
    list_FCR_DK = df_FCR21['DK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    del list_FCR_DK[0:432]
    list_FCR_DK = [float(i) for i in list_FCR_DK] 
    AVG_DK = mean(list_FCR_DK)/24

    Avg2021 = [AVG_AT,AVG_BE,AVG_CH,AVG_DE,AVG_FR,AVG_NL,AVG_SI, AVG_DK]


    return Avg2021
Avg2021 = FCR2021(df_FCR21)

def FCR2022(df_FCRR2022_raw):
    #AUSTRIA 
    list_FCR_AT = df_FCRR2022_raw['AT_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    del list_FCR_AT[1452:1458]
    del list_FCR_AT[1224:1230] 
    del list_FCR_AT[1038:1050]
    del list_FCR_AT[390:396] 
    
    list_FCR_AT = [float(i) for i in list_FCR_AT]
    AVG_AT = mean(list_FCR_AT)/6

    #Belgium 

    list_FCR_BE = df_FCRR2022_raw['BE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    list_FCR_BE = [float(i) for i in list_FCR_BE]
    AVG_BE = mean(list_FCR_BE)/6

    #Chech
    list_FCR_CH = df_FCRR2022_raw['CH_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    del list_FCR_CH[390:396]
    list_FCR_CH = [float(i) for i in list_FCR_CH] 
    AVG_CH = mean(list_FCR_CH)/6


    #Germany 
    list_FCR_DE = df_FCRR2022_raw['DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    list_FCR_DE = [float(i) for i in list_FCR_DE] 
    AVG_DE = mean(list_FCR_DE)/6

    #France
    list_FCR_FR = df_FCRR2022_raw['FR_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist()
    list_FCR_FR = [float(i) for i in list_FCR_FR] 
    AVG_FR = mean(list_FCR_FR)/6

    #Netherlands
    list_FCR_NL = df_FCRR2022_raw['NL_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    list_FCR_NL = [float(i) for i in list_FCR_NL] 
    AVG_NL = mean(list_FCR_NL)/6

    #Switzerland 
    list_FCR_SI = df_FCRR2022_raw['SI_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    del list_FCR_SI[390:396]
    list_FCR_SI = [float(i) for i in list_FCR_SI] 
    AVG_SI = mean(list_FCR_SI)/6

    #Denmark 
    list_FCR_DK = df_FCRR2022_raw['DK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'].tolist() 
    list_FCR_DK = [float(i) for i in list_FCR_DK] 
    AVG_DK = mean(list_FCR_DK)/6

    Avg2022 = [AVG_AT,AVG_BE,AVG_CH,AVG_DE,AVG_FR,AVG_NL,AVG_SI, AVG_DK]



    return Avg2022
Avg2022 = FCR2022(df_FCRR2022_raw)


#Plot FCR average Price (Gruped bar) 
def FCR_Avg_price_plot(Avg2019,Avg2020,Avg2021,Avg2022):


    #Creating DataFrame 
    x_label = ['2019', '2020','2021']
    #df_FCR_AVG = pd.DataFrame([Avg2019,Avg2020,Avg2021,Avg2022],columns=['Austria', 'Belgium','Czech','Germany','France','Netherlands','Switzerland','Denmark'],index = x_label)
    df_FCR_AVG = pd.DataFrame([Avg2019,Avg2020,Avg2021],columns=['Austria', 'Belgium','Czech','Germany','France','Netherlands','Switzerland','Denmark'],index = x_label)


    fig, ax = plt.subplots()
    ax = df_FCR_AVG.plot.bar(color=['firebrick','darkgoldenrod','teal','navy','darkmagenta','darkseagreen','darkviolet','crimson'],rot=20)


    ax.set_ylabel('[€/MW/h]')
    ax.set_title('Average FCR Price')
    plt.legend()
    plt.show()

    return 




### mFRR ### 
def mFRR_Avg_plot(df_DKmFRR_raw):
    

    Avg = df_DKmFRR_raw.groupby(pd.PeriodIndex(df_DKmFRR_raw['HourDK'],freq="Y"))['mFRR_UpPriceEUR'].mean().reset_index()

    x = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']


    fig, ax = plt.subplots(nrows=1,ncols=1)

    #ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
    ax.bar(x , Avg['mFRR_UpPriceEUR'], color='darkgreen',linestyle = '-', label ='mFRR Up Price', linewidth=1)

    ax.set_ylabel('[€/MW/h]')
    ax.legend(loc='upper left')



    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()   
mFRR_Avg_plot(df_DKmFRR_raw)


df_DKmFRR_raw






### aFRR ### 
def aFRR_Mavg_plot():

    Months = ['January','February', 'March', 'April','May','June','July','August','September','October','November','December']

    aFRR_2020avg_price = np.array([30.94758065,48.92473118,51.94892473,54.43548387,56.72043011,51.74731183,49.8655914,61.29032258,42.87634409,32.66129032,45.09408602,43.8172043])
    aFRR_2021avg_price = np.array([35.28225806,30.77956989,30.24193548,29.16666667,48.65591398,53.09139785,36.15591398,29.43548387,42.13709677,45.49731183,66.53225806,41.26344086])
    aFRR_2022avg_price = np.array([145.97,69.81,98.28,109.73,92.51,0,0,0,164.05,0,0,0])

    df_aFRR_Yavg = pd.DataFrame([aFRR_2020avg_price,aFRR_2021avg_price,aFRR_2022avg_price], columns= Months, index=['2020','2021','2022']).T


    fig, ax = plt.subplots(nrows=1,ncols=1)

    #ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
    ax = df_aFRR_Yavg.plot.bar(color=['firebrick','navy','darkseagreen'],rot=20)


    ax.set_ylabel('[€/MW/h]')
    ax.legend(loc='upper right')



    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()   
aFRR_Mavg_plot()

#Finland #NOT DONE YEAT 


df_FIafrr_raw
MontlyAvgFI = df_FIafrr_raw.groupby(pd.PeriodIndex(df_FIafrr_raw.index,freq="M"))['Automatic Frequency Restoration Reserve, price, down','Automatic Frequency Restoration Reserve, price, up'].mean().reset_index()



fig, ax = plt.subplots(nrows=1,ncols=1)

#ax.bar(x, df_Data_plot['SpotPriceEUR,,'], color='b',linestyle = 'solid', label ='Day-Ahead Price')
MontlyAvgFI.set_index('Start time UTC+02:00')

ax = MontlyAvgFI.plot.bar(color=['firebrick','navy','darkseagreen'],rot=20)


ax.set_ylabel('[€/MW/h]')
ax.legend(loc='upper right')
ax.set_xticks(np.arange(len(MontlyAvgFI)), MontlyAvgFI['Start time UTC+02:00'])
ax.set_xticks(ax.get_xticks()[::3])
#loc = plticker.MultipleLocator(1) # this locator puts ticks at regular intervals
#ax.xaxis.set_major_locator(loc)
#ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()   










df2019 = df_solar_prod.loc[TimeRange2019]
df2020 = df_solar_prod.loc[TimeRange2020]

#Grupe by mean so getting average of a periode(e.g. month)
x2019 = df2019.groupby(pd.PeriodIndex(df2019['time'], freq="M"))['P'].mean().reset_index()
x2020 = df2020.groupby(pd.PeriodIndex(df2020['time'], freq="M"))['P'].mean().reset_index()


x2019['time'] = x2019['time'].astype(str)
x2020['time'] = x2020['time'].apply(pd.to_datetime) 

x2019['time'] = x2019['time'].astype(str)
x2020['time'] = x2020['time'].apply(pd.to_datetime) 


## Calculate correlations EXAMPLE 
df_FIafrr2020_raw
df_SEafrr2020_raw
TimeRange2020_FI = (df_FIafrr2020_raw['Start time UTC+02:00'] > '2020-01-01') & (df_FIafrr2020_raw['Start time UTC+02:00']  <= '2020-12-31')

TimeRange2020_SE = (df_SEafrr2020_raw['Period'] > '2020-01-01') & (df_SEafrr2020_raw['Period']  <= '2020-12-31')

df_FI2020 = df_FIafrr2020_raw[TimeRange2020_FI]
df_SE2020 = df_SEafrr2020_raw[TimeRange2020_SE]

df = pd.DataFrame() #Creating empty dataframe


df['Finland Up'] = df_FI2020['Automatic Frequency Restoration Reserve, price, up']
df = df.reset_index(drop=True)
df['Sweden Up'] = df_SE2020['aFRR Upp Pris (EUR/MW)']
matrix = df.corr() 

print(matrix)
sns.heatmap(matrix)

matrix.style.background_gradient(cmap='coolwarm')
plt.show()


#Plot (Gruped bar) 

x_axis = np.arange(12) #Number of  stamps on x axis (Months)
x_label = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August', 'Sept','Okt', 'Nov', 'Dec']
width = 0.3

fig, ax = plt.subplots()

fig1 = ax.bar(x_axis-width,x2019['P'],width = width, label ='Avg Prod 2019') 
fig2 = ax.bar(x_axis,x2020['P'],width = width ,label ='Avg Prod 2020')

ax.set_ylabel('[MW/h]')
ax.set_title('PV Production')
plt.xticks(x_axis,x_label) 
plt.xticks(rotation=25)
plt.legend()
plt.show()




################# DISPLAY FULL DATA FRAME ################

#Displaying the full DAtaFrame 
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

print_full(df_FCR)



#### TEST ### 


# Define a function using lambda
stock = lambda A, amp, angle, phase: A * angle + amp * np.sin(angle + phase)

# Define parameters 
theta = np.linspace(0., 2 * np.pi, 250) # x-axis
np.random.seed(100)
noise = 0.2 * np.random.random(250)
y = stock(.1, .2, theta, 1.2) + noise # y-axis

# Create main container with size of 6x5
fig = plt.figure(figsize=(6, 5))
plt.subplots_adjust(bottom = 0., left = 0, top = 1., right = 1)

# Create first axes, the top-left plot with green plot
sub1 = fig.add_subplot(2,2,1) # two rows, two columns, fist cell
sub1.plot(theta, y, color = 'green')
sub1.set_xlim(1, 2)
sub1.set_ylim(0.2, .5)
sub1.set_ylabel('y', labelpad = 15)

# Create second axes, the top-left plot with orange plot
sub2 = fig.add_subplot(2,2,2) # two rows, two columns, second cell
sub2.plot(theta, y, color = 'orange')
sub2.set_xlim(5, 6)
sub2.set_ylim(.4, 1)

# Create third axes, a combination of third and fourth cell
sub3 = fig.add_subplot(2,2,(3,4)) # two rows, two colums, combined third and fourth cell
sub3.plot(theta, y, color = 'darkorchid', alpha = .7)
sub3.set_xlim(0, 6.5)
sub3.set_ylim(0, 1)
sub3.set_xlabel(r'$\theta$ (rad)', labelpad = 15)
sub3.set_ylabel('y', labelpad = 15)

# Create blocked area in third axes
sub3.fill_between((1,2), 0, 1, facecolor='green', alpha=0.2) # blocked area for first axes
sub3.fill_between((5,6), 0, 1, facecolor='orange', alpha=0.2) # blocked area for second axes

# Create left side of Connection patch for first axes
con1 = ConnectionPatch(xyA=(1, .2), coordsA=sub1.transData, 
                       xyB=(1, .3), coordsB=sub3.transData, color = 'green')
# Add left side to the figure
fig.add_artist(con1)

# Create right side of Connection patch for first axes
con2 = ConnectionPatch(xyA=(2, .2), coordsA=sub1.transData, 
                       xyB=(2, .3), coordsB=sub3.transData, color = 'green')
# Add right side to the figure
fig.add_artist(con2)

# Create left side of Connection patch for second axes
con3 = ConnectionPatch(xyA=(5, .4), coordsA=sub2.transData, 
                       xyB=(5, .5), coordsB=sub3.transData, color = 'orange')
# Add left side to the figure
fig.add_artist(con3)

# Create right side of Connection patch for second axes
con4 = ConnectionPatch(xyA=(6, .4), coordsA=sub2.transData, 
                       xyB=(6, .9), coordsB=sub3.transData, color = 'orange')
# Add right side to the figure
fig.add_artist(con4)

plt.show()

# Save figure with nice margin
plt.savefig('zoom_effect_2.png', dpi = 300, bbox_inches = 'tight', pad_inches = .1)
